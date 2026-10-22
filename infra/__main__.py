"""dongpa on GCP: the Cloud Run viewer, its image repo and domain, the trader's run records,
and the auto-trading jobs (Cloud Run Jobs on a Cloud Scheduler clock).

Every resource is ``protect``ed, so a stray ``pulumi destroy`` cannot take the site down.
Not managed here (set once by hand, see infra/README.md): the IAP OAuth client and
its ``gcloud iap settings``, the DNS record, and the Pulumi state bucket.

Nothing personal is written here: project, domain and who may sign in come from
the environment, which ``make infra-*`` fills from ``.env`` (see ``.env.example``).

Images: ``make deploy`` pushes a new image and rolls a revision with gcloud, so the
container image and gcloud's client stamp are ignored below. Everything else about
the service (env, scaling, IAP, IAM) changes only through this program.
The trader jobs work the same way with ``make trader-deploy``.

Broker keys: ``make trader-secrets`` creates one Secret Manager secret per key in .env
(``MERITZ_MAIN_APP_KEY`` → ``meritz-main-app-key``) and adds the value with gcloud, so
no key ever passes through Pulumi's state. This program only names the secrets
(``TRADER_SECRET_ENVS``) and lets the trader's service account read them.
"""

import os
import re

import pulumi
import pulumi_gcp as gcp


def env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    if default is None:
        raise SystemExit(f"{name} is not set: fill it in .env (see .env.example) and run through make infra-*")
    return default


PROJECT = env("GCP_PROJECT")
REGION = env("GCP_REGION", "asia-northeast1")
SERVICE = env("CLOUD_RUN_SERVICE", "dongpa-viewer")
DOMAIN = env("VIEWER_DOMAIN", "")          # empty: no custom domain mapping
IAP_MEMBERS = [m.strip() for m in env("IAP_MEMBERS", "").split(",") if m.strip()]  # e.g. user:me@example.com
TRADER_BUCKET = env("TRADER_BUCKET", f"{PROJECT}-dongpa-trader")
RATE_LIMIT = env("DONGPA_RATE_LIMIT", "12")
TRADER_IMAGE = env("TRADER_IMAGE", f"{REGION}-docker.pkg.dev/{PROJECT}/dongpa/dongpa-trader:latest")
TRADER_SECRET_ENVS = [name.strip() for name in env("TRADER_SECRET_ENVS", "").split(",") if name.strip()]
DATA_TOKEN_ENV = "GITHUB_DATA_TOKEN"
BROKER_SECRET_ENVS = [name for name in TRADER_SECRET_ENVS if name != DATA_TOKEN_ENV]
DATA_REPO = env("DONGPA_DATA_REPO", "")    # owner/name; empty: no data job, the jobs read the app's default data source
DATA_ACCOUNT = env("DONGPA_DATA_ACCOUNT", "main")
DATA_URL = f"https://raw.githubusercontent.com/{DATA_REPO}/main/data" if DATA_REPO else ""

project_number = gcp.organizations.get_project(project_id=PROJECT).number
VIEWER_SA = f"{project_number}-compute@developer.gserviceaccount.com"
IAP_AGENT = f"serviceAccount:service-{project_number}@gcp-sa-iap.iam.gserviceaccount.com"
SERVICE_PATH = f"projects/{PROJECT}/locations/{REGION}/services/{SERVICE}"

keep = pulumi.ResourceOptions(protect=True)

# --- APIs (ABANDON: removing one from this program must never switch the API off) ---
apis = {
    name: gcp.projects.Service(
        f"api-{name}", project=PROJECT, service=f"{name}.googleapis.com", deletion_policy="ABANDON", opts=keep,
    )
    for name in ("run", "artifactregistry", "iap", "storage", "secretmanager", "cloudscheduler")
}

# --- container images ---
images = gcp.artifactregistry.Repository(
    "images",
    project=PROJECT,
    location=REGION,
    repository_id="dongpa",
    description="dongpa viewer images",
    format="DOCKER",
    cleanup_policies=[
        {"id": "delete-older-than-30d", "action": "DELETE",
         "condition": {"older_than": "2592000s", "tag_state": "ANY"}},
        {"id": "keep-recent-2", "action": "KEEP", "most_recent_versions": {"keep_count": 2}},
    ],
    deletion_policy="DELETE",
    opts=keep,
)

# --- trader run records (python -m trader writes, the viewer reads) ---
trader_runs = gcp.storage.Bucket(
    "trader-runs",
    name=TRADER_BUCKET,
    project=PROJECT,
    location=REGION,
    uniform_bucket_level_access=True,
    public_access_prevention="enforced",
    opts=keep,
)
gcp.storage.BucketIAMMember(
    "trader-runs-viewer-read",
    bucket=trader_runs.name,
    role="roles/storage.objectViewer",
    member=f"serviceAccount:{VIEWER_SA}",
)
TRADER_RUNS = trader_runs.name.apply(lambda name: f"gs://{name}/runs")

# --- the viewer ---
viewer = gcp.cloudrunv2.Service(
    "viewer",
    project=PROJECT,
    location=REGION,
    name=SERVICE,
    ingress="INGRESS_TRAFFIC_ALL",
    iap_enabled=True,
    launch_stage="GA",
    scaling={"max_instance_count": 20},
    template={
        "service_account": VIEWER_SA,
        "timeout": "120s",
        "max_instance_request_concurrency": 80,
        "scaling": {"max_instance_count": 1},
        "containers": [{
            "image": env("VIEWER_IMAGE", f"{REGION}-docker.pkg.dev/{PROJECT}/dongpa/{SERVICE}:latest"),
            "envs": [
                {"name": "DONGPA_RATE_LIMIT", "value": RATE_LIMIT},
                {"name": "DONGPA_TRADER_RUNS", "value": TRADER_RUNS},
            ],
            "ports": {"container_port": 8080, "name": "http1"},
            "resources": {"cpu_idle": True, "startup_cpu_boost": True, "limits": {"cpu": "1", "memory": "1Gi"}},
            "startup_probe": {"failure_threshold": 1, "period_seconds": 240, "timeout_seconds": 240,
                              "tcp_socket": {"port": 8080}},
        }],
    },
    traffics=[{"percent": 100, "type": "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"}],
    deletion_policy="DELETE",
    opts=pulumi.ResourceOptions(
        protect=True,
        ignore_changes=["template.containers[0].image", "client", "clientVersion"],
    ),
)

# allUsers may reach the service; IAP in front decides who gets through.
gcp.cloudrunv2.ServiceIamMember(
    "viewer-invoker-public", project=PROJECT, location=REGION, name=SERVICE_PATH,
    role="roles/run.invoker", member="allUsers", opts=keep,
)
gcp.cloudrunv2.ServiceIamMember(
    "viewer-invoker-iap", project=PROJECT, location=REGION, name=SERVICE_PATH,
    role="roles/run.invoker", member=IAP_AGENT, opts=keep,
)
# Who gets through IAP: one binding per IAP_MEMBERS entry. Not protected, so taking
# someone out of IAP_MEMBERS revokes their access on the next `make infra-up`.
for member in IAP_MEMBERS:
    gcp.iap.WebCloudRunServiceIamMember(
        "viewer-iap-" + re.sub(r"[^a-z0-9]+", "-", member.lower()).strip("-"),
        project=PROJECT,
        location=REGION,
        cloud_run_service_name=f"projects/{PROJECT}/iap_web/cloud_run-{REGION}/services/{SERVICE}",
        role="roles/iap.httpsResourceAccessor",
        member=member,
    )

if DOMAIN:
    gcp.cloudrun.DomainMapping(
        "viewer-domain",
        project=PROJECT,
        location=REGION,
        name=DOMAIN,
        metadata={"namespace": PROJECT},
        spec={"route_name": viewer.name, "certificate_mode": ""},
        deletion_policy="DELETE",
        opts=keep,
    )

# --- auto trading: one job per step, run by Cloud Scheduler on NY time ---
trader_sa = gcp.serviceaccount.Account(
    "trader-sa", project=PROJECT, account_id="dongpa-trader", display_name="dongpa auto trading jobs", opts=keep,
)
trader_member = trader_sa.email.apply(lambda email: f"serviceAccount:{email}")
# The jobs check secret access when they are created: they wait for these grants.
trader_grants = [gcp.storage.BucketIAMMember(
    "trader-runs-trader-write", bucket=trader_runs.name, role="roles/storage.objectUser", member=trader_member,
)]
for name in TRADER_SECRET_ENVS:
    trader_grants.append(gcp.secretmanager.SecretIamMember(
        "trader-secret-" + name.lower().replace("_", "-"),
        project=PROJECT,
        secret_id=name.lower().replace("_", "-"),
        role="roles/secretmanager.secretAccessor",
        member=trader_member,
        opts=pulumi.ResourceOptions(depends_on=[apis["secretmanager"]]),
    ))

def secret_env(name: str) -> dict:
    return {"name": name, "value_source": {"secret_key_ref": {"secret": name.lower().replace("_", "-"), "version": "latest"}}}


TRADER_ENVS = [{"name": "DONGPA_TRADER_RUNS", "value": TRADER_RUNS}, *map(secret_env, BROKER_SECRET_ENVS)]
if DATA_URL:
    TRADER_ENVS.append({"name": "DONGPA_DATA_URL", "value": DATA_URL})
# Each step decides for itself whether it is time (holidays, early closes, its window)
# and records "closed" when it is not, so the clock only has to wake it.
# step: (schedule, args, envs, timeout)
TRADER_STEPS = {
    "orders": ("45 9 * * 1-5", ["orders", "--all", "--auto"], TRADER_ENVS, "240s"),
    # 12:45 is the stop check on early-close days; on a regular day it records "closed".
    "stop": ("45 12,15 * * 1-5", ["stop", "--all", "--auto"], TRADER_ENVS, "240s"),
}
if DATA_REPO and DATA_TOKEN_ENV in TRADER_SECRET_ENVS:
    # In session (the broker's previous close is final then), ten minutes before orders read data/.
    TRADER_STEPS["data"] = ("35 9 * * 1-5", ["data", "--account", DATA_ACCOUNT, "--push"], [
        *map(secret_env, TRADER_SECRET_ENVS), {"name": "DONGPA_DATA_REPO", "value": DATA_REPO},
    ], "600s")
for step, (schedule, args, envs, timeout) in TRADER_STEPS.items():
    job = gcp.cloudrunv2.Job(
        f"trader-{step}",
        project=PROJECT,
        location=REGION,
        name=f"dongpa-trader-{step}",
        launch_stage="GA",
        template={
            "task_count": 1,
            "template": {
                "service_account": trader_sa.email,
                # A rerun never sends twice (open orders halt the check), but nothing should rerun on its own.
                "max_retries": 0,
                "timeout": timeout,
                "containers": [{
                    "image": TRADER_IMAGE,
                    "args": args,
                    "envs": envs,
                    "resources": {"limits": {"cpu": "1", "memory": "1Gi"}},
                }],
            },
        },
        deletion_policy="DELETE",
        opts=pulumi.ResourceOptions(
            protect=True,
            depends_on=[apis["run"], *trader_grants],
            ignore_changes=["template.template.containers[0].image", "client", "clientVersion"],
        ),
    )
    gcp.cloudrunv2.JobIamMember(
        f"trader-{step}-scheduler", project=PROJECT, location=REGION, name=job.name,
        role="roles/run.invoker", member=trader_member,
    )
    gcp.cloudscheduler.Job(
        f"trader-{step}-clock",
        project=PROJECT,
        region=REGION,
        name=f"dongpa-trader-{step}",
        schedule=schedule,
        time_zone="America/New_York",
        attempt_deadline="60s",
        retry_config={"retry_count": 0},
        http_target={
            "http_method": "POST",
            "uri": job.name.apply(
                lambda name: f"https://run.googleapis.com/v2/projects/{PROJECT}/locations/{REGION}/jobs/{name}:run"),
            "oauth_token": {"service_account_email": trader_sa.email},
        },
        opts=pulumi.ResourceOptions(protect=True, depends_on=[apis["cloudscheduler"]]),
    )

pulumi.export("viewer_url", viewer.uri)
pulumi.export("trader_runs", TRADER_RUNS)
