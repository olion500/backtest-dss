"""dongpa on GCP: the Cloud Run viewer, its image repo and domain, and the trader's run records.

Every resource is ``protect``ed, so a stray ``pulumi destroy`` cannot take the site down.
Not managed here (set once by hand, see infra/README.md): the IAP OAuth client and
its ``gcloud iap settings``, the DNS record, and the Pulumi state bucket.

Nothing personal is written here: project, domain and who may sign in come from
the environment, which ``make infra-*`` fills from ``.env`` (see ``.env.example``).

Images: ``make deploy`` pushes a new image and rolls a revision with gcloud, so the
container image and gcloud's client stamp are ignored below. Everything else about
the service (env, scaling, IAP, IAM) changes only through this program.
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
    for name in ("run", "artifactregistry", "iap", "storage")
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

pulumi.export("viewer_url", viewer.uri)
pulumi.export("trader_runs", TRADER_RUNS)
