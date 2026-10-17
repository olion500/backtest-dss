# infra (Pulumi)

GCP resources for dongpa, as code (`__main__.py`, Python + uv). Nothing personal lives
in the repo: project, region, domain, who may sign in and the state bucket all come
from `.env` (copy `.env.example`), which every `make infra-*` target passes in.

```bash
make infra-preview   # what would change ("unchanged" when nobody edited the console)
make infra-up        # apply
```

Needs the Pulumi CLI (`curl -fsSL https://get.pulumi.com | sh`), gcloud and
application-default credentials (`gcloud auth application-default login`).
The stack holds no secrets, so its passphrase is empty. Stack files (`Pulumi.<stack>.yaml`)
are per person and gitignored.

## Managed here

| Resource | Notes |
|---|---|
| APIs run / artifactregistry / iap / storage | `ABANDON`: dropping one from the program never disables the API |
| Artifact Registry `dongpa` | keep 2 newest, delete after 30 days |
| Cloud Run `$CLOUD_RUN_SERVICE` | env, scaling, IAP, service account. The image is ignored: `make deploy` rolls it |
| Run invoker: allUsers + IAP agent | IAP in front decides who gets in |
| IAP access per `$IAP_MEMBERS` entry | `roles/iap.httpsResourceAccessor` |
| Domain mapping `$VIEWER_DOMAIN` | only when set |
| Bucket `$TRADER_BUCKET` | trader run records (`runs/`); the viewer's service account reads |

Every resource except the IAP access bindings is `protect`ed: `pulumi destroy` refuses until
protection is lifted on purpose. Clearing `VIEWER_DOMAIN` therefore needs
`pulumi state unprotect` on the mapping first; removing an `IAP_MEMBERS` entry just revokes access.

## First time in your own project

1. `cp .env.example .env` and fill in `GCP_PROJECT`, `IAP_MEMBERS` (and `VIEWER_DOMAIN` if you have one).
2. `make infra-init` creates the state bucket (`gs://<project>-pulumi`, versioned) and the stack.
3. `make infra-up` creates the APIs, image repo and bucket. Cloud Run needs an image, so if it
   fails there: `make image-push`, then `make infra-up` again.
4. IAP sign-in needs an OAuth client (below), then `make deploy` for each new image.

If the resources already exist (made by hand), import them instead of creating:
`pulumi import <type> <name> <id> --protect`, then `make infra-preview` until it shows no diff.

## Not managed (set once by hand)

- **IAP OAuth client**: a project without an organization cannot use Google's managed client.
  Create an OAuth client in the console (redirect URI
  `https://iap.googleapis.com/v1/oauth/clientIds/<CLIENT_ID>:handleRedirect`) and link it with
  `gcloud iap settings set` (`accessSettings.oauthSettings`). Its secret has no API.
- **DNS** for `VIEWER_DOMAIN`: CNAME to `ghs.googlehosted.com`. On Cloudflare keep it DNS only
  (a proxied record breaks the managed certificate's renewal).
- **The state bucket** itself (`make infra-init`).
- Anything else in the project that belongs to another app.
