# GitHub and Render deployment guide

This repository is a local-first educational research prototype. The steps below publish the source and deploy a **public demo** with a React static site and a FastAPI web service. They do not turn it into a clinical system.

> **Hosted-demo data warning:** when deployed, the browser sends uploaded 2D images to the Render API. Use only public, de-identified sample images. Do not upload patient scans, names, contact details, or other personal information. The project is not designed or validated for clinical use.

## What is and is not in Git

The current project folder `D:\BTS` is not yet a Git repository. The existing `.gitignore` excludes `.env`, the BRISC dataset and manifest, model weights and metadata, evaluation JSON, Python/Node environments, and generated notebook outputs. Those files are not transferred by a normal source-only GitHub push.

The deployed app can start without the dataset and checkpoints, but its model status will be **missing**, so it will not produce classifier, mask, or Grad-CAM results. The admin evaluation page also needs the held-out data, manifest, model packages, and recorded report files; those are not part of the source push.

## 1. Prepare the project before the first push

1. Open `D:\BTS` in VS Code and use **Terminal → New Terminal**.
2. Confirm `.env.example` contains only the non-working password placeholder. Create or keep your actual `.env` locally; it must remain untracked. Use a new, long password for any hosted demo.
3. If a real credential has ever been pushed to GitHub, changing the file is not enough: rotate that credential immediately and follow GitHub's sensitive-data removal process.
4. Check that the important exclusions apply:

```powershell
git check-ignore -v .env data\datasets\brisc2025\ data\manifests\brisc.csv assets\models\classifier\model.pt
```

Some of these paths may not exist in another checkout. Each existing sensitive or large local artifact should be reported as ignored. The public BRISC images, local manifest, model weights, and patient information should not be committed.

## 2. Create a GitHub repository and push the source

Create a **private** empty repository on GitHub. Do not initialize it with a README, `.gitignore`, or license, because this project already has those files. GitHub's documented import process is [here](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github).

From PowerShell in `D:\BTS`, initialize Git, review the staged file list, then commit:

```powershell
git init -b main
git add -A
git status --short
git diff --cached --stat
```

Review the `git status` output. It should show source and documentation, and should not show `.env`, data, model weights, `node_modules`, or `.venv`. If anything private or too large is staged, unstage it with `git restore --staged -- <path>` and check `.gitignore` before continuing.

Then commit and connect the empty GitHub repository. Replace the URL with the one GitHub gives you:

```powershell
git commit -m "Initial NEURO/LAB research workstation"
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git remote -v
git push -u origin main
```

For later updates:

```powershell
git add -A
git status --short
git commit -m "Describe the change"
git push
```

Keep the repo private until you have reviewed the entire staged file list and confirmed no credentials, patient data, BRISC files, or private checkpoints are included.

## 3. Deploy the FastAPI service on Render

The backend is in the repository root as `backend.app.main:app`. Render's FastAPI guide uses a Python web service bound to `0.0.0.0` and the service-provided `$PORT`; see [Deploy a FastAPI app](https://render.com/docs/deploy-fastapi).

1. Sign in to Render and choose **New → Web Service**.
2. Connect the GitHub repository and select branch `main`.
3. Leave the root directory blank so Render uses the repository root.
4. Set:

| Render field | Value |
|---|---|
| Runtime | Python |
| Build command | `pip install -r backend/requirements.txt` |
| Start command | `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT` |
| Health check path | `/api/health` |

5. In the Render service's **Environment** settings, add:

| Key | Value |
|---|---|
| `BTDS_ADMIN_USERNAME` | A new administrator username |
| `BTDS_ADMIN_PASSWORD` | A unique, long random password; never reuse the local password |
| `BTDS_CORS_ORIGINS` | The exact frontend origin, added after the static site is created |

Do not paste or upload the local `.env` file. Enter only the required values in Render's protected Environment settings. Render documents environment-variable setup [here](https://render.com/docs/configure-environment-variables).

Deploy the service. Its API address will be similar to `https://neurolab-api.onrender.com`. Confirm that `https://YOUR-API.onrender.com/api/health` returns JSON before creating the frontend.

### Model files for inference (optional)

For a working classifier/segmenter, supply each compatible model package (`model.json` and its named weights file) separately. Do not put model weights in Git. If using Render storage, attach a persistent disk to the API service, mount it at `/var/data`, set `BTDS_ASSET_ROOT=/var/data/assets`, and place the packages at:

```text
/var/data/assets/models/classifier/model.json
/var/data/assets/models/classifier/model.pt
/var/data/assets/models/segmenter/model.json
/var/data/assets/models/segmenter/model.pt
```

Install the matching CPU-only PyTorch and torchvision builds if the service will run inference; this project does not require CUDA on Render. A persistent disk is needed if model files are placed on the service filesystem and must survive redeploys; without it, Render's filesystem is ephemeral. Persistent disks are available for paid web services and have specific mount-path limitations, described in [Render's disk documentation](https://render.com/docs/disks).

Even with inference packages, the current Render deployment will not automatically have the BRISC test images, manifest, or local evaluation reports. The admin's test-evaluation action therefore will not have its local inputs until a separate, secure dataset-storage and path configuration is set up. Do not publish the dataset or test labels as a shortcut.

## 4. Deploy the React frontend as a Render Static Site

The Vite build uses `frontend/package-lock.json`, so use `npm ci`. For Vite, Render publishes the `dist` folder; Render's current static-site guidance lists these settings [here](https://render.com/tutorials/web-service-vs-static-site/static-sites).

1. In Render, choose **New → Static Site** and connect the same GitHub repository, branch `main`.
2. Set the root directory to `frontend`.
3. Set build command to `npm ci && npm run build` and publish directory to `dist`.
4. Add these build environment variables, substituting the API service URL:

| Key | Value |
|---|---|
| `VITE_API_BASE_URL` | `https://YOUR-API.onrender.com/api` |
| `VITE_PUBLIC_DEMO` | `true` |

Vite inserts these values when it builds the site, so changing either value requires a new frontend build/deploy. Do not put secrets in `VITE_` variables; frontend variables are public in the generated JavaScript.

5. After the static site has a URL, return to the API service's Environment settings and set `BTDS_CORS_ORIGINS` to that exact origin, for example `https://neurolab-demo.onrender.com` (no trailing slash). Save and redeploy the API.
6. If you use a custom domain, set the custom frontend origin in `BTDS_CORS_ORIGINS` and rebuild if `VITE_API_BASE_URL` changes.

The frontend uses `/activity` and `/evaluation` paths. In the Static Site's **Redirects/Rewrites**, add a rewrite from `/*` to `/index.html` so a browser refresh on those paths serves the React app.

## 5. Verify the hosted demo

1. Open the frontend URL. Its patient page should say that images are transmitted to the hosted demo and recommend public, de-identified sample data.
2. Open the API health URL. Without separately supplied model files, both model statuses should say missing; this is expected for a source-only deploy.
3. In browser Developer Tools → **Network**, check that API requests go to the Render API URL, not `127.0.0.1`.
4. If the browser reports a CORS error, check `BTDS_CORS_ORIGINS` against the exact frontend URL, save/redeploy the API, and rebuild the frontend after API URL changes.
5. Test administrator sign-in with the new Render-only credential. Do not place it in a screenshot, source file, or presentation.

## Deployment limitations to keep in mind

- This is a research demonstration, not a clinical service. Use only public, de-identified sample images.
- By default, data, model files, and evaluation reports are absent from the Git push. A source-only deployment will start but will not perform model inference.
- Uploaded 2D scans are sent to the hosted API. NIfTI preview is browser-side in the current interface. Never upload identifiable patient records.
- Admin sessions and activity counters live in API process memory and reset when that process restarts. Do not scale to multiple API instances expecting shared sessions or counts.
- Render's default service filesystem is ephemeral. Data that must persist needs a suitable persistent storage plan and explicit path configuration.
- No database, patient account system, rate limiting, clinical validation, external-site evaluation, or production security review is provided by this prototype.
- The frontend static site's `VITE_` settings are public build-time values; keep passwords and tokens only in the API service's private environment settings.

## Project-specific deployment changes

This guide's setup depends on the following configuration in this repository:

- `frontend/src/api.ts` reads `VITE_API_BASE_URL` and retains the local API URL as the development default.
- `VITE_PUBLIC_DEMO=true` changes the UI's upload/privacy messages for a hosted build.
- `BTDS_CORS_ORIGINS` configures the API's permitted browser origins.
- `.env.example` contains a non-working placeholder, not a login password.

After pushing the project, Render can watch the selected GitHub branch and deploy new commits automatically. Review each service's deploy log after pushes; see [Render deploys](https://render.com/docs/deploys).
