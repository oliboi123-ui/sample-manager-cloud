# Sample Manager cloud

Everyone's NEEDS A LOOK answers teach Sample Manager's sorter, and every plugin gets the improved sorter back.

* A plugin shares an answer **only if its user said yes** (asked once, at the first start; a switch in Settings).
* What one answer holds: a fingerprint of the file (its SHA-256: the same Splice download is the same file for
  everyone), the kind chosen, a hash of the pack's folder name (so the training holds whole packs out, as the app's own
  test bench does), the sound's measurements (numbers) and the training rows the app made of them. **Never** the sound,
  file names, folders, the user's name or anything else about them. A plugin is a random id it made itself.
  IP addresses are only used, in memory, to slow down floods.
* Every night (03:00 UTC) the server trains a new candidate on the base (the developer's sorted library and downloads)
  plus the shared answers, with a port of the app's own trainer (`app/train.py`; its test proves it trains exactly what
  the app's `--lab-train` trains). Several people who answered the same file count once; if they disagree it isn't used.
* **The gate:** a candidate replaces the model in use only if, on the base examples, none of its checks decides more
  sounds wrongly by itself, nor decides less than half as many (bad or too few answers). The app's rule "no wrong moves"
  holds for every model it ever gets.
* Publishing is by hand at first (`AUTO_PUBLISH=0`): look at the candidate's report, then publish it. Later
  `AUTO_PUBLISH=1` publishes every candidate that passes the gate.
* Models are **signed** (ECDSA P-256). The plugin only uses a model whose signature matches the public key built into
  it, so nobody else can send it one. It checks for a new model once a day; offline it keeps working as before. A revoked
  model is dropped by the plugins at their next check (they go back to the previous one, or the built-in).

## How it runs

Python 3.12, FastAPI, Postgres on Railway (SQLite on a PC for tests). One process; the training runs in a thread.

| Variable | |
|---|---|
| `DATABASE_URL` | Railway: `${{Postgres.DATABASE_URL}}` (a PC: `sqlite:///file.db`, or nothing) |
| `ADMIN_TOKEN` | a long random secret: the admin calls need it (`Authorization: Bearer …`). Without it they are off |
| `AUTO_PUBLISH` | `0` (default): candidates wait for you; `1`: published when they pass the gate |
| `TRAIN_HOUR_UTC` | `3` |
| `MODEL_SIGNING_KEY` | optional: your own key (`python tools/make_key.py`). Without it the server makes one on its first start and keeps it in the database (`GET /v1/public-key` gives the half the plugin needs) |
| `MAX_PER_INSTALL` | `5000`: at most this many answers of one plugin are used |

## Putting it on Railway (from GitHub)

1. **GitHub:** create a new **private** repository (e.g. `sample-manager-cloud`), empty (no README). Then, in this
   folder (`Documents\Sample Manager\cloud`, already a git repository with everything committed):
   ```
   git remote add origin https://github.com/<you>/sample-manager-cloud.git
   git push -u origin main
   ```
2. **Railway:** New Project → **Deploy from GitHub repo** → pick it. Then in the project: **+ Create → Database →
   PostgreSQL**.
3. In the service's **Variables**: `DATABASE_URL` = `${{Postgres.DATABASE_URL}}`, `ADMIN_TOKEN` = a long random
   string (PowerShell: `[guid]::NewGuid().ToString('N') + [guid]::NewGuid().ToString('N')`), `AUTO_PUBLISH` = `0`.
4. **Settings → Networking → Generate Domain**: the address, e.g. `https://sample-manager-cloud.up.railway.app`.
   `<address>/health` should say `{"ok":true,"scheme":1,"store":"postgres","since":"<when this deploy started>"}`.
   On Railway the service won't start without Postgres (a SQLite file, its answers and its signing key would be lost at
   every redeploy): the deploy fails with "DATABASE_URL is missing…" in its log and the one before keeps running. If the
   database service isn't called `Postgres`, `${{Postgres.DATABASE_URL}}` is empty: pick it with **Add Reference**.
   Railway holds variable changes as *staged changes* until you press **Deploy**.
5. Give the address to whoever builds the plugin: it goes into `src\config.default.json` (`Cloud.Url`) with the key
   from `<address>/v1/public-key` (`Cloud.PublicKey`). Until then the plugin never asks to share and never connects.

The service keeps its data in Postgres (answers, models, its signing key). Pushing to `main` doesn't deploy by itself
(the service has no GitHub trigger; to add one: the service's **Settings → Source**, branch `main`):
`python tools/railway.py deploy` deploys the newest commit and waits for it.

From a PC, `python tools/railway.py look` shows the services, which variables are set (secret values masked) and the
recent deploys; `python tools/railway.py logs <deployment id>` a deploy's log; `deploy` a commit. It needs a project token (project
**Settings → Tokens**, for `production`) saved in `%USERPROFILE%\.railway-token`.

## The admin calls (PowerShell)

```powershell
$cloud = "https://<address>"; $h = @{ Authorization = "Bearer <ADMIN_TOKEN>" }
irm "$cloud/v1/admin/stats" -Headers $h                          # plugins, answers, sounds, the model in use
irm "$cloud/v1/admin/train" -Method Post -Headers $h             # a candidate now (not only at night)
irm "$cloud/v1/admin/models" -Headers $h                         # every model and its gate
(irm "$cloud/v1/admin/models/3" -Headers $h).report              # one model's report: what it decides, the gate
irm "$cloud/v1/admin/models/3/publish" -Method Post -Headers $h  # the plugins get it at their next daily check
irm "$cloud/v1/admin/models/3/revoke" -Method Post -Headers $h   # they drop it again
```

## On a PC

```
python -m venv .venv && .venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\python -m unittest discover -s tests -v        (12 tests: the trainer = the app's, the API, the gate)
.venv\Scripts\uvicorn app.main:app --port 8765               (SQLite in samplemanager-cloud.db)
```

## The base and the features

`data/base.tsv.gz` = the developer's sorted library + downloads as training rows, made by the developer build of the
app: `SampleManagerDev.exe --lab-cloudbase sorting\library.tsv sorting\downloads.tsv cloud\data\base.tsv.gz`
(`build.bat dev`). Its first line says the **scheme** (`SortFeatures.Scheme` in the app): what the checks hear. Answers
and models of another scheme are never mixed; when the app changes its features it raises the scheme, the base is made
again, and older plugins keep the models of theirs. `data/builtin.txt` = the app's built-in model (`src\sortmodel.txt`):
what the gate compares the first candidate with. `tests/fixtures/app-trained.txt` = what the app's `--lab-train`
trains from the same base (the test that the port trains the same).

## Not yet

* Answers from many fake plugins could push a sound's label (a plugin id costs nothing to make). The gate keeps such a
  model from ever reaching anyone if it does worse on the base; answers about sounds the base doesn't cover can only
  move the checks a little, and every check still needs the name and the sound to agree before it files anything.
* Sorting a download by what others answered for the very same file (its hash) is the next step.
* **A stronger model (scheme 2).** `research/` (needs scikit-learn and the developer's test bench) shows that the
  checks' limit is the model, not the number of answers: more honest answers make today's logistic checks decide
  *less* of unseen downloads, while gradient-boosted trees on each check's features + the whole fingerprint decide 2–3×
  as much at under 0.3% wrong, and only with the downloads' answers (on the library alone their bar doesn't carry over
  to downloads). Every answer already carries the fingerprint, so what is shared now counts for it.
