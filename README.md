# Sample Manager cloud

Everyone's NEEDS A LOOK answers teach Sample Manager's sorter, and every plugin gets the improved sorter back.

* A plugin shares an answer **only if its user said yes** (asked once, at the first start; a switch in Settings).
* What one answer holds: a fingerprint of the file (its SHA-256: the same Splice download is the same file for
  everyone), the kind chosen, a hash of the pack's folder name (so the training holds whole packs out, as the app's own
  test bench does), the sound's measurements (numbers), what its name suggested (the **name hint**, 8 numbers: a drum
  hit / 808 / played note / loop / FX, how sure the name is, a drum machine's own name, the pack's name saying drums or
  music; never the name itself) and the training rows the app made of them. **Never** the sound, file names, folders,
  the user's name or anything else about them. A plugin is a random id it made itself. IP addresses are only used, in
  memory, to slow down floods.
* **Scheme 2** (the app's `SortFeatures.Scheme`, since 2026-10-29): each check is gradient-boosted trees
  (`app/trees.py`, scikit-learn) on its own features ++ the whole fingerprint ++ the name hint; the checks that stop a
  name (note, loop) hear the sound alone. The model text carries the decision's bars (`bars ...`). Scheme 1's answers
  (older plugins) are still taken: their rows are scheme 2's own features, joined with the fingerprint and no hint.
* Every night (03:00 UTC), when answers came in, the server trains a candidate on the base (`data/base2.tsv.gz`: the
  developer's sorted library and downloads) plus the shared answers, with the very trainer that made the app's built-in
  model (`tools/builtin.py`: with no answers the candidate *is* the built-in, number for number). Several people who
  answered the same file count once; if they disagree it isn't used. Training takes 5-10 minutes.
* **The gate:** a candidate replaces the model in use only if, on the base's examples held out the same way (the same
  packs in the same 8 groups; the answers never count as the truth here), none of its checks makes more wrong decisions
  than the model in use beyond the rule that makes a check "sure" at all (at most 0.3% wrong of what it decides), none
  decides less than half as many, and none is missing. Copies of the base with flipped answers from three fake plugins
  are rejected (the tests).
* `AUTO_PUBLISH=1` (as on Railway since 2026-10-29: the user wants it to run on its own): every candidate that passes
  the gate is published. Then each plugin tries it with its own decision on what its user answered before it uses it
  (the app's `Cloud.Accepts`): one that would file more of them into another division stays out on that PC.
* Models are **signed** (ECDSA P-256). The plugin only uses a model whose signature matches the public key built into
  it, so nobody else can send it one. It checks for a new model once a day; offline it keeps the last one. A revoked
  model is dropped by the plugins at their next check (they go back to the previous one, or the built-in).
* **The same file** (`POST /v1/kinds`): a plugin whose user shares asks for files at least 3 plugins answered, all
  alike, by the first 4 hex digits of their hashes only (thousands of files share them: the server never learns which
  file a plugin has); the plugin compares the full hashes and sorts its waiting downloads that match (the sound must
  agree). The SANDBOX's answers never count.

## How it runs

Python 3.12, FastAPI, Postgres on Railway (SQLite on a PC for tests). One process; the training runs in a thread.

| Variable | |
|---|---|
| `DATABASE_URL` | Railway: `${{Postgres.DATABASE_URL}}` (a PC: `sqlite:///file.db`, or nothing) |
| `ADMIN_TOKEN` | a long random secret: the admin calls need it (`Authorization: Bearer …`). Without it they are off |
| `AUTO_PUBLISH` | `0` (default): candidates wait for you; `1`: published when they pass the gate (Railway: `1`) |
| `CONSENSUS_MIN` | `3`: how many plugins must answer a file alike before `/v1/kinds` gives it |
| `BASE_FILE`, `BUILTIN_FILE`, `BUILTIN_HELDOUT` | the base, the built-in model and its held-out score (`data/base2.tsv.gz`, `data/builtin2.txt`, `data/builtin2-heldout.json`; the tests use `tests/fixtures/*-small*`) |
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
   `<address>/health` should say `{"ok":true,"scheme":2,"store":"postgres","since":"<when this deploy started>"}`.
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
irm "$cloud/v1/admin/train" -Method Post -Headers $h             # a candidate now (not only at night): on Railway the call
                                                                 # ends 502 after 5 minutes (its proxy); the training goes on: see models
irm "$cloud/v1/admin/models" -Headers $h                         # every model and its gate
(irm "$cloud/v1/admin/models/3" -Headers $h).report              # one model's report: what it decides, the gate
irm "$cloud/v1/admin/models/3/publish" -Method Post -Headers $h  # the plugins get it at their next daily check
irm "$cloud/v1/admin/models/3/revoke" -Method Post -Headers $h   # they drop it again
```

## On a PC

```
python -m venv .venv && .venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\python -m unittest discover -s tests -v        (17 tests, ~8 min: the trees train on a fifth of the base)
.venv\Scripts\uvicorn app.main:app --port 8765               (SQLite in samplemanager-cloud.db)
```

## The base and the features

`data/base2.tsv.gz` = the developer's sorted library + downloads as scheme 2's training rows (each: the check's own
features ++ the fingerprint ++ the name hint), made by the developer build of the app:
`SampleManagerDev.exe --lab-strongbase sorting\library.tsv sorting\downloads.tsv cloud\data\base2.tsv.gz`
(`build.bat dev`). Its first line says the **scheme** (`SortFeatures.Scheme` in the app): what the checks hear. When the
app changes its features it raises the scheme and the base is made again. The app's built-in model and what the gate
compares the first candidate with:
```
python tools/builtin.py data/base2.tsv.gz data/builtin2.txt data/builtin2-heldout.json report.txt
copy data\builtin2.txt ..\src\sortmodel.txt          (then build.bat)
python tools/builtin.py --small data/base2.tsv.gz tests/fixtures/base2-small.tsv.gz tests/fixtures/builtin2-small.txt tests/fixtures/builtin2-small-heldout.json
```
The report's "sure but wrong" lines are the base sounds the held-out checks are surest the base's own label is wrong
about: worth a look (the golden truth: a misfiled sound in the developer's library). Scheme 1's `data/base.tsv.gz`,
`data/builtin.txt` and `tests/fixtures/app-trained.txt` stay for the test that `app/train.py` (the logistic port) trains
exactly what the app's `--lab-train` trains.

## Not yet

* Answers from many fake plugins could push a sound's label (a plugin id costs nothing to make). The gate keeps such a
  model from ever reaching anyone if it does worse on the base; answers about sounds the base doesn't cover can only
  move the checks a little, and every check still needs the name and the sound to agree before it files anything.
* `research/` (needs scikit-learn and the developer's test bench) is how scheme 2 was chosen: the checks' limit was
  the model, not the number of answers.
