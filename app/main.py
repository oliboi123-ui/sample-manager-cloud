"""Sample Manager's cloud: everyone's NEEDS A LOOK answers (shared only with their permission) teach the sorter, and the
plugins get the improved checks back, signed. See README.md.

    POST   /v1/answers               a plugin's answers (batched; kind null = taken back)
    GET    /v1/model?have=&scheme=   the newest published model for these features (version 0 = use the built-in)
    POST   /v1/kinds                 what several people answered for the very same files (by the first 4 hex digits of
                                     their hashes only: the server never learns which file a plugin has)
    DELETE /v1/installs/{id}         forget everything a plugin shared
    GET    /v1/public-key            the key the plugin checks models with
    GET    /health
    admin (Authorization: Bearer ADMIN_TOKEN):
    POST   /v1/admin/train           train a candidate now (else: every night at TRAIN_HOUR_UTC when answers came in)
    GET    /v1/admin/models          every model: candidate / rejected / published / revoked, and its gate
    GET    /v1/admin/models/{v}      one, with its report and text
    POST   /v1/admin/models/{v}/publish   (a rejected one only with ?force=1)
    POST   /v1/admin/models/{v}/revoke
    GET    /v1/admin/stats
"""
from __future__ import annotations

import asyncio
import hmac
import json
import math
import os
import re
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from . import base, sign, train, trees
from .store import Store

HERE = Path(__file__).resolve().parent.parent


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


class Settings:
    def __init__(self) -> None:
        self.database_url = env("DATABASE_URL")
        self.admin_token = env("ADMIN_TOKEN")
        self.signing_key = env("MODEL_SIGNING_KEY")
        self.auto_publish = env("AUTO_PUBLISH", "0") == "1"
        self.train_hour = int(env("TRAIN_HOUR_UTC", "3"))
        # scheme 2 (the app's SortFeatures.Scheme): the base, the app's built-in model and its held-out score on the base
        self.base_file = env("BASE_FILE", str(HERE / "data" / "base2.tsv.gz"))
        self.builtin_file = env("BUILTIN_FILE", str(HERE / "data" / "builtin2.txt"))
        self.builtin_heldout = env("BUILTIN_HELDOUT", str(HERE / "data" / "builtin2-heldout.json"))
        self.max_per_install = int(env("MAX_PER_INSTALL", "5000"))
        self.consensus = int(env("CONSENSUS_MIN", "3"))
        self.hosted = any(env(n) for n in ("RAILWAY_PROJECT_ID", "RAILWAY_ENVIRONMENT_ID", "RAILWAY_ENVIRONMENT"))


# what the plugins may send (src/App/Cloud.cs; the kinds: src/Classifier.cs SType)
KINDS = {"Kick", "Bass808", "Snare", "Clap", "Snap", "Rim", "Hat", "OpenHat", "Cymbal", "Perc", "Tom", "Fill", "LoopDrum",
         "LoopPerc", "LoopHat", "LoopMelodic", "LoopGuitar", "LoopBass", "LoopVinyl", "Stem", "Chord", "Pad", "Bass", "Guitar",
         "OneShot", "Vox", "VocalPhrase", "FxRiser", "FxImpact", "FxAtmos", "Fx", "Midi", "Inspiration"}
FP_COLUMNS = 64  # Fingerprint.Columns
SCHEME = 2       # what this server trains; it takes scheme 1's answers too (their rows are scheme 2's own features)
ACCEPTS = (1, 2)
MAX_ANSWERS, MAX_ROWS, MAX_BODY, MAX_PREFIXES = 200, 20, 2_000_000, 500
HEX64, HEX16, HEX4 = re.compile(r"^[0-9a-f]{64}$"), re.compile(r"^[0-9a-f]{16}$"), re.compile(r"^[0-9a-f]{4}$")
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class RowIn(BaseModel):
    check: str
    label: int
    x: list[float]


class AnswerIn(BaseModel):
    hash: str
    pack: str = ""
    kind: str | None = None
    fp: list[float] = []
    hint: list[float | None] = []   # scheme 2: what the name suggested (Sorter.Hint; null = nothing to go by), never the name
    rows: list[RowIn] = []


class BatchIn(BaseModel):
    install: str
    app: str = ""
    scheme: int
    answers: list[AnswerIn]


class KindsIn(BaseModel):
    prefixes: list[str]


class Cloud:
    """Everything the service holds: settings, storage, the base examples, the signing key, the training lock."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or Settings()
        # on Railway a SQLite file is lost at every redeploy, and with it the signing key the plugins trust and every
        # answer: refuse to start (the deploy fails, the one before keeps running) instead of starting over unnoticed
        if self.s.hosted and not self.s.database_url.startswith("postgres"):
            raise RuntimeError("DATABASE_URL is missing or not a Postgres address: in the service's Variables set "
                               "DATABASE_URL = ${{Postgres.DATABASE_URL}} (Add Reference), then deploy")
        self.started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.store = Store(self.s.database_url or None)
        self.scheme, self.base = base.load(self.s.base_file, arrays=True)
        if self.scheme != SCHEME:
            raise RuntimeError(f"{self.s.base_file} is scheme {self.scheme}: this server trains scheme {SCHEME} (--lab-strongbase)")
        # each check's own features (a row: own ++ fingerprint ++ name hint): what a shared answer's rows carry
        self.dims: dict[str, int] = {}
        for r in self.base:
            self.dims.setdefault(r.check, len(r.x) - trees.FP - trees.HINT)
        self.builtin = Path(self.s.builtin_file).read_text(encoding="utf-8") if Path(self.s.builtin_file).exists() else ""
        self.builtin_heldout = json.loads(Path(self.s.builtin_heldout).read_text(encoding="utf-8")) if Path(self.s.builtin_heldout).exists() else {}
        # the decision's bars every model carries (the built-in's: the app was measured with them)
        self.bars = next((line for line in self.builtin.split("\n") if line.startswith("bars ")), trees.BARS)
        # the signing key: MODEL_SIGNING_KEY if set, else the one this server made itself the first time (kept in the
        # database, so nobody has to handle it; its public half: GET /v1/public-key, built into the plugin)
        if self.s.signing_key:
            self.key = sign.load_key(self.s.signing_key)
        else:
            kept = self.store.get("signing_key")
            if not kept:
                kept = sign.key_text(sign.new_key())
                self.store.set("signing_key", kept)
            self.key = sign.load_key(kept)
        self.training = threading.Lock()
        self.limits: dict[str, deque] = {}
        self.limit_lock = threading.Lock()

    # ------------------------------------------------------------ limits

    def allow(self, who: str, per_minute: int) -> bool:
        now = time.monotonic()
        with self.limit_lock:
            q = self.limits.setdefault(who, deque())
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= per_minute:
                return False
            q.append(now)
            if len(self.limits) > 50_000:  # (forget the quiet ones)
                for k in [k for k, v in self.limits.items() if not v or now - v[-1] > 60]:
                    del self.limits[k]
            return True

    # ------------------------------------------------------------ answers

    def check_batch(self, b: BatchIn) -> list[dict]:
        if not UUID.match(b.install):
            raise HTTPException(400, "install: a random UUID, lower case")
        if b.scheme not in ACCEPTS:
            raise HTTPException(409, f"features scheme {b.scheme} is not one this server takes ({ACCEPTS}): keep them for a newer one")
        if len(b.app) > 32 or len(b.answers) > MAX_ANSWERS:
            raise HTTPException(400, f"at most {MAX_ANSWERS} answers at a time")
        out = []
        for a in b.answers:
            if not HEX64.match(a.hash):
                raise HTTPException(400, "hash: SHA-256 of the file, 64 hex digits")
            if a.kind is None:
                out.append({"hash": a.hash, "kind": None})
                continue
            if a.kind not in KINDS:
                raise HTTPException(400, f"kind {a.kind!r} unknown")
            if not HEX16.match(a.pack):
                raise HTTPException(400, "pack: 16 hex digits")
            if len(a.fp) != FP_COLUMNS or not _finite(a.fp):
                raise HTTPException(400, f"fp: {FP_COLUMNS} finite numbers")
            if b.scheme >= 2 and a.hint and (len(a.hint) != trees.HINT or not _finite([v for v in a.hint if v is not None])):
                raise HTTPException(400, f"hint: {trees.HINT} numbers (null = missing)")
            if len(a.rows) > MAX_ROWS:
                raise HTTPException(400, f"at most {MAX_ROWS} rows per answer")
            rows = []
            for r in a.rows:
                if r.check not in self.dims or len(r.x) != self.dims[r.check] or r.label not in (0, 1) or not _finite(r.x):
                    raise HTTPException(400, f"row {r.check!r}: unknown check, or not {self.dims.get(r.check, '?')} finite numbers")
                rows.append({"check": r.check, "label": r.label, "x": r.x})
            out.append({"hash": a.hash, "kind": a.kind, "pack": a.pack, "fp": a.fp, "hint": list(a.hint) if b.scheme >= 2 else [], "rows": rows})
        return out

    # ------------------------------------------------------------ training

    def cloud_rows(self) -> tuple[list[train.Row], dict]:
        """
        The shared answers as scheme 2 rows: one per sound and check, its features = the row's own ++ the answer's
        fingerprint ++ its name hint (missing: scheme 1's answers, or nothing to go by = NaN). Several people who answered
        the same file count as one (its file hash); if they disagree, it isn't used. At most MAX_PER_INSTALL answers of
        each plugin (its newest). Never the SANDBOX's.
        """
        per_install: dict[str, int] = {}
        votes: dict[tuple[str, str], dict] = {}
        installs = set()
        nan = float("nan")
        for install, h, pack, scheme, fp, hint, rows in self.store.training_answers(ACCEPTS):
            n = per_install.get(install, 0)
            if n >= self.s.max_per_install:
                continue
            per_install[install] = n + 1
            installs.add(install)
            h8 = [nan if v is None else float(v) for v in hint] if hint and len(hint) == trees.HINT else [nan] * trees.HINT
            for r in rows:
                if r["check"] not in self.dims or len(r["x"]) != self.dims[r["check"]] or len(fp) != FP_COLUMNS:
                    continue
                v = votes.setdefault((h, r["check"]), {"labels": {}, "x": r["x"] + fp + h8, "pack": pack})
                v["labels"][install] = bool(r["label"])
        out, disagreed = [], 0
        for (h, check), v in votes.items():
            labels = set(v["labels"].values())
            if len(labels) > 1:
                disagreed += 1
                continue
            out.append(train.Row(check=check, pack=v["pack"], label=labels.pop(), x=v["x"], source="shared", path=h[:12]))
        return out, {"installs": len(installs), "sounds": len({h for h, _ in votes}), "rows": len(out), "disagreed": disagreed}

    def current_heldout(self) -> dict:
        """What the model in use scored on the base, held out (the newest published one's, else the built-in's)."""
        m = self.store.published(self.scheme)
        return m["gate"].get("heldout", {}) if m else self.builtin_heldout

    def train_now(self, why: str) -> dict:
        """A candidate: trained on the base + the shared answers, checked by the gate, published only if allowed."""
        if not self.training.acquire(blocking=False):
            raise HTTPException(409, "already training")
        try:
            t0 = time.time()
            shared, info = self.cloud_rows()
            version = self.store.next_version()
            when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
            about = (f"cloud model {version}: trained {when} UTC on {len(self.base)} base examples + {info['rows']} shared "
                     f"({info['sounds']} sounds from {info['installs']} plugins), scheme {self.scheme}")
            report: list[str] = [f"{about} ({why})"]
            text, checks, heldout = trees.train_model(self.base + shared, about, report, self.bars)
            g = trees.gate(heldout, self.current_heldout())
            report.append("== the gate: on the base's examples held out (packs it never heard), never more wrong decisions than the model in use")
            for name, c in g.checks.items():
                now, new = c["current"] or {}, c["candidate"] or {}
                report.append(f"   {name:<22} now {now.get('decided', 0):5} decided, {now.get('wrong', 0):3} wrong | "
                              f"new {new.get('decided', 0):5} decided, {new.get('wrong', 0):3} wrong")
            report.extend("   NO: " + w for w in g.why)
            status = "candidate" if g.passed else "rejected"
            signature = sign.sign(self.key, version, self.scheme, text) if self.key else ""
            if g.passed and self.s.auto_publish and self.key:
                status = "published"
            self.store.add_model({"version": version, "scheme": self.scheme, "created": int(time.time()), "status": status,
                                  "text": text, "signature": signature, "report": "\n".join(report),
                                  "gate": json.dumps({"passed": g.passed, "why": g.why, "checks": g.checks, "heldout": heldout}),
                                  "answers": info["rows"]})
            self.store.set("trained_at", str(int(time.time())))
            self.store.set("trained_upto", str(self.store.newest_answer(ACCEPTS)))
            print(f"trained model {version} ({why}): {status}, {time.time() - t0:.1f} s", flush=True)
            return {"version": version, "status": status, "gate": {"passed": g.passed, "why": g.why}, "seconds": round(time.time() - t0, 1)}
        finally:
            self.training.release()

    def kinds(self, prefixes: list[str]) -> list[dict]:
        """
        The files under these hash prefixes that at least CONSENSUS_MIN plugins answered, all alike (one that disagrees
        and the file isn't given): their full hash, the kind, how many said it. The plugin compares the full hashes itself.
        """
        said: dict[str, list[tuple[str, int]]] = {}
        for h, kind, n in self.store.kinds_of(prefixes):
            said.setdefault(h, []).append((kind, n))
        return [{"hash": h, "kind": k[0][0], "installs": k[0][1]} for h, k in sorted(said.items())
                if len(k) == 1 and k[0][1] >= self.s.consensus]

    def due(self) -> bool:
        """Something new to learn from, and not trained in the last 20 hours."""
        last = int(self.store.get("trained_at", "0"))
        upto = int(self.store.get("trained_upto", "0"))
        return time.time() - last > 20 * 3600 and self.store.newest_answer(ACCEPTS) > upto


def _finite(v: list[float]) -> bool:
    return all(math.isfinite(t) and abs(t) <= 1e6 for t in v)


def create_app(cloud: Cloud | None = None, nightly: bool = True) -> FastAPI:
    holder: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        c = holder["cloud"] = cloud or Cloud()
        task = asyncio.create_task(_nightly(c)) if nightly else None
        print(f"Sample Manager cloud: scheme {c.scheme}, {len(c.base)} base examples, "
              f"{'signing' if c.key else 'NO SIGNING KEY (models can not be published)'}, "
              f"auto-publish {'on' if c.s.auto_publish else 'off'}", flush=True)
        yield
        if task:
            task.cancel()

    app = FastAPI(title="Sample Manager cloud", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)

    def the() -> Cloud:
        return holder["cloud"]

    def ip(request: Request) -> str:
        """Who is asking, for the limits only (never stored): Railway's proxy puts it in X-Real-IP / at the end of X-Forwarded-For."""
        real = request.headers.get("x-real-ip", "").strip()
        if real:
            return real
        fwd = request.headers.get("x-forwarded-for", "")
        return fwd.split(",")[-1].strip() if fwd else (request.client.host if request.client else "?")

    def admin(authorization: str = Header(default="")) -> None:
        token = the().s.admin_token
        if not token or not hmac.compare_digest(authorization.encode(), f"Bearer {token}".encode()):
            raise HTTPException(403, "admin only")

    @app.middleware("http")
    async def body_limit(request: Request, call_next):
        n = request.headers.get("content-length")
        if n is not None and n.isdigit() and int(n) > MAX_BODY:
            return JSONResponse({"detail": "too large"}, status_code=413)
        return await call_next(request)

    @app.get("/", response_class=PlainTextResponse)
    def root() -> str:
        return "Sample Manager cloud\n"

    @app.get("/health")
    def health() -> dict:
        """Up, its features' scheme, where it keeps its data (postgres; sqlite = a file that a redeploy loses), since when."""
        return {"ok": True, "scheme": the().scheme, "store": "postgres" if the().store.pg else "sqlite", "since": the().started}

    @app.get("/v1/public-key")
    def public_key() -> dict:
        """What the plugin checks a model's signature with (config Cloud.PublicKey): base64 of X || Y, P-256."""
        return {"key": sign.public_text(the().key)}

    @app.post("/v1/answers")
    def answers(batch: BatchIn, request: Request) -> dict:
        c = the()
        if not c.allow("ip:" + ip(request), 30) or not c.allow("install:" + batch.install, 20):
            raise HTTPException(429, "too many: try again in a minute")
        good = c.check_batch(batch)
        n = c.store.put_answers(batch.install, batch.app, batch.scheme, good, int(time.time()))
        return {"accepted": n}

    @app.get("/v1/model")
    def model(request: Request, have: int = 0, scheme: int = 0) -> dict:
        c = the()
        if not c.allow("ip:" + ip(request), 30):
            raise HTTPException(429, "too many: try again in a minute")
        m = c.store.published(scheme) if scheme == c.scheme else None
        if m is None:
            return {"version": 0}
        if m["version"] == have:
            return {"version": have, "same": True}
        return {"version": m["version"], "scheme": m["scheme"], "text": m["text"], "signature": m["signature"]}

    @app.post("/v1/kinds")
    def kinds(q: KindsIn, request: Request) -> dict:
        c = the()
        if not c.allow("ip:" + ip(request), 30):
            raise HTTPException(429, "too many: try again in a minute")
        if len(q.prefixes) > MAX_PREFIXES or not all(HEX4.match(p) for p in q.prefixes):
            raise HTTPException(400, f"prefixes: at most {MAX_PREFIXES}, each the first 4 hex digits of a file's SHA-256")
        return {"kinds": c.kinds(sorted(set(q.prefixes)))}

    @app.delete("/v1/installs/{install}")
    def forget(install: str, request: Request) -> dict:
        c = the()
        if not UUID.match(install):
            raise HTTPException(400, "install: a random UUID, lower case")
        if not c.allow("ip:" + ip(request), 30):
            raise HTTPException(429, "too many: try again in a minute")
        return {"deleted": c.store.delete_install(install)}

    @app.post("/v1/admin/train", dependencies=[Depends(admin)])
    async def admin_train() -> dict:
        return await asyncio.to_thread(the().train_now, "asked")

    @app.get("/v1/admin/models", dependencies=[Depends(admin)])
    def admin_models() -> list:
        return the().store.models()

    @app.get("/v1/admin/models/{version}", dependencies=[Depends(admin)])
    def admin_model(version: int) -> dict:
        m = the().store.model(version)
        if m is None:
            raise HTTPException(404, "no such model")
        return m

    @app.post("/v1/admin/models/{version}/publish", dependencies=[Depends(admin)])
    def admin_publish(version: int, force: int = 0) -> dict:
        c = the()
        m = c.store.model(version)
        if m is None:
            raise HTTPException(404, "no such model")
        if not m["signature"]:
            raise HTTPException(409, "not signed: set MODEL_SIGNING_KEY and train again")
        if m["status"] == "rejected" and not force:
            raise HTTPException(409, "the gate rejected it (publish with ?force=1 if you are sure)")
        c.store.set_status(version, "published")
        return {"version": version, "status": "published"}

    @app.post("/v1/admin/models/{version}/revoke", dependencies=[Depends(admin)])
    def admin_revoke(version: int) -> dict:
        c = the()
        if c.store.model(version) is None:
            raise HTTPException(404, "no such model")
        c.store.set_status(version, "revoked")
        m = c.store.published(c.scheme)
        return {"version": version, "status": "revoked", "now": m["version"] if m else 0}

    @app.get("/v1/admin/stats", dependencies=[Depends(admin)])
    def admin_stats() -> dict:
        c = the()
        m = c.store.published(c.scheme)
        return {**c.store.stats(), "scheme": c.scheme, "base": len(c.base), "published": m["version"] if m else 0,
                "auto_publish": c.s.auto_publish, "signing": c.key is not None}

    return app


async def _nightly(c: Cloud) -> None:
    """Every night at TRAIN_HOUR_UTC: a new candidate if answers came in (published by itself only with AUTO_PUBLISH=1)."""
    while True:
        await asyncio.sleep(600)
        try:
            if datetime.now(timezone.utc).hour == c.s.train_hour and c.due():
                await asyncio.to_thread(c.train_now, "nightly")
        except Exception as e:  # (the next night tries again)
            print(f"nightly training failed: {e!r}", flush=True)


app = create_app()
