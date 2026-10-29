"""The cloud, tested on this PC (SQLite, no network):   python -m unittest discover -s tests -v   (from cloud/)"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app import base, sign, train  # noqa: E402
from app.main import Cloud, Settings, create_app  # noqa: E402

BASE = HERE.parent / "data" / "base.tsv.gz"


class TrainerMatchesTheApp(unittest.TestCase):
    """The port trains exactly what the app's own trainer (--lab-train) trains from the same examples."""

    def test_same_checks_as_the_app(self):
        scheme, rows = base.load(BASE)
        self.assertGreater(scheme, 0)
        text, checks = train.train_model(rows, "test", [], use_base_folds=True)
        app = train.read_model((HERE / "fixtures" / "app-trained.txt").read_text(encoding="utf-8"))
        self.assertEqual(set(app), set(checks))
        for name, c in app.items():
            p = checks[name]
            self.assertEqual(c.sure, p.sure, name)
            for a, b in ((c.w, p.w), (c.mean, p.mean), (c.sd, p.sd)):
                self.assertLess(abs(a - b).max(), 1e-9, name)
            self.assertLess(abs(c.bias - p.bias), 1e-9, name)
        again = train.read_model(text)  # (what the plugin parses: the very same numbers)
        for name, c in checks.items():
            self.assertTrue((again[name].w == c.w).all() and again[name].bias == c.bias and again[name].sure == c.sure, name)


class Api(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.key = sign.new_key()
        s = Settings()
        s.database_url = "sqlite:///" + os.path.join(cls.tmp.name, "test.db")
        s.admin_token = "secret"
        s.signing_key = sign.key_text(cls.key)
        s.auto_publish = False
        cls.cloud = Cloud(s)
        cls.client = TestClient(create_app(cls.cloud, nightly=False))
        cls.client.__enter__()
        cls.admin = {"Authorization": "Bearer secret"}

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        cls.cloud.store.conn.close()
        cls.tmp.cleanup()

    def post(self, install, answers, scheme=None):
        return self.client.post("/v1/answers", json={"install": install, "app": "test", "scheme": scheme or self.cloud.scheme, "answers": answers})

    @staticmethod
    def answer(h, kind="Kick", label=0, pack="0123456789abcdef"):
        return {"hash": h, "pack": pack, "kind": kind, "fp": [0.5] * 64, "rows": [{"check": "loop", "label": label, "x": [0.1] * 7}]}

    def test_health(self):
        h = self.client.get("/health").json()
        self.assertEqual((h["ok"], h["store"]), (True, "sqlite"))

    def test_public_key(self):
        self.assertEqual(self.client.get("/v1/public-key").json()["key"], sign.public_text(self.key))

    def test_a_server_without_a_key_makes_one_and_keeps_it(self):
        s = Settings()
        s.database_url = "sqlite:///" + os.path.join(self.tmp.name, "keys.db")
        s.signing_key = ""
        first = Cloud(s)
        key = sign.public_text(first.key)
        first.store.conn.close()
        again = Cloud(s)
        self.assertEqual(sign.public_text(again.key), key)
        again.store.conn.close()

    def test_on_railway_it_wont_start_without_postgres(self):
        for url in ("", "sqlite:///" + os.path.join(self.tmp.name, "lost.db")):
            s = Settings()
            s.hosted, s.database_url = True, url
            with self.assertRaises(RuntimeError, msg=url):
                Cloud(s)
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "lost.db")))

    def test_answers_kept_replaced_and_taken_back(self):
        me = str(uuid.uuid4())
        h1, h2 = "a" * 64, "b" * 64
        self.assertEqual(self.post(me, [self.answer(h1), self.answer(h2)]).json(), {"accepted": 2})
        self.assertEqual(self.post(me, [self.answer(h1, "Tom")]).status_code, 200)
        mine = [a for a in self.cloud.store.answers(self.cloud.scheme) if a[0] == me]
        self.assertEqual(sorted((a[1], a[2]) for a in mine), [(h1, "Tom"), (h2, "Kick")])
        self.post(me, [{"hash": h2, "kind": None}])
        self.assertEqual([a[1] for a in self.cloud.store.answers(self.cloud.scheme) if a[0] == me], [h1])

    def test_bad_answers_are_refused(self):
        me = str(uuid.uuid4())
        good = self.answer("c" * 64)
        cases = {
            "install": ({"install": "not-a-uuid"}, 400),
            "hash": ({"hash": "xyz"}, 400),
            "kind": ({"kind": "Banjo"}, 400),
            "pack": ({"pack": "short"}, 400),
            "fp": ({"fp": [0.5] * 10}, 400),
            "nan": ({"fp": [float("nan")] * 64}, 400),
            "check": ({"rows": [{"check": "sharpness", "label": 1, "x": [0.1] * 7}]}, 400),
            "dims": ({"rows": [{"check": "loop", "label": 1, "x": [0.1] * 6}]}, 400),
            "label": ({"rows": [{"check": "loop", "label": 2, "x": [0.1] * 7}]}, 400),
            "huge": ({"rows": [{"check": "loop", "label": 1, "x": [1e9] * 7}]}, 400),
        }
        for what, (change, code) in cases.items():
            body = {"install": me, "app": "test", "scheme": self.cloud.scheme, "answers": [{**good, **change}]}
            if what == "install":
                body["install"] = change["install"]
                body["answers"] = [good]
            raw = json.dumps(body).encode()  # (NaN too: httpx won't write it, a bad client could)
            self.assertEqual(self.client.post("/v1/answers", content=raw, headers={"content-type": "application/json"}).status_code, code, what)
        self.assertEqual(self.post(me, [good], scheme=self.cloud.scheme + 1).status_code, 409)
        self.assertEqual(self.post(me, [good] * 201).status_code, 400)
        big = self.client.post("/v1/answers", content=b"x" * 2_100_000, headers={"content-type": "application/json"})
        self.assertEqual(big.status_code, 413)

    def test_too_many_requests(self):
        me = str(uuid.uuid4())
        codes = [self.post(me, [self.answer("d" * 64)]).status_code for _ in range(22)]
        self.assertIn(429, codes)
        self.assertEqual(codes[0], 200)
        self.cloud.limits.clear()

    def test_admin_needs_the_token(self):
        self.assertEqual(self.client.get("/v1/admin/stats").status_code, 403)
        self.assertEqual(self.client.get("/v1/admin/stats", headers={"Authorization": "Bearer nope"}).status_code, 403)
        self.assertEqual(self.client.get("/v1/admin/stats", headers=self.admin).status_code, 200)

    def test_forget_an_install(self):
        me = str(uuid.uuid4())
        self.post(me, [self.answer("e" * 64), self.answer("f" * 64)])
        self.assertEqual(self.client.delete(f"/v1/installs/{me}").json(), {"deleted": 2})
        self.assertFalse([a for a in self.cloud.store.answers(self.cloud.scheme) if a[0] == me])

    def test_people_who_disagree_dont_count(self):
        h = "1" * 64
        self.post(str(uuid.uuid4()), [self.answer(h, label=1)])
        self.post(str(uuid.uuid4()), [self.answer(h, label=0)])
        rows, info = self.cloud.cloud_rows()
        self.assertFalse([r for r in rows if r.path == h[:12]])
        self.assertGreaterEqual(info["disagreed"], 1)
        self.cloud.limits.clear()

    def test_a_model_is_trained_signed_published_and_revoked(self):
        self.assertEqual(self.client.get(f"/v1/model?scheme={self.cloud.scheme}").json(), {"version": 0})
        t = self.client.post("/v1/admin/train", headers=self.admin).json()
        self.assertEqual(t["status"], "candidate", t)
        v = t["version"]
        # not published yet: the plugins still get "use your built-in"
        self.assertEqual(self.client.get(f"/v1/model?scheme={self.cloud.scheme}").json(), {"version": 0})
        self.assertEqual(self.client.post(f"/v1/admin/models/{v}/publish", headers=self.admin).json()["status"], "published")
        m = self.client.get(f"/v1/model?scheme={self.cloud.scheme}&have=0").json()
        self.assertEqual(m["version"], v)
        self.assertTrue(sign.verify(self.key.public_key(), v, self.cloud.scheme, m["text"], m["signature"]))
        self.assertFalse(sign.verify(self.key.public_key(), v, self.cloud.scheme, m["text"] + " ", m["signature"]))
        self.assertFalse(sign.verify(self.key.public_key(), v + 1, self.cloud.scheme, m["text"], m["signature"]))
        self.assertEqual(set(train.read_model(m["text"])), set(train.read_model(self.cloud.builtin)))
        self.assertEqual(self.client.get(f"/v1/model?scheme={self.cloud.scheme}&have={v}").json(), {"version": v, "same": True})
        self.assertEqual(self.client.get(f"/v1/model?scheme={self.cloud.scheme + 1}").json(), {"version": 0})
        self.assertEqual(self.client.post(f"/v1/admin/models/{v}/revoke", headers=self.admin).json()["now"], 0)
        self.assertEqual(self.client.get(f"/v1/model?scheme={self.cloud.scheme}").json(), {"version": 0})
        self.assertEqual(self.client.get(f"/v1/admin/models/{v}", headers=self.admin).json()["status"], "revoked")

    def test_the_gate_stops_bad_answers(self):
        """Many people "answer" snares as claps and claps as snares (copies of the base's own sounds): never published."""
        rows = [r for r in self.cloud.base if r.check == "pair:Snare|Clap"]
        people = [str(uuid.uuid4()) for _ in range(3)]
        for start in range(0, len(rows), 200):
            batch = [{"hash": f"{i:064x}", "pack": r.pack, "kind": "Clap" if r.label else "Snare", "fp": [0.5] * 64,
                      "rows": [{"check": r.check, "label": 0 if r.label else 1, "x": r.x}]} for i, r in enumerate(rows[start:start + 200], start)]
            for p in people:
                self.cloud.limits.clear()
                self.assertEqual(self.post(p, batch).status_code, 200)
        t = self.cloud.train_now("test")
        self.assertEqual(t["status"], "rejected", t)
        self.assertTrue(any(w.startswith("pair:Snare|Clap") for w in t["gate"]["why"]), t)
        v = t["version"]
        self.assertEqual(self.client.post(f"/v1/admin/models/{v}/publish", headers=self.admin).status_code, 409)
        for p in people:
            self.client.delete(f"/v1/installs/{p}")
        self.cloud.limits.clear()


if __name__ == "__main__":
    unittest.main()
