"""Every test on a real Postgres, as on Railway: a throwaway one of its own (pgserver), gone afterwards.
    pip install pgserver
    python tests/on_postgres.py      (from cloud/)
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pgserver

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as folder:
        server = pgserver.get_server(Path(folder) / "pg", cleanup_mode="delete")
        try:
            os.environ["TEST_DATABASE_URL"] = server.get_uri()
            suite = unittest.defaultTestLoader.discover(str(HERE))
            ok = unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
        finally:
            server.cleanup()
    sys.exit(0 if ok else 1)
