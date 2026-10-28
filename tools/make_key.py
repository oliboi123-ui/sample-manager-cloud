"""A signing key of your own (only if you want to set MODEL_SIGNING_KEY yourself; without it the server makes one on its
first start and keeps it in its database). Prints the env var's value and the public key the plugin needs.
    python tools/make_key.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import sign  # noqa: E402

key = sign.new_key()
print("MODEL_SIGNING_KEY (Railway variable, keep it secret):")
print(sign.key_text(key))
print()
print("Cloud.PublicKey (the plugin's config.default.json):")
print(sign.public_text(key))
