"""Download the two audited QM8 author files by immutable Git blob ID."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


def fetch(directory):
    manifest_path = Path(__file__).with_name("qm8_author_download_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    directory.mkdir(parents=True, exist_ok=True)
    for item in manifest:
        destination = directory / item["name"]
        if destination.exists():
            content = destination.read_bytes()
        else:
            url = ("https://api.github.com/repos/raghurama123/ExcitedStatesQM8/git/blobs/"
                   + item["git_blob_sha1"])
            request = Request(url, headers={"User-Agent": "TriMo-Gate-reproduction",
                                            "Accept": "application/vnd.github+json"})
            with urlopen(request, timeout=120) as response:
                record = json.load(response)
            if record.get("encoding") != "base64":
                raise ValueError("Expected base64 Git blob response")
            content = base64.b64decode(record["content"])
        blob_sha = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if (len(content) != item["size"] or blob_sha != item["git_blob_sha1"]
                or hashlib.sha256(content).hexdigest() != item["sha256"]):
            raise ValueError(f"Author source verification failed: {item['name']}")
        if not destination.exists():
            destination.write_bytes(content)
        print(item["name"], "verified")
    (directory / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    fetch(parser.parse_args().output)
