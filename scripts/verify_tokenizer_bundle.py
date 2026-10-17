"""Check bundle completeness offline before publication or distribution to tests."""

import argparse
import json
import re
import zipfile
from pathlib import Path, PurePosixPath


def verify(archive, manifest_path, full=False):
    models = json.loads(Path(manifest_path).read_text())["models"]
    with zipfile.ZipFile(archive) as bundle:
        bad_file = bundle.testzip()
        if bad_file:
            raise RuntimeError("Corrupt tokenizer asset: " + bad_file)
        names = set(bundle.namelist())
        for name in names:
            if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
                raise RuntimeError("Unsafe tokenizer archive path: " + name)
        index = json.loads(bundle.read("index.json"))
        for model in models:
            if model["requires_token"] and not full:
                continue
            model_id = model["repo_id"]
            if model_id not in index:
                raise RuntimeError(
                    "Model missing from tokenizer bundle; run Refresh tokenizer assets: " + model_id
                )
            entry = index[model_id]
            if entry["path"] != model_id.replace("/", "--"):
                raise RuntimeError("Invalid tokenizer directory: " + model_id)
            if not re.fullmatch(r"[0-9a-f]{40}", entry["commit"]):
                raise RuntimeError("Missing resolved commit: " + model_id)
            revision = model.get("revision")
            if revision and revision not in (entry["revision"], entry["commit"]):
                raise RuntimeError("Pinned revision missing from tokenizer bundle: " + model_id)
            if not any(name.startswith(entry["path"] + "/") for name in names):
                raise RuntimeError("Tokenizer directory is empty: " + model_id)
        if full and "microsoft/Phi-3-small-8k-instruct" in index:
            if not any(name.startswith("tiktoken/") for name in names):
                raise RuntimeError("Missing prepared tiktoken vocabulary cache")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--manifest", default="tests/tokenizer_assets.json")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    index = verify(args.archive, args.manifest, args.full)
    print(f"Verified {len(index)} tokenizer snapshots without contacting Hugging Face")


if __name__ == "__main__":
    main()
