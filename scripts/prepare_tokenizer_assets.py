"""Build a portable, credential-free tokenizer bundle for offline CI tests."""

import argparse
import hashlib
import json
import os
import re
import shutil
import time
import zipfile
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from huggingface_hub import get_token, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

# Include custom tokenizer/config implementations and chat templates, but no weights.
ASSET_PATTERNS = ["*.json", "*.model", "*.tiktoken", "*.txt", "*.py", "*.jinja", "*.bpe"]


def rate_limit_wait(error):
    """Honor both Hub reset headers and standard Retry-After responses."""
    headers = error.response.headers
    delays = []
    retry_after = headers.get("Retry-After")
    if retry_after:
        try:
            delays.append(float(retry_after))
        except ValueError:
            try:
                reset = parsedate_to_datetime(retry_after)
                delays.append((reset - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError):
                pass
    reset = re.search(r"\bt=(\d+)", headers.get("RateLimit", ""))
    if reset:
        delays.append(float(reset.group(1)))
    # Some Hub API errors provide the reset only in their response message.
    reset = re.search(r"Retry after (\d+) seconds", str(error), re.IGNORECASE)
    if reset:
        delays.append(float(reset.group(1)))
    # Without a reset hint, wait for a complete Hub rate-limit window.
    return max([0, *delays]) + 1 if delays else 301


def download_snapshot(model_id, revision, max_attempts=10, retry_budget=1200):
    """Retry API-tree 429s too; the SDK's file-download retries do not cover them."""
    deadline = time.monotonic() + retry_budget
    for attempt in range(1, max_attempts + 1):
        try:
            return snapshot_download(
                model_id, revision=revision, allow_patterns=ASSET_PATTERNS, max_workers=1
            )
        except HfHubHTTPError as error:
            if error.response is None or error.response.status_code != 429:
                raise
            delay = rate_limit_wait(error)
            if attempt == max_attempts or time.monotonic() + delay > deadline:
                raise
            print(
                f"Hub rate limit for {model_id}; retrying in {delay:g}s "
                f"(attempt {attempt}/{max_attempts})",
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError("No snapshot download attempts configured")


def prepare_tiktoken_cache(output):
    # Phi-3-small calls get_encoding("cl100k_base") even when loaded from local files.
    # Seed tiktoken's URL-hash cache from the vocabulary in its pinned Hub snapshot.
    output = Path(output)
    phi_vocab = output / "microsoft--Phi-3-small-8k-instruct" / "cl100k_base.tiktoken"
    if phi_vocab.is_file():
        cache = output / "tiktoken"
        cache.mkdir(exist_ok=True)
        url = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
        shutil.copyfile(phi_vocab, cache / hashlib.sha1(url.encode()).hexdigest())


def prepare(manifest_path, output, include_token_required=False):
    manifest = json.loads(Path(manifest_path).read_text())
    output = Path(output)
    # A public-only rebuild must never retain files from a previous full bundle.
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    index = {}
    for entry in manifest["models"]:
        if entry["requires_token"] and not include_token_required:
            continue
        model_id = entry["repo_id"]
        print("Preparing " + model_id, flush=True)
        # Serial repositories and serial file downloads avoid a download stampede.
        revision = entry.get("revision", "main")
        snapshot = Path(download_snapshot(model_id, revision))
        relative = model_id.replace("/", "--")
        destination = output / relative
        if destination.exists():
            shutil.rmtree(destination)
        # Dereference Hub cache symlinks for portable Windows/Linux/macOS artifacts.
        shutil.copytree(snapshot, destination)
        index[model_id] = {"path": relative, "revision": revision, "commit": snapshot.name}
    prepare_tiktoken_cache(output)
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n")


def archive_fingerprint(archive):
    """Compare file contents, ignoring ZIP timestamps and compression details."""
    if not Path(archive).is_file():
        return None
    digest = hashlib.sha256()
    with zipfile.ZipFile(archive) as bundle:
        for name in sorted(bundle.namelist()):
            digest.update(name.encode() + b"\0")
            digest.update(hashlib.sha256(bundle.read(name)).digest())
    return digest.hexdigest()


def write_archive(root, archive, public_only=False, manifest_path=None):
    root, archive = Path(root), Path(archive)
    index = json.loads((root / "index.json").read_text())
    if public_only:
        models = json.loads(Path(manifest_path).read_text())["models"]
        public = {entry["repo_id"] for entry in models if not entry["requires_token"]}
        index = {model: entry for model, entry in index.items() if model in public}
    prefixes = tuple(entry["path"] + "/" for entry in index.values())
    archive.parent.mkdir(parents=True, exist_ok=True)
    old = archive_fingerprint(archive)
    temporary = archive.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(root.rglob("*")):
            name = path.relative_to(root).as_posix()
            if path.is_file() and name.startswith(prefixes):
                bundle.write(path, name)
            elif path.is_file() and not public_only and name.startswith("tiktoken/"):
                bundle.write(path, name)
        bundle.writestr("index.json", json.dumps(index, indent=2) + "\n")
    changed = archive_fingerprint(temporary) != old
    temporary.replace(archive)
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="tests/tokenizer_assets.json")
    parser.add_argument("--output", default=".ci/tokenizer-assets")
    parser.add_argument("--archive", default=".ci/tokenizer-assets.zip")
    parser.add_argument("--include-token-required", action="store_true")
    parser.add_argument(
        "--public-archive", help="Also export the public-only assets without secrets"
    )
    args = parser.parse_args()
    include = args.include_token_required or bool(get_token())
    prepare(args.manifest, args.output, include)
    changed = write_archive(args.output, args.archive)
    print(f"Created {args.archive}; changed={changed}", flush=True)
    if args.public_archive and Path(args.public_archive) != Path(args.archive):
        public_changed = write_archive(
            args.output, args.public_archive, public_only=True, manifest_path=args.manifest
        )
    else:
        public_changed = changed
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write(f"changed={str(changed).lower()}\n")
            output.write(f"public_changed={str(public_changed).lower()}\n")


if __name__ == "__main__":
    main()
