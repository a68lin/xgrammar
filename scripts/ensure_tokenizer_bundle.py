"""Restore offline CI operation after a missing or incomplete tokenizer cache."""

import argparse
import zipfile
from pathlib import Path

from verify_tokenizer_bundle import verify


def complete(archive, manifest, full=False):
    try:
        verify(archive, manifest, full)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, zipfile.BadZipFile) as error:
        print("Tokenizer bundle needs fetching: " + str(error), flush=True)
        return False
    return True


def ensure(archive, manifest, output, full=False):
    if complete(archive, manifest, full):
        print("Using complete cached tokenizer bundle without contacting Hugging Face", flush=True)
        return False

    # Cache hits need only the standard library; install/import the Hub client on misses.
    from prepare_tokenizer_assets import prepare, write_archive

    prepare(manifest, output, include_token_required=full)
    # Discard a corrupt old ZIP only after all downloads have succeeded.
    Path(archive).unlink(missing_ok=True)
    write_archive(output, archive)
    verify(archive, manifest, full)
    print("Fetched one complete tokenizer bundle for all test jobs", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--manifest", default="tests/tokenizer_assets.json")
    parser.add_argument("--output", default=".ci/tokenizer-assets")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        raise SystemExit(0 if complete(args.archive, args.manifest, args.full) else 1)
    ensure(args.archive, args.manifest, args.output, args.full)


if __name__ == "__main__":
    main()
