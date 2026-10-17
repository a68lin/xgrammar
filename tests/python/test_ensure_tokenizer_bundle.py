"""Exercise cache-hit and fetch-fallback paths without network requests."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("state", ["missing", "corrupt", "incomplete"])
@pytest.mark.parametrize("full", [False, True])
def test_fetch_once_then_reuse(tmp_path, monkeypatch, state, full):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    ensure = importlib.import_module("ensure_tokenizer_bundle")
    prepare = importlib.import_module("prepare_tokenizer_assets")
    manifest = tmp_path / "models.json"
    models = [
        {"repo_id": "example/public", "requires_token": False},
        {"repo_id": "example/gated", "requires_token": True},
    ]
    manifest.write_text(json.dumps({"models": models}))
    archive = tmp_path / "bundle.zip"
    root = tmp_path / "candidate"
    if state == "corrupt":
        archive.write_bytes(b"not a ZIP")
    elif state == "incomplete":
        root.mkdir()
        (root / "index.json").write_text("{}")
        prepare.write_archive(root, archive)
    calls = []

    def download(manifest_path, output, include_token_required=False):
        calls.append(include_token_required)
        output = Path(output)
        output.mkdir(exist_ok=True)
        index = {}
        for model in models:
            if model["requires_token"] and not include_token_required:
                continue
            relative = model["repo_id"].replace("/", "--")
            directory = output / relative
            directory.mkdir(exist_ok=True)
            (directory / "config.json").write_text("{}")
            index[model["repo_id"]] = {"path": relative, "revision": "main", "commit": "a" * 40}
        (output / "index.json").write_text(json.dumps(index))

    monkeypatch.setattr(prepare, "prepare", download)
    assert ensure.ensure(archive, manifest, root, full)
    assert calls == [full]
    assert not ensure.ensure(archive, manifest, root, full)
    assert calls == [full]


def test_failed_fetch_does_not_replace_old_bundle(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    ensure = importlib.import_module("ensure_tokenizer_bundle")
    prepare = importlib.import_module("prepare_tokenizer_assets")
    manifest = tmp_path / "models.json"
    manifest.write_text('{"models": [{"repo_id": "example/public", "requires_token": false}]}')
    archive = tmp_path / "bundle.zip"
    archive.write_bytes(b"old incomplete archive")

    def fail(*args, **kwargs):
        raise RuntimeError("Hub download failed")

    monkeypatch.setattr(prepare, "prepare", fail)
    with pytest.raises(RuntimeError, match="Hub download failed"):
        ensure.ensure(archive, manifest, tmp_path / "candidate")
    assert archive.read_bytes() == b"old incomplete archive"
