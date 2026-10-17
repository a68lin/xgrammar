"""Test refresh/publication boundaries without accessing Hugging Face."""

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest


def script(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[2] / "scripts" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def manifest(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "models": [
                    {"repo_id": "example/public", "requires_token": False},
                    {"repo_id": "example/pinned", "requires_token": True, "revision": "a" * 40},
                ]
            }
        )
    )
    return path


@pytest.fixture
def snapshots(tmp_path):
    paths = []
    for sha in ["b" * 40, "a" * 40]:
        path = tmp_path / "snapshots" / sha
        path.mkdir(parents=True)
        (path / "config.json").write_text('{"vocab_size": 2}')
        paths.append(path)
    return paths


def test_refresh_resolves_main_and_retains_explicit_pins(
    tmp_path, manifest, snapshots, monkeypatch
):
    assets = script("prepare_tokenizer_assets")
    calls = []

    def download(model, revision):
        calls.append((model, revision))
        return snapshots[len(calls) - 1]

    monkeypatch.setattr(assets, "download_snapshot", download)
    root = tmp_path / "assets"
    assets.prepare(manifest, root, include_token_required=True)
    assert calls == [("example/public", "main"), ("example/pinned", "a" * 40)]
    index = json.loads((root / "index.json").read_text())
    assert index["example/public"]["commit"] == "b" * 40
    full, public = tmp_path / "full.zip", tmp_path / "public.zip"
    assert assets.write_archive(root, full)
    assert assets.write_archive(root, public, public_only=True, manifest_path=manifest)
    verifier = script("verify_tokenizer_bundle")
    assert len(verifier.verify(full, manifest, full=True)) == 2
    assert len(verifier.verify(public, manifest)) == 1
    with zipfile.ZipFile(public) as bundle:
        assert not any("pinned" in name for name in bundle.namelist())
    with pytest.raises(RuntimeError, match="Model missing"):
        verifier.verify(public, manifest, full=True)
    assert not assets.write_archive(root, full)
    (root / "example--public" / "config.json").write_text('{"vocab_size": 3}')
    assert assets.write_archive(root, full)


def test_failed_refresh_does_not_replace_published_archive(
    tmp_path, manifest, snapshots, monkeypatch
):
    assets = script("prepare_tokenizer_assets")
    archive = tmp_path / "published.zip"
    archive.write_bytes(b"previous bundle")

    def download(model, revision):
        if model.endswith("pinned"):
            raise RuntimeError("Download failed")
        return snapshots[0]

    monkeypatch.setattr(assets, "download_snapshot", download)
    with pytest.raises(RuntimeError, match="Download failed"):
        assets.prepare(manifest, tmp_path / "candidate", include_token_required=True)
    assert archive.read_bytes() == b"previous bundle"


def test_public_refresh_skips_gated_models_and_clears_old_files(
    tmp_path, manifest, snapshots, monkeypatch
):
    assets = script("prepare_tokenizer_assets")
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "old-secret").write_text("not an asset")
    calls = []

    def download(model, revision):
        calls.append(model)
        return snapshots[0]

    monkeypatch.setattr(assets, "download_snapshot", download)
    assets.prepare(manifest, root)
    assert calls == ["example/public"]
    assert not (root / "old-secret").exists()


def test_bundle_verifier_rejects_wrong_revision(tmp_path, manifest):
    verifier = script("verify_tokenizer_bundle")
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(
            "index.json",
            json.dumps(
                {
                    "example/public": {
                        "path": "example--public",
                        "revision": "main",
                        "commit": "b" * 40,
                    },
                    "example/pinned": {
                        "path": "example--pinned",
                        "revision": "main",
                        "commit": "b" * 40,
                    },
                }
            ),
        )
        bundle.writestr("example--public/config.json", "{}")
        bundle.writestr("example--pinned/config.json", "{}")
    with pytest.raises(RuntimeError, match="Pinned revision missing"):
        verifier.verify(archive, manifest, full=True)


def test_public_cli_publishes_once_when_archive_paths_match(
    tmp_path, manifest, snapshots, monkeypatch
):
    assets = script("prepare_tokenizer_assets")
    archive = tmp_path / "public.zip"
    outputs = tmp_path / "github-output"
    monkeypatch.setattr(assets, "get_token", lambda: None)
    monkeypatch.setattr(assets, "download_snapshot", lambda *args: snapshots[0])
    monkeypatch.setenv("GITHUB_OUTPUT", str(outputs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "assets"),
            "--archive",
            str(archive),
            "--public-archive",
            str(archive),
        ],
    )
    assets.main()
    assert outputs.read_text() == "changed=true\npublic_changed=true\n"
    outputs.write_text("")
    assets.main()
    assert outputs.read_text() == "changed=false\npublic_changed=false\n"
