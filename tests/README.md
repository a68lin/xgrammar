To test, run `pytest .` under `xgrammar` folder. You may need to do the following:

```bash
pip install sentencepiece
pip install protobuf
pip install -U "huggingface_hub[cli]"
huggingface-cli login --token YOUR_HF_TOKEN
```

Make sure you also have access to the gated models, which should only require you to agree
some terms on the models' website on huggingface.

Python tests should load Hugging Face tokenizers with `tokenizer_utils.load_tokenizer`.
It caches model/option combinations for the lifetime of the test process and returns an
independent copy for each call. This avoids repeated Hub metadata requests during
parametrized tests while keeping tokenizer mutations isolated between tests.

## Scheduled tokenizer bundles and offline CI

`tests/tokenizer_assets.json` lists model IDs and whether they are used only by
`hf_token_required` tests. Ordinary entries do not pin a commit: the refresh workflow
fetches their current Hub `main`. Only tests intentionally targeting a release keep
an explicit `revision` (DeepSeek V4.1 and the two MiMo checkpoints).

`refresh_tokenizer_assets.yaml` runs weekly (Monday at 06:23 UTC) or on manual dispatch.
It downloads tokenizer/config files serially with bounded 429 retries, not model
weights. It creates full and public-only ZIPs, recording resolved SHAs in `index.json`.
It verifies archive integrity and required assets, then publishes changed bundles as
`tokenizer-bundle-v2-full-<run>-<attempt>` and `tokenizer-bundle-v2-public-<run>-<attempt>`.
File-content fingerprints avoid duplicate caches when a refresh produces identical
files. Publishing does not depend on model tests passing; newer assets can expose
compatibility regressions in subsequent CI runs. A failed/incomplete download never
publishes a candidate. No update commits, PRs, or auto-merge configuration are needed.

Only refresh runs in the default branch's cache scope publish bundles. The optional
`source_ref` dispatch input lets maintainers exercise a candidate implementation before
merging it; it does not change that cache scope. Without `HF_TOKEN`, refresh only
publishes public assets. Full snapshots and public subsets use distinct prefixes.

The unit-test preparation job only restores the newest available matching cache;
it does not contact the Hub or save caches. It checks that the bundle covers the PR's
model list and explicit pins, then shares the selected ZIP with all 39 test jobs as
one workflow artifact. The selected cache key is recorded in the job summary. Tests
load local paths from `index.json` with `HF_HUB_OFFLINE=1`; `XGRAMMAR_TEST_OFFLINE=1`
also rejects socket connections during pytest. Individual jobs never fetch assets.

If the cache is absent/evicted, or lacks a model newly required by a PR, preparation
fails with recovery instructions. Run **Refresh tokenizer assets** on the default
branch (set `source_ref` to the PR branch for a new model), then rerun unit tests.
A new HF snapshot may change results between runs of the same source commit; the
cache key and resolved SHAs identify the exact resources used. Scheduled refreshes
become active only once the workflow is present on the default branch.

To prepare and inspect a bundle locally:

```bash
python scripts/prepare_tokenizer_assets.py
python scripts/verify_tokenizer_bundle.py .ci/tokenizer-assets.zip
python -m zipfile -e .ci/tokenizer-assets.zip .ci/extracted
XGRAMMAR_TOKENIZER_ASSETS="$PWD/.ci/extracted" TIKTOKEN_CACHE_DIR="$PWD/.ci/extracted/tiktoken" \
  HF_HUB_OFFLINE=1 XGRAMMAR_TEST_OFFLINE=1 \
  python -m pytest tests/python/test_tokenizer_utils.py
```

With an authorized `HF_TOKEN`, also pass `--full` to the verification command.
Normal developer tests without `XGRAMMAR_TOKENIZER_ASSETS` retain Hub loading.
