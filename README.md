# tohu

Restores Hungarian accents (ékezetek) in text typed without them: `kerlek kuldd at` → `kérlek küldd át`.
A 75M-parameter character-level Transformer, plus a word-frequency lexicon and a guard that
leaves English words alone. Runs locally on macOS through Core ML.

## Accuracy

Word accuracy on held-out text (3000 lines per set, `scripts/analyze_errors.py`, cleaned test sets):

| | web formal | web informal | subtitles | Wikipedia | challenge | Hunglish | 1-4 words |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `tohu.pt` (75.4M) | 99.61 | 99.51 | 99.28 | 99.39 | 99.53 | 99.33 | 98.73 |
| previous (38.7M) | 99.55 | 99.43 | 99.23 | 99.30 | 99.53 | 99.26 | 98.58 |
| `tohu_small.pt` (4.9M) | 99.10 | 98.87 | 98.65 | 98.57 | 98.70 | 98.72 | 97.84 |

About 40% of the remaining errors are pairs where both forms are real words (meg/még, levő/lévő),
about 20% are names, and about 20% are rare words, many of them typos in the reference text.

## Training

```sh
# download + filter corpora -> data/ (~5.8B characters; the jobs can run in parallel)
for s in 000 001 002 003; do for k in 00000 00001 00002; do
  uv run python scripts/prepare_data.py fineweb --shard ${s}_$k; done; done
uv run python scripts/prepare_data.py fineweb-test
for p in 0 1 2 3; do uv run python scripts/prepare_data.py wiki --part $p; done
uv run python scripts/prepare_data.py subs --part 1 && uv run python scripts/prepare_data.py subs --part 2
uv run python scripts/prepare_data.py english
uv run python scripts/build_lexicon.py                 # -> models/lexicon.tsv.gz
uv run python scripts/clean_data.py --procs 16         # drop lines with rare under-accented words -> data/clean/
# drop lines with accent typos a trained model is sure about -> data/clean2/ (~3 h on GPU)
TOHU_DATA=data/clean uv run python scripts/model_clean.py --model models/tohu.pt --out data/clean2
# 75M model: ~21 h on an RTX 4080 SUPER (16 GB); add --resume to continue after an interruption
TOHU_DATA=data/clean2 uv run python scripts/train.py --name v3 --dim 640 --layers 15 --heads 10 --amp \
  --tokens 24576 --steps 400000 --lr 6e-4 --warmup 4000 --workers 12 --eval-every 10000 --eval-lines 600 \
  --buckets 64:.1,128:.2,256:.3,384:.2,512:.2 --ema 0.9999
TOHU_DATA=data/clean uv run python scripts/analyze_errors.py --model models/v3_last.pt
```

## Layout

```
tohu/              Python package: model, training, reference inference (python -m tohu "...")
scripts/           data prep, training, evaluation, export_coreml.py
tests/             challenge.txt, a hand-written evaluation set
models/            (untracked) checkpoints and lexicon; models/export/ holds what the app ships
macos/             Swift package
  Sources/TohuCore/    Core ML model + decoding (a port of tohu/restore.py)
  Sources/TohuCLI/     `tohu` command-line tool
  Sources/TohuApp/     menu-bar app: keeps the model loaded, local API, hotkeys, writer panel
  Tests/               golden tests: Swift must match the Python outputs exactly
  scripts/build-app.sh builds Tohu.app
raycast/           Raycast extension, a client of the app's local API
```

## Build

```sh
# 1. Export the trained model for Core ML (re-run after every retrain)
uv run --group export python scripts/export_coreml.py

# 2. Test the Swift port against the Python outputs, then build and install the app
cd macos && scripts/test.sh
scripts/build-app.sh --install          # -> /Applications/Tohu.app

# 3. Raycast extension (opens it in Raycast in development mode)
cd ../raycast && npm install && npm run dev
```

## Using it

Tohu.app runs in the menu bar and keeps the model loaded (tens of ms per sentence, under a second for a page).

| Where | How |
| --- | --- |
| Any app | select text, press **⌃⌥E**: the selection is replaced with the accented text |
| Writer panel | **⌃⌥T**: type without accents, **⌘↩** pastes into the app you came from |
| Services menu | right-click a selection → Services → *Restore Hungarian Accents* |
| Raycast | *Add Accents to Selection*, *Add Accents as You Type* |
| Scripts | `curl -s localhost:7744/restore -d '{"text":"szia hogy vagy"}'` |
| Terminal | `/Applications/Tohu.app/Contents/Helpers/tohu --explain "..."` |

The hotkey and the writer's paste need **Accessibility** access (System Settings → Privacy &
Security → Accessibility). The app is ad-hoc signed, so after a rebuild you may need to switch it
off and on again there.

### Local API (127.0.0.1:7744)

| Request | Response |
| --- | --- |
| `GET /health` | `{"status": "ready" \| "loading", "version": 1}` |
| `POST /restore {"text"}` | `{"text"}` |
| `POST /analyze {"text"}` | `{"text", "words": [{start, end, utf16Start, utf16End, text, confidence, alternatives}]}` |

It listens on loopback only and refuses requests from web pages (requests with an `Origin` header).
To change the port: `defaults write dev.tohu.Tohu port -int 7745`.
