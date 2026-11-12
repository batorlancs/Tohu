# tohu

Restores Hungarian accents (ékezetek) in text typed without them: `kerlek kuldd at` → `kérlek küldd át`.
A 4.9M-parameter character-level Transformer, plus a word-frequency lexicon and a guard that
leaves English words alone. Runs locally on macOS through Core ML.

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

Tohu.app runs in the menu bar and keeps the model loaded (about 200 ms to load, a few ms per sentence).

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
