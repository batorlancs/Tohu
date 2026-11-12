#!/usr/bin/env bash
# swift test, working around Command Line Tools-only setups: SwiftPM's build system there doesn't
# pass the swift-testing macro plugin to the compiler ("plugin for module 'TestingMacros' not found").
set -euo pipefail
cd "$(dirname "$0")/.."
PLUGINS="$(xcode-select -p)/usr/lib/swift/host/plugins/testing"
if [[ -d "$PLUGINS" ]]; then
  exec swift test -Xswiftc -plugin-path -Xswiftc "$PLUGINS" "$@"
fi
exec swift test "$@"
