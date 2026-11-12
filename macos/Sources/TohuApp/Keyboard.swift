import AppKit
import ApplicationServices
import Carbon.HIToolbox

/// Synthetic ⌘C / ⌘V and clipboard save/restore. Posting key events needs Accessibility permission.
@MainActor
enum Keyboard {
    static var isTrusted: Bool { AXIsProcessTrusted() }

    /// Shows the system prompt (once per launch at most) that leads to Privacy & Security → Accessibility.
    @discardableResult
    static func requestTrust() -> Bool {
        AXIsProcessTrustedWithOptions(["AXTrustedCheckOptionPrompt": true] as CFDictionary)
    }

    static func openAccessibilitySettings() {
        NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")!)
    }

    /// ⌘ + key, with the other modifiers cleared so a still-held hotkey (⌃⌥) doesn't leak in.
    static func pressCommand(_ keyCode: Int) {
        let src = CGEventSource(stateID: .privateState)
        for down in [true, false] {
            let e = CGEvent(keyboardEventSource: src, virtualKey: CGKeyCode(keyCode), keyDown: down)
            e?.flags = .maskCommand
            e?.post(tap: .cghidEventTap)
        }
    }

    static func copy() { pressCommand(kVK_ANSI_C) }
    static func paste() { pressCommand(kVK_ANSI_V) }

    /// Waits until the hotkey's modifiers are physically released (max ~1 s), so the synthetic
    /// ⌘C isn't seen as ⌃⌥⌘C by apps that read the live modifier state.
    static func waitForModifierRelease() async {
        for _ in 0..<50 {
            let flags = CGEventSource.flagsState(.combinedSessionState)
            if flags.isDisjoint(with: [.maskControl, .maskAlternate, .maskCommand, .maskShift]) { return }
            try? await Task.sleep(for: .milliseconds(20))
        }
    }
}

/// A full copy of the general pasteboard (all items, all types), to put back after we borrow it.
@MainActor
struct ClipboardSnapshot {
    private let items: [[NSPasteboard.PasteboardType: Data]]

    init() {
        items = (NSPasteboard.general.pasteboardItems ?? []).map { item in
            Dictionary(uniqueKeysWithValues: item.types.compactMap { t in item.data(forType: t).map { (t, $0) } })
        }
    }

    func restore() {
        let pb = NSPasteboard.general
        pb.clearContents()
        guard !items.isEmpty else { return }
        pb.writeObjects(items.map { types in
            let item = NSPasteboardItem()
            for (t, d) in types { item.setData(d, forType: t) }
            return item
        })
    }
}
