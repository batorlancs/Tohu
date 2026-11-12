import AppKit

/// "Fix selection": copy the selected text in the frontmost app, restore accents, paste the result
/// over the selection, then put the user's clipboard back.
@MainActor
enum SelectionFixer {
    enum Outcome { case fixed, unchanged, noSelection, notTrusted, failed(String) }

    static func run(engine: Engine) async -> Outcome {
        guard Keyboard.requestTrust() else { return .notTrusted }
        await Keyboard.waitForModifierRelease()

        let pb = NSPasteboard.general
        let saved = ClipboardSnapshot()
        let before = pb.changeCount
        Keyboard.copy()
        // The target app updates the pasteboard asynchronously.
        for _ in 0..<25 where pb.changeCount == before { try? await Task.sleep(for: .milliseconds(20)) }
        guard pb.changeCount != before, let text = pb.string(forType: .string), !text.isEmpty else {
            saved.restore()
            return .noSelection
        }

        let fixed: String
        do {
            fixed = try await engine.restore(text)
        } catch {
            saved.restore()
            return .failed("\(error)")
        }
        if fixed == text {
            saved.restore()
            return .unchanged
        }
        pb.clearContents()
        pb.setString(fixed, forType: .string)
        Keyboard.paste()
        // Give the target app time to read the pasteboard before restoring it.
        try? await Task.sleep(for: .milliseconds(400))
        saved.restore()
        return .fixed
    }
}
