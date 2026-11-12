import AppKit

/// macOS Services entry "Restore Hungarian Accents" (right-click → Services, or bind a shortcut in
/// System Settings → Keyboard → Keyboard Shortcuts → Services). Declared in Info.plist (NSServices).
/// Works in any Cocoa text field without Accessibility permission: macOS hands us the selection
/// and replaces it with what we return.
final class ServiceProvider: NSObject {
    private let engine: Engine

    init(engine: Engine) {
        self.engine = engine
    }

    @objc func restoreAccents(_ pboard: NSPasteboard, userData: String?, error: AutoreleasingUnsafeMutablePointer<NSString?>) {
        guard let text = pboard.string(forType: .string) else {
            error.pointee = "No text was selected." as NSString
            return
        }
        // Services are synchronous; restoration takes milliseconds.
        guard let restorer = engine.restorer, let fixed = try? restorer.restore(text) else {
            error.pointee = "Tohu is still loading its model." as NSString
            return
        }
        pboard.clearContents()
        pboard.setString(fixed, forType: .string)
    }
}
