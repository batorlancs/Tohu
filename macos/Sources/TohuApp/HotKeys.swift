import AppKit
import Carbon.HIToolbox

/// System-wide hotkeys via Carbon's RegisterEventHotKey: works without Accessibility permission
/// and doesn't steal the key from other apps unless it is pressed with exactly these modifiers.
@MainActor
final class HotKeys {
    static let shared = HotKeys()

    struct Shortcut {
        let keyCode: Int
        let modifiers: Int  // Carbon: controlKey, optionKey, cmdKey, shiftKey
        let display: String
    }

    static let fixSelection = Shortcut(keyCode: kVK_ANSI_E, modifiers: controlKey | optionKey, display: "⌃⌥E")
    static let openWriter = Shortcut(keyCode: kVK_ANSI_T, modifiers: controlKey | optionKey, display: "⌃⌥T")

    private var handlers: [UInt32: () -> Void] = [:]
    private var refs: [EventHotKeyRef] = []
    private var nextID: UInt32 = 1

    private init() {
        var spec = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
        InstallEventHandler(GetApplicationEventTarget(), { _, event, _ in
            var hk = EventHotKeyID()
            GetEventParameter(event, EventParamName(kEventParamDirectObject), EventParamType(typeEventHotKeyID),
                              nil, MemoryLayout<EventHotKeyID>.size, nil, &hk)
            let id = hk.id
            MainActor.assumeIsolated { HotKeys.shared.handlers[id]?() }
            return noErr
        }, 1, &spec, nil, nil)
    }

    /// Returns false if another app already owns the combination.
    @discardableResult
    func register(_ s: Shortcut, _ handler: @escaping () -> Void) -> Bool {
        let id = nextID
        nextID += 1
        var ref: EventHotKeyRef?
        let status = RegisterEventHotKey(UInt32(s.keyCode), UInt32(s.modifiers),
                                         EventHotKeyID(signature: OSType(0x544F_4855), id: id),  // 'TOHU'
                                         GetApplicationEventTarget(), 0, &ref)
        guard status == noErr, let ref else { return false }
        refs.append(ref)
        handlers[id] = handler
        return true
    }
}
