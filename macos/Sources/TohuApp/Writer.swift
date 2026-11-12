import AppKit
import SwiftUI
import TohuCore

/// Spotlight-style panel: type without accents, see the accented text live, ⌘↩ pastes it into
/// the app you came from. Uncertain words are highlighted; picking an alternative writes that
/// accent into the input, and since typed accents are always kept, the choice sticks.
@MainActor
final class WriterController: NSObject, NSWindowDelegate {
    private let engine: Engine
    private var panel: NSPanel?
    private var previousApp: NSRunningApplication?
    private let model = WriterModel()

    init(engine: Engine) {
        self.engine = engine
        super.init()
        model.engine = engine
    }

    func toggle() {
        if let panel, panel.isVisible { return close() }
        show()
    }

    func show() {
        let front = NSWorkspace.shared.frontmostApplication
        if front?.bundleIdentifier != Bundle.main.bundleIdentifier { previousApp = front }
        let panel = self.panel ?? makePanel()
        self.panel = panel
        panel.center()
        panel.makeKeyAndOrderFront(nil)
        NSApp.activate()
    }

    func close() {
        panel?.orderOut(nil)
        previousApp?.activate()
    }

    /// Put the result on the clipboard and paste it into the previous app (if allowed to type).
    func commit() {
        let text = model.output
        guard !text.isEmpty else { return close() }
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(text, forType: .string)
        model.clear()
        close()
        guard Keyboard.isTrusted else { return }  // text stays on the clipboard
        Task {
            try? await Task.sleep(for: .milliseconds(150))
            Keyboard.paste()
        }
    }

    func copy() {
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(model.output, forType: .string)
    }

    private func makePanel() -> NSPanel {
        let p = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 640, height: 340),
                        styleMask: [.titled, .closable, .resizable, .fullSizeContentView],
                        backing: .buffered, defer: false)
        p.titleVisibility = .hidden
        p.titlebarAppearsTransparent = true
        p.isMovableByWindowBackground = true
        p.level = .floating
        p.hidesOnDeactivate = false
        p.isReleasedWhenClosed = false
        p.collectionBehavior = [.moveToActiveSpace, .fullScreenAuxiliary]
        p.delegate = self
        p.contentView = NSHostingView(rootView: WriterView(
            model: model, onCommit: { [weak self] in self?.commit() },
            onCopy: { [weak self] in self?.copy() }, onClose: { [weak self] in self?.close() }))
        return p
    }

    nonisolated func windowWillClose(_ notification: Notification) {
        MainActor.assumeIsolated { _ = previousApp?.activate() }
    }
}

@MainActor
@Observable
final class WriterModel {
    var input = "" { didSet { if input != oldValue { schedule() } } }
    private(set) var output = ""
    private(set) var words: [Word] = []
    private(set) var error: String?
    var engine: Engine?
    static let threshold = 0.9

    private var pending: Task<Void, Never>?

    var uncertain: [Word] { words.filter { $0.confidence < Self.threshold && !$0.alternatives.isEmpty } }

    func clear() {
        input = ""
        output = ""
        words = []
        pins = [:]
    }

    private func schedule() {
        pending?.cancel()
        let text = input
        pending = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(80))
            guard !Task.isCancelled, let self, let engine = self.engine else { return }
            do {
                let a = try await engine.analyze(text)
                guard !Task.isCancelled, text == self.input else { return }
                self.apply(output: a.text, words: a.words)
                self.error = nil
            } catch {
                self.error = "\(error)"
            }
        }
    }

    /// A word the user picked by hand; kept while that span of the input is unchanged.
    private struct Pin { let end: Int; let input: [UInt16]; let form: String }
    private var pins: [Int: Pin] = [:]  // keyed by UTF-16 start

    func choose(_ form: String, for w: Word) {
        let u = Array(input.utf16)
        guard w.utf16End <= u.count, w.utf16End - w.utf16Start == form.utf16.count else { return }
        pins[w.utf16Start] = Pin(end: w.utf16End, input: Array(u[w.utf16Start..<w.utf16End]), form: form)
        apply(output: output, words: words)
    }

    /// Restoration never changes length, so output offsets are input offsets.
    private func apply(output: String, words: [Word]) {
        let inp = Array(input.utf16)
        var out = Array(output.utf16)
        pins = pins.filter { start, p in p.end <= inp.count && Array(inp[start..<p.end]) == p.input && p.end <= out.count }
        for (start, p) in pins { out.replaceSubrange(start..<p.end, with: Array(p.form.utf16)) }
        self.output = String(decoding: out, as: UTF16.self)
        self.words = words.filter { pins[$0.utf16Start] == nil }
    }

    /// Output with uncertain words highlighted.
    var attributedOutput: AttributedString {
        var s = AttributedString(output)
        let u = Array(output.utf16)
        for w in uncertain where w.utf16End <= u.count {
            let lo = String(decoding: u[..<w.utf16Start], as: UTF16.self).count
            let len = w.text.count
            let start = s.characters.index(s.startIndex, offsetBy: lo)
            let end = s.characters.index(start, offsetBy: len)
            s[start..<end].backgroundColor = .orange.opacity(0.25)
            s[start..<end].underlineStyle = .single
        }
        return s
    }
}

struct WriterView: View {
    @Bindable var model: WriterModel
    let onCommit: () -> Void
    let onCopy: () -> Void
    let onClose: () -> Void
    @FocusState private var focused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            TextEditor(text: $model.input)
                .font(.system(size: 15))
                .scrollContentBackground(.hidden)
                .focused($focused)
                .frame(minHeight: 90)
                .overlay(alignment: .topLeading) {
                    if model.input.isEmpty {
                        Text("Írj ékezetek nélkül… (irj ekezetek nelkul)")
                            .font(.system(size: 15)).foregroundStyle(.tertiary)
                            .padding(.leading, 5).allowsHitTesting(false)
                    }
                }
            Divider()
            ScrollView {
                Text(model.attributedOutput)
                    .font(.system(size: 15))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(minHeight: 60)
            if !model.uncertain.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        ForEach(model.uncertain, id: \.start) { w in
                            Menu {
                                ForEach(w.alternatives, id: \.text) { a in
                                    Button("\(a.text)  \(Int((a.probability * 100).rounded()))%") { model.choose(a.text, for: w) }
                                }
                                Button("\(w.text)  \(Int((w.confidence * 100).rounded()))%  (keep)") { model.choose(w.text, for: w) }
                            } label: {
                                Text("\(w.text) ?")
                            }
                            .menuStyle(.borderlessButton)
                            .fixedSize()
                            .padding(.horizontal, 8).padding(.vertical, 3)
                            .background(.orange.opacity(0.15), in: Capsule())
                        }
                    }
                }
            }
            HStack {
                if let e = model.error { Text(e).foregroundStyle(.red).font(.caption) }
                Spacer()
                Button("Copy", action: onCopy).keyboardShortcut("c", modifiers: [.command, .shift])
                Button("Paste", action: onCommit).keyboardShortcut(.return, modifiers: .command)
                    .buttonStyle(.borderedProminent)
            }
            .font(.callout)
        }
        .padding(16)
        .padding(.top, 8)
        .frame(minWidth: 420, minHeight: 240)
        .onExitCommand(perform: onClose)
        .onAppear { focused = true }
    }
}
