import AppKit
import ServiceManagement
import SwiftUI
import TohuCore

@main
struct TohuApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate

    var body: some Scene {
        MenuBarExtra {
            MenuContent(app: delegate.app)
        } label: {
            Image(systemName: delegate.app.statusSymbol)
        }
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let app = AppController()

    func applicationDidFinishLaunching(_ notification: Notification) {
        app.start()
    }
}

@MainActor
@Observable
final class AppController {
    enum ModelState: Equatable { case loading, ready(ms: Int), failed(String) }

    private(set) var model: ModelState = .loading
    private(set) var apiError: String?
    private(set) var apiListening = false
    private(set) var lastAction: String?
    private(set) var hotkeysFailed: [String] = []
    var launchAtLogin = SMAppService.mainApp.status == .enabled

    let engine = Engine()
    let port: UInt16
    @ObservationIgnored private var server: APIServer?
    @ObservationIgnored private lazy var writer = WriterController(engine: engine)
    @ObservationIgnored private lazy var services = ServiceProvider(engine: engine)

    init() {
        let p = UserDefaults.standard.integer(forKey: "port")
        port = (1...65535).contains(p) ? UInt16(p) : APIServer.defaultPort
    }

    var statusSymbol: String {
        switch model {
        case .loading: "hourglass"
        case .ready: apiError == nil ? "character.cursor.ibeam" : "exclamationmark.triangle"
        case .failed: "exclamationmark.triangle"
        }
    }

    func start() {
        NSApp.servicesProvider = services
        NSUpdateDynamicServices()
        if !HotKeys.shared.register(HotKeys.fixSelection, { [weak self] in self?.fixSelection() }) {
            hotkeysFailed.append(HotKeys.fixSelection.display)
        }
        if !HotKeys.shared.register(HotKeys.openWriter, { [weak self] in self?.writer.toggle() }) {
            hotkeysFailed.append(HotKeys.openWriter.display)
        }
        startServer()
        loadModel()
    }

    func loadModel() {
        model = .loading
        engine.restorer = nil
        let dir = Engine.resourceURL
        Task.detached(priority: .userInitiated) { [weak self] in
            let t = ContinuousClock.now
            let result = Result { try Restorer(resources: dir) }
            let ms = Int((ContinuousClock.now - t) / .milliseconds(1))
            await self?.finishLoading(result, ms: ms)
        }
    }

    private func finishLoading(_ result: Result<Restorer, Error>, ms: Int) {
        switch result {
        case .success(let r):
            engine.restorer = r
            model = .ready(ms: ms)
        case .failure(let e):
            model = .failed("\(e)")
        }
    }

    private func startServer() {
        let s = APIServer(port: port, engine: engine) { [weak self] err in
            Task { @MainActor in
                self?.apiError = err
                self?.apiListening = err == nil
            }
        }
        s.start()
        server = s
    }

    func openWriter() { writer.show() }

    func fixSelection() {
        Task {
            switch await SelectionFixer.run(engine: engine) {
            case .fixed: lastAction = nil
            case .unchanged: lastAction = "Selection already had its accents"
            case .noSelection:
                lastAction = "No text was selected"
                NSSound.beep()
            case .notTrusted: lastAction = "Needs Accessibility access to copy and paste"
            case .failed(let e):
                lastAction = e
                NSSound.beep()
            }
        }
    }

    func setLaunchAtLogin(_ on: Bool) {
        do {
            if on { try SMAppService.mainApp.register() } else { try SMAppService.mainApp.unregister() }
        } catch {
            lastAction = "Launch at login: \(error.localizedDescription)"
        }
        launchAtLogin = SMAppService.mainApp.status == .enabled
    }

    var apiURL: String { "http://127.0.0.1:\(port)" }
}

struct MenuContent: View {
    @Bindable var app: AppController

    var body: some View {
        switch app.model {
        case .loading: Text("Loading model…")
        case .ready(let ms): Text("Model loaded (\(ms) ms)")
        case .failed(let e): Text("Model failed: \(e)")
        }
        if let e = app.apiError {
            Text("API unavailable: \(e)")
        } else if app.apiListening {
            Text("API on \(app.apiURL)")
        }
        if let a = app.lastAction { Text(a) }

        Divider()
        Button("Open Writer        \(HotKeys.openWriter.display)") { app.openWriter() }
        Button("Fix Selection       \(HotKeys.fixSelection.display)") { app.fixSelection() }
        if !app.hotkeysFailed.isEmpty {
            Text("Shortcut taken by another app: \(app.hotkeysFailed.joined(separator: ", "))")
        }
        if !Keyboard.isTrusted {
            Button("Allow Accessibility (for paste)…") {
                Keyboard.requestTrust()
                Keyboard.openAccessibilitySettings()
            }
        }

        Divider()
        Toggle("Launch at Login", isOn: Binding(get: { app.launchAtLogin }, set: { app.setLaunchAtLogin($0) }))
        Button("Copy API URL") {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(app.apiURL, forType: .string)
        }
        Button("Reload Model") { app.loadModel() }
        Divider()
        Button("Quit Tohu") { NSApp.terminate(nil) }.keyboardShortcut("q")
    }
}
