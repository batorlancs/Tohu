import Foundation
import TohuCore

/// Thread-safe handle on the loaded model, shared by the UI (main thread) and the API server
/// (network queue). `nil` while loading or after a failed load.
final class Engine: @unchecked Sendable {
    private let lock = NSLock()
    private var _restorer: Restorer?

    var restorer: Restorer? {
        get { lock.withLock { _restorer } }
        set { lock.withLock { _restorer = newValue } }
    }

    /// Where the exported model lives: $TOHU_RESOURCES (development), else the app bundle.
    static var resourceURL: URL {
        if let dir = ProcessInfo.processInfo.environment["TOHU_RESOURCES"] { return URL(filePath: dir) }
        return Bundle.main.resourceURL!
    }

    /// Runs restoration off the main thread.
    func restore(_ text: String) async throws -> String {
        guard let r = restorer else { throw EngineError.notReady }
        return try await Task.detached(priority: .userInitiated) { try r.restore(text) }.value
    }

    func analyze(_ text: String) async throws -> Analysis {
        guard let r = restorer else { throw EngineError.notReady }
        return try await Task.detached(priority: .userInitiated) { try r.analyze(text) }.value
    }
}

enum EngineError: Error, CustomStringConvertible {
    case notReady
    var description: String { "the model is not loaded yet" }
}
