import Foundation
import Network
import TohuCore

/// Minimal HTTP/1.1 server on 127.0.0.1 for local clients (Raycast, the CLI, scripts).
///
///   GET  /health                      {"status": "ready" | "loading", "version": 1}
///   POST /restore  {"text": "..."}    {"text": "..."}
///   POST /analyze  {"text": "..."}    {"text": "...", "words": [{start, end, utf16Start, utf16End,
///                                                               text, confidence, alternatives}]}
///
/// Only loopback is bound, and requests carrying an Origin header (i.e. from web pages) are refused.
final class APIServer: @unchecked Sendable {
    static let defaultPort: UInt16 = 7744
    static let apiVersion = 1
    private static let maxRequest = 4 << 20

    let port: UInt16
    private let engine: Engine
    private let queue = DispatchQueue(label: "tohu.api", attributes: .concurrent)
    private var listener: NWListener?
    private let onState: @Sendable (String?) -> Void  // nil = listening, else an error message

    init(port: UInt16, engine: Engine, onState: @escaping @Sendable (String?) -> Void) {
        self.port = port
        self.engine = engine
        self.onState = onState
    }

    func start() {
        do {
            let params = NWParameters.tcp
            params.requiredLocalEndpoint = .hostPort(host: .ipv4(.loopback), port: NWEndpoint.Port(rawValue: port)!)
            params.allowLocalEndpointReuse = true
            let l = try NWListener(using: params)
            l.newConnectionHandler = { [weak self] c in self?.accept(c) }
            l.stateUpdateHandler = { [onState, port] state in
                switch state {
                case .ready: onState(nil)
                case .failed(let e): onState("port \(port): \(e.localizedDescription)")
                default: break
                }
            }
            l.start(queue: queue)
            listener = l
        } catch {
            onState("port \(port): \(error.localizedDescription)")
        }
    }

    func stop() { listener?.cancel() }

    // ------------------------------------------------------------ connections
    private func accept(_ c: NWConnection) {
        c.start(queue: queue)
        receive(c, buffer: Data())
    }

    private func receive(_ c: NWConnection, buffer: Data) {
        c.receive(minimumIncompleteLength: 1, maximumLength: 1 << 16) { [weak self] data, _, isComplete, error in
            guard let self else { return c.cancel() }
            var buf = buffer
            if let data { buf.append(data) }
            switch Request.parse(buf) {
            case .complete(let req):
                Task { await self.send(c, self.route(req)) }
            case .incomplete where buf.count > Self.maxRequest:
                Task { await self.send(c, .error(413, "request too large")) }
            case .incomplete where isComplete || error != nil:
                c.cancel()
            case .incomplete:
                self.receive(c, buffer: buf)
            case .invalid:
                Task { await self.send(c, .error(400, "malformed request")) }
            }
        }
    }

    private func send(_ c: NWConnection, _ r: Response) async {
        let reason = [200: "OK", 400: "Bad Request", 403: "Forbidden", 404: "Not Found", 405: "Method Not Allowed",
                      413: "Payload Too Large", 500: "Internal Server Error", 503: "Service Unavailable"][r.status] ?? ""
        var head = "HTTP/1.1 \(r.status) \(reason)\r\n"
        head += "Content-Type: application/json; charset=utf-8\r\nContent-Length: \(r.body.count)\r\nConnection: close\r\n\r\n"
        c.send(content: Data(head.utf8) + r.body, completion: .contentProcessed { _ in c.cancel() })
    }

    // ------------------------------------------------------------ routing
    private struct TextBody: Decodable { let text: String }
    private struct TextResult: Encodable { let text: String }
    private struct Health: Encodable { let status: String; let version: Int }

    private func route(_ req: Request) async -> Response {
        if req.headers["origin"] != nil { return .error(403, "browser requests are not allowed") }
        switch (req.method, req.path) {
        case ("GET", "/health"), ("GET", "/"):
            return .json(Health(status: engine.restorer == nil ? "loading" : "ready", version: Self.apiVersion))
        case ("POST", "/restore"), ("POST", "/analyze"):
            guard let body = try? JSONDecoder().decode(TextBody.self, from: req.body) else {
                return .error(400, #"expected JSON body {"text": "..."}"#)
            }
            guard engine.restorer != nil else { return .error(503, "model is still loading") }
            do {
                if req.path == "/restore" { return .json(TextResult(text: try await engine.restore(body.text))) }
                return .json(try await engine.analyze(body.text))
            } catch {
                return .error(500, "\(error)")
            }
        case (_, "/health"), (_, "/restore"), (_, "/analyze"):
            return .error(405, "method not allowed")
        default:
            return .error(404, "not found")
        }
    }
}

private struct Response {
    let status: Int
    let body: Data

    static func json(_ v: some Encodable) -> Response {
        let enc = JSONEncoder()
        enc.outputFormatting = [.withoutEscapingSlashes]
        return (try? enc.encode(v)).map { Response(status: 200, body: $0) } ?? .error(500, "encoding failed")
    }

    static func error(_ status: Int, _ message: String) -> Response {
        Response(status: status, body: (try? JSONEncoder().encode(["error": message])) ?? Data())
    }
}

private struct Request {
    let method: String
    let path: String
    let headers: [String: String]  // lowercased names
    let body: Data

    enum Parse { case complete(Request), incomplete, invalid }

    static func parse(_ buf: Data) -> Parse {
        guard let sep = buf.firstRange(of: Data("\r\n\r\n".utf8)) else { return .incomplete }
        guard let head = String(data: buf[..<sep.lowerBound], encoding: .utf8) else { return .invalid }
        let lines = head.components(separatedBy: "\r\n")
        let parts = lines[0].split(separator: " ")
        guard parts.count >= 2 else { return .invalid }
        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            guard let colon = line.firstIndex(of: ":") else { continue }
            headers[line[..<colon].lowercased()] = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
        }
        let length = Int(headers["content-length"] ?? "0") ?? -1
        guard length >= 0 else { return .invalid }
        let bodyStart = sep.upperBound
        guard buf.count - bodyStart >= length else { return .incomplete }
        let path = String(parts[1].split(separator: "?", maxSplits: 1)[0])
        return .complete(Request(method: String(parts[0]), path: path, headers: headers,
                                 body: buf[bodyStart..<bodyStart + length]))
    }
}
