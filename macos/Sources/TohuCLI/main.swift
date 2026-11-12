// tohu CLI:  tohu "szia, hogy vagy?"     or     echo "..." | tohu
//
//   --explain            list uncertain words with alternatives (stderr)
//   --json               print the full analysis as JSON
//   --resources DIR      exported model directory (default: $TOHU_RESOURCES, the app bundle, <repo>/models/export)
//   --threshold X        confidence below which a word counts as uncertain (default 0.9)
//   --no-guard           disable the English-word guard
import CoreML
import Foundation
import TohuCore

var args = Array(CommandLine.arguments.dropFirst())
@MainActor func flag(_ name: String) -> Bool {
    guard let i = args.firstIndex(of: name) else { return false }
    args.remove(at: i)
    return true
}
@MainActor func option(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    defer { args.removeSubrange(i...i + 1) }
    return args[i + 1]
}

let explain = flag("--explain"), json = flag("--json"), noGuard = flag("--no-guard")
let threshold = option("--threshold").flatMap(Double.init) ?? 0.9
/// --resources, $TOHU_RESOURCES, the app bundle (when run as Tohu.app/Contents/Helpers/tohu), the repo.
func defaultResources() -> String {
    let exe = URL(filePath: CommandLine.arguments[0]).resolvingSymlinksInPath()
    let bundled = exe.deletingLastPathComponent().appending(path: "../Resources").standardized
    if FileManager.default.fileExists(atPath: bundled.appending(path: "manifest.json").path) { return bundled.path }
    return URL(filePath: #filePath).deletingLastPathComponent().appending(path: "../../../models/export").standardized.path
}
let resources = URL(filePath: option("--resources") ?? ProcessInfo.processInfo.environment["TOHU_RESOURCES"] ?? defaultResources())

let text = args.isEmpty ? String(decoding: FileHandle.standardInput.readDataToEndOfFile(), as: UTF8.self) : args.joined(separator: " ")

do {
    var t = ContinuousClock.now
    let restorer = try Restorer(resources: resources, guard: !noGuard)
    let loadTime = ContinuousClock.now - t
    t = ContinuousClock.now
    let analysis = try restorer.analyze(text)
    let runTime = ContinuousClock.now - t

    if json {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .withoutEscapingSlashes]
        print(String(decoding: try enc.encode(analysis), as: UTF8.self))
    } else {
        print(analysis.text)
    }
    if explain {
        let unsure = analysis.words.filter { $0.confidence < threshold }
        var err = FileHandle.standardError
        print("\n[load \(loadTime.formatted(.units(allowed: [.milliseconds]))), \(text.unicodeScalars.count) chars in \(runTime.formatted(.units(allowed: [.milliseconds]))), \(unsure.count) uncertain word(s)]", to: &err)
        for w in unsure {
            let alts = w.alternatives.map { "\($0.text) (\(Int(($0.probability * 100).rounded()))%)" }.joined(separator: ", ")
            print("  \(w.text.padding(toLength: 20, withPad: " ", startingAt: 0)) \(Int((w.confidence * 100).rounded()))%  alt: \(alts)", to: &err)
        }
    }
} catch {
    FileHandle.standardError.write(Data("tohu: \(error)\n".utf8))
    exit(1)
}

extension FileHandle: @retroactive TextOutputStream {
    public func write(_ string: String) { write(Data(string.utf8)) }
}
