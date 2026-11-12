import Foundation
import Testing

@testable import TohuCore

/// Compares the Swift port against models/export/golden.jsonl, written by scripts/export_coreml.py
/// from the Python Restorer running the same Core ML model.
struct GoldenTests {
    static let export = URL(filePath: #filePath).deletingLastPathComponent()
        .appending(path: "../../../models/export").standardized

    struct Case: Decodable {
        struct W: Decodable { let start: Int, end: Int, text: String, confidence: Double }
        let input: String
        let output: String
        let words: [W]
    }

    static let cases: [Case] = {
        guard let data = try? Data(contentsOf: export.appending(path: "golden.jsonl")) else { return [] }
        return String(decoding: data, as: UTF8.self).split(separator: "\n")
            .map { try! JSONDecoder().decode(Case.self, from: Data($0.utf8)) }
    }()

    static let restorer = try? Restorer(resources: export)

    @Test func goldenFileExists() {
        #expect(!Self.cases.isEmpty, "run: uv run --group export python scripts/export_coreml.py")
        #expect(Self.restorer != nil)
    }

    @Test func outputsMatchPython() throws {
        let r = try #require(Self.restorer)
        var mismatches: [String] = []
        for c in Self.cases {
            let a = try r.analyze(c.input)
            if a.text != c.output {
                mismatches.append("in:  \(c.input)\npy:  \(c.output)\nswi: \(a.text)")
                continue
            }
            let got = a.words.map { "\($0.start)-\($0.end) \($0.text)" }
            let want = c.words.map { "\($0.start)-\($0.end) \($0.text)" }
            if got != want { mismatches.append("words differ for: \(c.input)\npy:  \(want)\nswi: \(got)") }
            for (g, w) in zip(a.words, c.words) where abs(g.confidence - w.confidence) > 1e-3 {
                mismatches.append("confidence \(g.text): py \(w.confidence) swift \(g.confidence)")
            }
        }
        #expect(mismatches.isEmpty, "\(mismatches.count)/\(Self.cases.count) mismatches:\n\(mismatches.prefix(10).joined(separator: "\n\n"))")
    }

    @Test func tokenizerMatchesPythonRegex() {
        let t = scalars("deadline-ra 2010-től, a_b x-  szó-")
        #expect(tokenRanges(t).map { string(t[$0]) } == ["deadline-ra"])
        #expect(letterRuns(t).map { string(t[$0]) } == ["deadline", "ra", "től", "a", "b", "x", "szó"])
    }
}
