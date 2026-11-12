import CoreML
import Foundation

public enum TohuError: Error, CustomStringConvertible {
    case badResource(String)
    case missingResource(URL)

    public var description: String {
        switch self {
        case .badResource(let m): "bad resource: \(m)"
        case .missingResource(let u): "missing resource: \(u.path) (run scripts/export_coreml.py)"
        }
    }
}

public struct Alternative: Codable, Sendable, Equatable {
    public let text: String
    public let probability: Double
}

/// A restored word with the model's confidence. `start`/`end` are offsets in Unicode scalars
/// (= Python code points); `utf16Start`/`utf16End` are the same span for JavaScript / NSString.
public struct Word: Codable, Sendable, Equatable {
    public let start: Int
    public let end: Int
    public let utf16Start: Int
    public let utf16End: Int
    public let text: String
    public let confidence: Double
    public let alternatives: [Alternative]
}

public struct Analysis: Codable, Sendable {
    public let text: String
    public let words: [Word]
}

/// Swift port of `tohu.restore.Restorer`: unaccented (or partially accented) Hungarian in,
/// accented out. Accents the user typed are always kept.
///
/// Pipeline per text:
///   1. character model -> log-probabilities of each vowel variant
///   2. per word: score(form) = model_logprob(form) + alpha * log P_lex(form | unaccented word)
///   3. English guard: never accent a plainly English word that isn't Hungarian
public final class Restorer: @unchecked Sendable {
    /// Files in a resource directory (models/export/ or the app bundle's Resources).
    public static let resourceFiles = ["Tohu.mlmodelc", "lexicon.bin", "guard.json", "manifest.json"]

    let alphabet: Alphabet
    private let network: AccentNetwork
    private let lexicon: Lexicon
    private let englishGuard: EnglishGuard
    private let alpha: Double
    private let minTotal: Int
    public let guardEnabled: Bool

    public init(resources dir: URL, computeUnits: MLComputeUnits = .cpuOnly, guard guardEnabled: Bool = true) throws {
        for f in Self.resourceFiles where !FileManager.default.fileExists(atPath: dir.appending(path: f).path) {
            throw TohuError.missingResource(dir.appending(path: f))
        }
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: dir.appending(path: "manifest.json")))
        guard manifest.version == 1 else { throw TohuError.badResource("manifest version \(manifest.version)") }
        alphabet = Alphabet(manifest)
        network = try AccentNetwork(url: dir.appending(path: "Tohu.mlmodelc"), manifest: manifest, computeUnits: computeUnits)
        lexicon = try Lexicon(url: dir.appending(path: "lexicon.bin"))
        englishGuard = try EnglishGuard(url: dir.appending(path: "guard.json"))
        alpha = manifest.alpha
        minTotal = manifest.minTotal
        self.guardEnabled = guardEnabled
    }

    public func restore(_ text: String) throws -> String {
        let t = alphabet.normalize(scalars(text))
        return string(try decode(t, logits: network.logits(t, alphabet: alphabet), wantWords: false).out)
    }

    /// Restored text plus per-word confidence and alternatives (for UI highlighting).
    public func analyze(_ text: String) throws -> Analysis {
        let t = alphabet.normalize(scalars(text))
        let (out, words) = try decode(t, logits: network.logits(t, alphabet: alphabet), wantWords: true)
        // UTF-16 offsets for clients that index strings that way (JavaScript, NSString).
        var utf16 = [Int](repeating: 0, count: out.count + 1)
        for (i, c) in out.enumerated() { utf16[i + 1] = utf16[i] + (c > 0xFFFF ? 2 : 1) }
        return Analysis(text: string(out), words: words.compactMap { w in
            guard out[w.range].elementsEqual(w.form) else { return nil }  // undone by the English guard
            return Word(start: w.range.lowerBound, end: w.range.upperBound,
                        utf16Start: utf16[w.range.lowerBound], utf16End: utf16[w.range.upperBound],
                        text: string(w.form), confidence: w.confidence, alternatives: w.alternatives)
        })
    }

    // ------------------------------------------------------------ decoding
    private struct Decoded {
        let range: Range<Int>
        let form: Scalars
        let confidence: Double
        let alternatives: [Alternative]
    }

    /// Log-softmax over the valid variants of each base vowel; -inf elsewhere. Float32 like NumPy.
    private func logprobs(_ t: Scalars, _ logits: [Float]) -> [Float] {
        let k = Alphabet.numClasses
        var lp = [Float](repeating: -.infinity, count: logits.count)
        for (i, c) in t.enumerated() where alphabet.baseVowels.contains(c) {
            let n = alphabet.validLabels(c)
            let z = logits[i * k..<i * k + n]
            let m = z.max()!
            let lse = m + log(z.reduce(Float(0)) { $0 + exp($1 - m) })
            for j in 0..<n { lp[i * k + j] = logits[i * k + j] - lse }
        }
        return lp
    }

    private func decode(_ t: Scalars, logits: [Float], wantWords: Bool) -> (out: Scalars, words: [Decoded]) {
        let k = Alphabet.numClasses
        let lp = logprobs(t, logits)
        var chars = t
        for (i, c) in t.enumerated() where alphabet.baseVowels.contains(c) {  // typed accents are kept
            let row = lp[i * k..<(i + 1) * k]
            chars[i] = alphabet.applyLabel(c, row.firstIndex(of: row.max()!)! - i * k)
        }

        var words: [Decoded] = []
        for r in tokenRanges(t) {
            let vowelPos = r.filter { alphabet.baseVowels.contains(t[$0]) }
            if vowelPos.isEmpty { continue }
            let w = Array(t[r])
            let low = lower(w)
            let modelForm = lower(chars[r])
            // Python would mis-index here; only happens for exotic case mappings (e.g. 'İ').
            guard low.count == w.count, modelForm.count == w.count else { continue }

            var cands: [Scalars: Double] = [:]
            var unseen = 0.0
            if let forms = lexicon.forms(key: alphabet.deaccent(low)) {
                let total = forms.reduce(0) { $0 + $1.count }
                if total >= minTotal {
                    let denom = Double(total) + 0.5 * Double(forms.count + 1)
                    for (f, c) in forms { cands[f] = log((Double(c) + 0.5) / denom) }
                    unseen = log(0.5 / denom)
                }
            }
            // forms must agree with accents the user typed
            cands = cands.filter { f, _ in
                f.count == w.count && r.allSatisfy { alphabet.baseVowels.contains(t[$0]) || f[$0 - r.lowerBound] == low[$0 - r.lowerBound] }
            }
            if cands[modelForm] == nil { cands[modelForm] = unseen }

            func modelLP(_ f: Scalars) -> Float {
                vowelPos.reduce(Float(0)) { acc, i in
                    guard let label = alphabet.labelOf[f[i - r.lowerBound]] else { return -.infinity }
                    return acc + lp[i * k + label]
                }
            }
            // Python: sorted(((score, form) ...), reverse=True) -> ties broken by larger form.
            let scored = cands.map { (score: Double(modelLP($0.key)) + alpha * $0.value, form: $0.key) }
                .sorted { $0.score != $1.score ? $0.score > $1.score : $1.form.lexicographicallyPrecedes($0.form) }
            let best = matchCase(scored[0].form, w)
            chars.replaceSubrange(r, with: best)

            if wantWords {
                let zmax = scored[0].score
                let e = scored.map { exp($0.score - zmax) }
                let sum = e.reduce(0, +)
                let alts = zip(scored.dropFirst(), e.dropFirst()).compactMap { s, p in
                    p / sum > 0.02 ? Alternative(text: string(matchCase(s.form, w)), probability: p / sum) : nil
                }
                words.append(Decoded(range: r, form: best, confidence: e[0] / sum, alternatives: alts))
            }
        }
        if guardEnabled { chars = englishGuard.apply(source: t, output: chars) }
        return (chars, words)
    }

    private func matchCase(_ form: Scalars, _ original: Scalars) -> Scalars {
        zip(form, original).map { f, o in isUpper(o) ? upper(f) : f }
    }
}
