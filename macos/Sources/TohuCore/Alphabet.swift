import Foundation

/// Contents of models/export/manifest.json, written by scripts/export_coreml.py.
struct Manifest: Decodable {
    let version: Int
    let chars: [String]
    let variants: [String: String]
    let normalize: [String: String]
    let buckets: [Int]
    let chunk: Int
    let ctx: Int
    let alpha: Double
    let minTotal: Int

    enum CodingKeys: String, CodingKey {
        case version, chars, variants, normalize, buckets, chunk, ctx, alpha
        case minTotal = "min_total"
    }
}

/// Character tables from tohu/chars.py, built from the manifest so the alphabet lives in one place.
struct Alphabet {
    static let numClasses = 4
    static let pad: Int32 = 0, unknown: Int32 = 1

    let ids: [UInt32: Int32]
    let variants: [UInt32: Scalars]  // lowercase base vowel -> variants (index = label)
    let labelOf: [UInt32: Int]  // any vowel, any case -> label
    let toBase: [UInt32: UInt32]  // accented vowel -> base vowel (same case)
    let normalizeMap: [UInt32: UInt32]
    let baseVowels: Set<UInt32>

    init(_ m: Manifest) {
        var ids: [UInt32: Int32] = [:]
        for (i, c) in m.chars.enumerated() {
            ids[c.unicodeScalars.first!.value] = Int32(i + 2)
        }
        var variants: [UInt32: Scalars] = [:], labelOf: [UInt32: Int] = [:], toBase: [UInt32: UInt32] = [:]
        for (base, vs) in m.variants {
            let b = base.unicodeScalars.first!.value
            let vars = scalars(vs)
            variants[b] = vars
            for (label, v) in vars.enumerated() {
                for (c, bb) in [(v, b), (upper(v), upper(b))] {
                    labelOf[c] = label
                    toBase[c] = bb
                }
            }
        }
        self.ids = ids
        self.variants = variants
        self.labelOf = labelOf
        self.toBase = toBase
        self.normalizeMap = Dictionary(uniqueKeysWithValues: m.normalize.map {
            ($0.key.unicodeScalars.first!.value, $0.value.unicodeScalars.first!.value)
        })
        self.baseVowels = Set(variants.keys.flatMap { [$0, upper($0)] })
    }

    func normalize(_ t: Scalars) -> Scalars { t.map { normalizeMap[$0] ?? $0 } }

    func deaccent(_ t: Scalars) -> Scalars { t.map { toBase[$0] ?? $0 } }

    func encode(_ t: ArraySlice<UInt32>) -> [Int32] {
        t.map { $0 > 0xFFFF ? Self.unknown : (ids[$0] ?? Self.unknown) }
    }

    func validLabels(_ base: UInt32) -> Int { variants[lower([base])[0]]!.count }

    func applyLabel(_ base: UInt32, _ label: Int) -> UInt32 {
        let v = variants[lower([base])[0]]![label]
        return isUpper(base) ? upper(v) : v
    }
}
