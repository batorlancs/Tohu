import Foundation

/// Port of `english_guard` (tohu/restore.py): undo accents put on a word that is plainly English
/// and not Hungarian. Uses the wordfreq numbers exported to guard.json.
struct EnglishGuard {
    private let en: [Scalars: (zipfEn: Double, zipfHu: Double)]
    private let huKnown: Set<Scalars>  // accent variants of English words with zipf_hu >= 1.5

    private struct File: Decodable {
        let en: [String: [Double]]
        let hu_known: [String]
    }

    init(url: URL) throws {
        let f = try JSONDecoder().decode(File.self, from: Data(contentsOf: url))
        var en: [Scalars: (Double, Double)] = [:]
        en.reserveCapacity(f.en.count)
        for (w, z) in f.en where z.count == 2 { en[scalars(w)] = (z[0], z[1]) }
        self.en = en
        self.huKnown = Set(f.hu_known.map(scalars))
    }

    func apply(source: Scalars, output: Scalars) -> Scalars {
        var res = output
        for r in letterRuns(source) {
            let wIn = source[r], wOut = output[r]
            if wIn.elementsEqual(wOut) { continue }
            guard let z = en[lower(wIn)] else { continue }  // not in the list: zipf_en < 3
            if z.zipfEn >= 3.0, !huKnown.contains(lower(wOut)), z.zipfHu < z.zipfEn {
                res.replaceSubrange(r, with: wIn)
            }
        }
        return res
    }
}
