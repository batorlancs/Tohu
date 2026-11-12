import Foundation

/// Memory-mapped word-form lexicon (models/export/lexicon.bin, see scripts/export_coreml.py).
/// Unaccented key -> accented forms with corpus counts; binary search over sorted keys,
/// so loading is instant and the 40 MB file stays in the page cache, not the heap.
final class Lexicon: @unchecked Sendable {
    private let data: Data
    let count: Int
    private let blobStart: Int

    init(url: URL) throws {
        data = try Data(contentsOf: url, options: .alwaysMapped)
        guard data.count >= 12, data.prefix(8) == Data("TOHULEX1".utf8) else {
            throw TohuError.badResource("lexicon.bin: bad header")
        }
        count = Int(data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 8, as: UInt32.self).littleEndian })
        blobStart = 12 + 4 * (count + 1)
        guard data.count >= blobStart else { throw TohuError.badResource("lexicon.bin: truncated") }
    }

    /// Accented forms (lowercase) and their counts for an unaccented lowercase key.
    func forms(key: Scalars) -> [(form: Scalars, count: Int)]? {
        let k = Array(string(key).utf8)
        return data.withUnsafeBytes { (buf: UnsafeRawBufferPointer) -> [(Scalars, Int)]? in
            func offset(_ i: Int) -> Int {
                Int(buf.loadUnaligned(fromByteOffset: 12 + 4 * i, as: UInt32.self).littleEndian) + blobStart
            }
            func line(_ i: Int) -> UnsafeRawBufferPointer.SubSequence { buf[offset(i)..<offset(i + 1)] }
            func key(_ l: UnsafeRawBufferPointer.SubSequence) -> UnsafeRawBufferPointer.SubSequence {
                l.prefix { $0 != 0x09 }
            }
            var lo = 0, hi = count
            while lo < hi {
                let mid = (lo + hi) / 2
                let mk = key(line(mid))
                if mk.lexicographicallyPrecedes(k) {
                    lo = mid + 1
                } else if k.lexicographicallyPrecedes(mk) {
                    hi = mid
                } else {
                    let l = line(mid)
                    let rest = String(decoding: l.dropFirst(mk.count + 1), as: UTF8.self)
                    return rest.split(separator: " ").compactMap { item in
                        guard let colon = item.lastIndex(of: ":"), let c = Int(item[item.index(after: colon)...])
                        else { return nil }
                        return (scalars(String(item[..<colon])), c)
                    }
                }
            }
            return nil
        }
    }
}
