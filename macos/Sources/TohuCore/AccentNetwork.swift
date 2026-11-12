import CoreML
import Foundation

/// The Core ML character model: ids (1, T) int32 -> logits (1, T, 4), T one of the exported buckets.
/// Long texts are split into overlapping windows exactly like `Restorer._logits` in Python.
final class AccentNetwork: @unchecked Sendable {
    private let model: MLModel
    private let buckets: [Int]
    private let chunk: Int, ctx: Int
    private let lock = NSLock()

    init(url: URL, manifest: Manifest, computeUnits: MLComputeUnits) throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        model = try MLModel(contentsOf: url, configuration: config)
        buckets = manifest.buckets.sorted()
        chunk = manifest.chunk
        ctx = manifest.ctx
        precondition(chunk + 2 * ctx <= buckets.last!, "windows must fit the largest bucket")
    }

    /// Per-character logits, flat (t.count * 4).
    func logits(_ t: Scalars, alphabet: Alphabet) throws -> [Float] {
        let k = Alphabet.numClasses
        var out = [Float](repeating: 0, count: t.count * k)
        let space = UInt32(UInt8(ascii: " "))
        var pos = 0
        while pos < t.count {
            var end = min(t.count, pos + chunk)
            if end < t.count, let sp = (pos + chunk / 2..<end).last(where: { t[$0] == space }), sp > 0 {
                end = sp
            }
            let ws = max(0, pos - ctx), we = min(t.count, end + ctx)
            let y = try window(alphabet.encode(t[ws..<we]))
            for i in pos..<end {
                for c in 0..<k { out[i * k + c] = y[(i - ws) * k + c] }
            }
            pos = end
        }
        return out
    }

    private func window(_ ids: [Int32]) throws -> [Float] {
        let k = Alphabet.numClasses
        let bucket = buckets.first { $0 >= ids.count }!
        let input = try MLMultiArray(shape: [1, NSNumber(value: bucket)], dataType: .int32)
        input.withUnsafeMutableBufferPointer(ofType: Int32.self) { buf, _ in
            buf.initialize(repeating: Alphabet.pad)
            for (i, id) in ids.enumerated() { buf[i] = id }
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: ["ids": MLFeatureValue(multiArray: input)])
        let result = try lock.withLock { try model.prediction(from: provider) }
        guard let y = result.featureValue(for: "logits")?.multiArrayValue else {
            throw TohuError.badResource("model has no 'logits' output")
        }
        let s1 = y.strides[1].intValue, s2 = y.strides[2].intValue
        var out = [Float](repeating: 0, count: ids.count * k)
        switch y.dataType {
        case .float32:
            y.withUnsafeBufferPointer(ofType: Float.self) { buf in
                for i in 0..<ids.count { for c in 0..<k { out[i * k + c] = buf[i * s1 + c * s2] } }
            }
        default:
            for i in 0..<ids.count {
                for c in 0..<k { out[i * k + c] = y[[0, NSNumber(value: i), NSNumber(value: c)]].floatValue }
            }
        }
        return out
    }
}
