/// Text is handled as arrays of Unicode scalars (= Python code points), never as Swift Characters:
/// the model works per code point, and Python's string indices (Word.start/end) are code points.
typealias Scalars = [UInt32]

extension Unicode.Scalar {
    /// Python `str.isalpha()`: general category L*.
    var isPyAlpha: Bool {
        switch properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter: true
        default: false
        }
    }

    /// Python regex `\w`: alphanumeric (incl. all numerics) or underscore.
    var isPyWord: Bool { isPyAlpha || properties.numericType != nil || self == "_" }

    /// Python regex `[^\W\d_]`: a word character that is not a decimal digit or underscore.
    var isPyLetter: Bool { isPyWord && properties.numericType != .decimal && self != "_" }
}

@inline(__always) func scalar(_ v: UInt32) -> Unicode.Scalar { Unicode.Scalar(v) ?? "\u{FFFD}" }

func scalars(_ s: String) -> Scalars { s.unicodeScalars.map(\.value) }

func string(_ s: some Sequence<UInt32>) -> String {
    var out = String.UnicodeScalarView()
    out.append(contentsOf: s.map(scalar))
    return String(out)
}

/// Python `str.lower()` (full case mapping, may change length).
func lower(_ s: some Sequence<UInt32>) -> Scalars {
    s.flatMap { scalars(scalar($0).properties.lowercaseMapping) }
}

func isUpper(_ v: UInt32) -> Bool { scalar(v).properties.isUppercase }

/// Single-scalar uppercase; keeps the character when the mapping would change length (ß -> SS),
/// so that text length never changes.
func upper(_ v: UInt32) -> UInt32 {
    let m = scalar(v).properties.uppercaseMapping.unicodeScalars
    return m.count == 1 ? m.first!.value : v
}

/// Ranges of `TOKEN_RE = (?<![\w-])[^\W\d_]+(?:-[^\W\d_]+)*(?![\w-])` (tohu/lexicon.py):
/// whole words incl. hyphenated compounds ("deadline-ra"), but not "2010-től".
func tokenRanges(_ t: Scalars) -> [Range<Int>] {
    let hyphen = UInt32(UInt8(ascii: "-"))
    func letter(_ i: Int) -> Bool { i < t.count && scalar(t[i]).isPyLetter }
    func wordOrHyphen(_ i: Int) -> Bool { i >= 0 && i < t.count && (t[i] == hyphen || scalar(t[i]).isPyWord) }

    var out: [Range<Int>] = []
    var i = 0
    while i < t.count {
        guard letter(i), !wordOrHyphen(i - 1) else { i += 1; continue }
        var e = i
        while letter(e) { e += 1 }
        while e < t.count, t[e] == hyphen, letter(e + 1) {
            e += 1
            while letter(e) { e += 1 }
        }
        // Backtracking can only end the match in front of a letter or a hyphen, so if the greedy
        // match fails the lookahead, no match starts at i (and none can start inside it).
        if !wordOrHyphen(e) {
            out.append(i..<e)
            i = e
        } else {
            i += 1
        }
    }
    return out
}

/// Ranges of `WORD_RE = [^\W\d_]+` (tohu/restore.py): plain letter runs.
func letterRuns(_ t: Scalars) -> [Range<Int>] {
    var out: [Range<Int>] = []
    var i = 0
    while i < t.count {
        guard scalar(t[i]).isPyLetter else { i += 1; continue }
        var e = i
        while e < t.count, scalar(t[e]).isPyLetter { e += 1 }
        out.append(i..<e)
        i = e
    }
    return out
}
