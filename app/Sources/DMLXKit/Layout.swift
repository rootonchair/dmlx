import Foundation

/// One region of an image's layout (task 17). The user owns where it is, its name or exact text, and any
/// description they wrote; the agent fills what they left empty. Boxes the user placed or changed are
/// `byUser` and stay put when the agent rewrites the caption on later turns.
///
/// `box` is [y_min, x_min, y_max, x_max] in 0...1000 from the top left, the engine's form. The app never
/// shows or sends the model's JSON; it only exchanges these boxes.
public struct LayoutBox: Codable, Hashable, Identifiable, Sendable {
    public enum Kind: String, Codable, Sendable { case obj, text }

    public var id: String
    public var kind: Kind
    public var label: String
    public var desc: String
    public var descByUser: Bool
    public var box: [Int]
    public var byUser: Bool

    enum CodingKeys: String, CodingKey {
        case id, kind, label, desc, box
        case descByUser = "desc_by_user"
        case byUser = "by_user"
    }

    public init(
        id: String = "b" + UUID().uuidString.prefix(8).lowercased(), kind: Kind = .obj, label: String = "",
        desc: String = "", descByUser: Bool = false, rect: NormalizedRect, byUser: Bool = true
    ) {
        self.id = id
        self.kind = kind
        self.label = label
        self.desc = desc
        self.descByUser = descByUser
        self.box = Self.box(from: rect)
        self.byUser = byUser
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        kind = (try? c.decode(Kind.self, forKey: .kind)) ?? .obj
        label = (try? c.decode(String.self, forKey: .label)) ?? ""
        desc = (try? c.decodeIfPresent(String.self, forKey: .desc)) ?? ""
        descByUser = (try? c.decodeIfPresent(Bool.self, forKey: .descByUser)) ?? false
        let raw = try c.decode([Int].self, forKey: .box)
        box = raw.count == 4 ? raw.map { min(max($0, 0), 1000) } : [0, 0, 1000, 1000]
        byUser = (try? c.decodeIfPresent(Bool.self, forKey: .byUser)) ?? false
    }

    /// The box as a rectangle in 0...1 of the image, for drawing.
    public var rect: NormalizedRect {
        NormalizedRect(
            x: Double(box[1]) / 1000, y: Double(box[0]) / 1000,
            width: Double(box[3] - box[1]) / 1000, height: Double(box[2] - box[0]) / 1000)
    }

    /// Moving or resizing a box clears a description the agent wrote: it may say where the box was
    /// ("below the title"), and the model follows words over boxes. The user's own description stays.
    public mutating func place(_ rect: NormalizedRect) {
        let moved = Self.box(from: rect)
        if moved != box && !descByUser { desc = "" }
        box = moved
        byUser = true
    }

    /// A box the agent will finish: the user has named it but not described it.
    public var needsAgentDescription: Bool { !descByUser }

    static func box(from r: NormalizedRect) -> [Int] {
        [Int((r.y * 1000).rounded()), Int((r.x * 1000).rounded()),
         Int(((r.y + r.height) * 1000).rounded()), Int(((r.x + r.width) * 1000).rounded())]
    }

    public var jsonObject: [String: Any] {
        ["id": id, "kind": kind.rawValue, "label": label, "desc": desc, "desc_by_user": descByUser,
         "box": box, "by_user": byUser]
    }

    /// Whether two layouts put the same boxes in the same places (names and descriptions aside).
    public static func sameGeometry(_ a: [LayoutBox], _ b: [LayoutBox]) -> Bool {
        Dictionary(a.map { ($0.id, $0.box) }, uniquingKeysWith: { x, _ in x })
            == Dictionary(b.map { ($0.id, $0.box) }, uniquingKeysWith: { x, _ in x })
    }

    /// A user's boxes, ready to send: empty names are dropped (a text box needs its words), and a
    /// description only counts as the user's when it is not empty.
    public static func sendable(_ boxes: [LayoutBox]) -> [LayoutBox] {
        boxes.compactMap { b in
            var b = b
            b.label = b.label.trimmingCharacters(in: .whitespacesAndNewlines)
            b.desc = b.desc.trimmingCharacters(in: .whitespacesAndNewlines)
            b.descByUser = b.descByUser && !b.desc.isEmpty
            if b.kind == .text && b.label.isEmpty { return nil }
            return b
        }
    }
}

/// The FIBO place word for a box: FIBO places things by words, so the editor shows what it will get.
public enum LayoutWords {
    public static func location(_ b: LayoutBox) -> String {
        let cy = Double(b.box[0] + b.box[2]) / 2, cx = Double(b.box[1] + b.box[3]) / 2
        let row = cy < 333 ? "top" : cy > 667 ? "bottom" : ""
        let col = cx < 333 ? "left" : cx > 667 ? "right" : ""
        let word = [row, col].filter { !$0.isEmpty }.joined(separator: " ")
        return word.isEmpty ? "center" : word
    }
}
