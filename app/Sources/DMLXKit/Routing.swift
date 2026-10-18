import Foundation

/// Decides what a composer message does, without an LLM (spec §11.3).
public enum Intent: Equatable, Sendable {
    case none
    case generate
    case edit(sourceIds: [String], fromTurn: String?)
    case video  // not in the beta; the UI explains instead of routing
}

public struct RoutingContext: Sendable {
    /// The image the user explicitly targeted with "Edit" (result id + turn id).
    public var targetResultId: String?
    public var targetTurnId: String?
    /// Latest finished image in the thread.
    public var latestResultId: String?
    public var latestTurnId: String?
    /// Images attached to the composer (uploads or "Use as reference").
    public var attachmentIds: [String]

    public init(
        targetResultId: String? = nil, targetTurnId: String? = nil, latestResultId: String? = nil,
        latestTurnId: String? = nil, attachmentIds: [String] = []
    ) {
        self.targetResultId = targetResultId
        self.targetTurnId = targetTurnId
        self.latestResultId = latestResultId
        self.latestTurnId = latestTurnId
        self.attachmentIds = attachmentIds
    }
}

public enum Router {
    static let editLead = try! NSRegularExpression(
        pattern: #"^(make|add|remove|change|turn|keep|now|replace|put|give|darker|brighter|more|less|without|with|swap|fix|move|erase|delete|recolor|colou?r|zoom|crop|extend|try)\b"#,
        options: [.caseInsensitive])
    static let videoWords = try! NSRegularExpression(
        pattern: #"\b(animate|animated|animation|video|clip|motion)\b"#, options: [.caseInsensitive])
    static let freshLead = try! NSRegularExpression(
        pattern: #"^(new|another|generate|create|draw|paint|imagine)\b"#, options: [.caseInsensitive])

    static func matches(_ re: NSRegularExpression, _ text: String) -> Bool {
        re.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil
    }

    public static func route(_ raw: String, _ ctx: RoutingContext) -> Intent {
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return .none }
        if matches(videoWords, text) { return .video }

        // Explicit target or attachments always mean "edit these".
        if let target = ctx.targetResultId {
            return .edit(sourceIds: [target] + ctx.attachmentIds.filter { $0 != target }, fromTurn: ctx.targetTurnId)
        }
        if !ctx.attachmentIds.isEmpty {
            return .edit(sourceIds: ctx.attachmentIds, fromTurn: nil)
        }
        // A follow-up phrased as a change edits the latest image, unless it asks for something new.
        if let latest = ctx.latestResultId, matches(editLead, text), !matches(freshLead, text) {
            return .edit(sourceIds: [latest], fromTurn: ctx.latestTurnId)
        }
        return .generate
    }
}
