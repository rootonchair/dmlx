import DMLXKit
import Foundation
import Observation
import SwiftUI
import TipKit

/// The in-app guide: short tips anchored to the controls they explain, each shown once.
///  1. After the first-run setup closes: the message box, attaching images, size, new image or edit,
///     and chats (one at a time, with Next).
///  2. After the first image: versions and Variations, then Crop / Draw.
///  3. The first time the Crop / Draw editor opens: its tools (inline in the editor).
/// Help → Show Tips Again starts over: every tip gets a new id for the new round.
@MainActor
@Observable
final class Guide {
    static let shared = Guide()

    private static let roundKey = "guideRound"
    private(set) var round = UserDefaults.standard.integer(forKey: Guide.roundKey)
    private(set) var chat = Guide.chatTips(round: 0)
    private(set) var image = Guide.imageTips(round: 0)
    private(set) var editor = EditorTip(round: 0)

    private init() { rebuild() }

    /// Called once at launch. Tip state lives in the data folder, so DMLX_HOME keeps tests apart.
    static func configure() {
        let folder = Store.dataDirectory().appendingPathComponent("tips", isDirectory: true)
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        if ProcessInfo.processInfo.environment["DMLX_RESET_TIPS"] != nil { try? Tips.resetDatastore() }
        try? Tips.configure([.datastoreLocation(.url(folder)), .displayFrequency(.immediate)])
    }

    /// The chat tour waits until the setup is closed and an image model is ready.
    func setReady(_ ready: Bool) {
        if GuideState.ready != ready { GuideState.ready = ready }
    }

    /// The image tips wait for the first finished image.
    func imageShown() {
        if !GuideState.hasImage { GuideState.hasImage = true }
    }

    /// The current tip is done: show the next one in its group.
    func next(_ tip: any Tip) { tip.invalidate(reason: .actionPerformed) }

    func showAgain() {
        round += 1
        UserDefaults.standard.set(round, forKey: Self.roundKey)
        rebuild()
    }

    private func rebuild() {
        chat = Self.chatTips(round: round)
        image = Self.imageTips(round: round)
        editor = EditorTip(round: round)
    }

    private static func chatTips(round: Int) -> TipGroup {
        TipGroup(.ordered) {
            ComposerTip(round: round)
            AttachTip(round: round)
            SizeTip(round: round)
            ModeTip(round: round)
            ChatsTip(round: round)
        }
    }

    private static func imageTips(round: Int) -> TipGroup {
        TipGroup(.ordered) {
            VersionsTip(round: round)
            CropDrawTip(round: round)
        }
    }
}

/// Rule parameters shared by the tips (TipKit keeps their values).
enum GuideState {
    @Parameter static var ready: Bool = false
    @Parameter static var hasImage: Bool = false
}

// MARK: the tips

private let next = Tips.Action(id: "next", title: "Next")
private let done = Tips.Action(id: "done", title: "Got It")

struct ComposerTip: Tip {
    let round: Int
    var id: String { "composer-\(round)" }
    var title: Text { Text("Describe an image") }
    var message: Text? {
        Text("Write what you want to see and press Return. Then reply in plain words to change it: “make the boat red”, “the first one, but foggy”. The agent reads the chat and finds the image you mean.")
    }
    var image: Image? { Image(systemName: "text.bubble") }
    var rules: [Rule] { [#Rule(GuideState.$ready) { $0 == true }] }
    var actions: [Action] { [next] }
}

struct AttachTip: Tip {
    let round: Int
    var id: String { "attach-\(round)" }
    var title: Text { Text("Use your own images") }
    var message: Text? {
        Text("Attach up to 3 images from your Mac, or drop them on the message box, to edit them or use them as a reference: “a lighthouse painted like this”.")
    }
    var image: Image? { Image(systemName: "paperclip") }
    var actions: [Action] { [next] }
}

struct SizeTip: Tip {
    let round: Int
    var id: String { "size-\(round)" }
    var title: Text { Text("Shape and size") }
    var message: Text? {
        Text("Choose square, portrait or landscape for new images. An edit keeps the size of the image it changes.")
    }
    var image: Image? { Image(systemName: "aspectratio") }
    var actions: [Action] { [next] }
}

struct ModeTip: Tip {
    let round: Int
    var id: String { "mode-\(round)" }
    var title: Text { Text("New image or edit") }
    var message: Text? {
        Text("The agent decides whether to make a new image or edit one, and on Auto it picks the model. After you type, click here to choose “Create new” or “Edit current” yourself. The menu beside it picks the model for new images and for edits.")
    }
    var image: Image? { Image(systemName: "wand.and.stars") }
    var actions: [Action] { [next] }
}

struct ChatsTip: Tip {
    let round: Int
    var id: String { "chats-\(round)" }
    var title: Text { Text("One chat per idea") }
    var message: Text? {
        Text("A chat keeps every image and version, and the agent can look back through it. Start a new creation for something unrelated; your chats stay under Recent.")
    }
    var image: Image? { Image(systemName: "bubble.left.and.bubble.right") }
    var actions: [Action] { [done] }
}

struct VersionsTip: Tip {
    let round: Int
    var id: String { "versions-\(round)" }
    var title: Text { Text("Every result is kept") }
    var message: Text? {
        Text("Variations makes another take with a new seed. Each take is a version: switch between them with v1, v2… The download button saves the image to Pictures/dMLX.")
    }
    var image: Image? { Image(systemName: "square.stack") }
    var rules: [Rule] { [#Rule(GuideState.$hasImage) { $0 == true }] }
    var actions: [Action] { [next] }
}

struct CropDrawTip: Tip {
    let round: Int
    var id: String { "cropdraw-\(round)" }
    var title: Text { Text("Point at what to change") }
    var message: Text? {
        Text("Crop / Draw lets you crop to an area or circle a part of the image. Then attach it and say what to do.")
    }
    var image: Image? { Image(systemName: "pencil.and.outline") }
    var actions: [Action] { [done] }
}

struct EditorTip: Tip {
    let round: Int
    var id: String { "editor-\(round)" }
    var title: Text { Text("Crop, Paint and Erase") }
    var message: Text? {
        Text("Crop keeps only the area you care about. Paint marks what to change. Erase removes a wrong mark, never the image. Then click Attach edited image and say what to do, for example “remove the cup I circled, and the red marks”.")
    }
    var image: Image? { Image(systemName: "paintbrush.pointed") }
}

// MARK: anchoring

extension View {
    /// Shows `T` here when it is the current tip of `group`; its Next / Got It button moves the group on.
    /// `edge`: the side of this control the tip opens on (macOS flips it when there is no room).
    func guideTip<T: Tip>(_ type: T.Type, in group: TipGroup?, edge: Edge = .top) -> some View {
        let tip = group?.currentTip as? T
        return popoverTip(tip, arrowEdge: edge) { _ in
            if let tip { Guide.shared.next(tip) }
        }
    }
}
