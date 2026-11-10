import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Sizes on disk and in downloads, in decimal GB (bytes / 1e9) everywhere in the app (task 19): the engine
/// reports `size_gb` the same way, so a model card never shows two different numbers for one model.
public enum DiskSize {
    public static func gb(bytes: Int) -> Double { Double(bytes) / 1e9 }

    /// "4.3 GB".
    public static func text(gb: Double) -> String { String(format: "%.1f GB", gb) }

    /// An amount downloaded so far: "1.0 GB", or in MB below 0.1 GB, where one decimal of a GB would read
    /// "0.0 GB" (test round 3 of task 19: a 40 MB partial read "Resume download · 0.0 of 4.6 GB").
    public static func amount(bytes: Int) -> String {
        if bytes >= 100_000_000 { return text(gb: gb(bytes: bytes)) }
        if bytes <= 0 { return "0 MB" }
        return String(format: "%.0f MB", max(1, (Double(bytes) / 1e6).rounded()))
    }

    /// "1.0 of 4.6 GB", or "40 MB of 4.6 GB" for less than 0.1 GB.
    public static func progress(done: Int, total: Int) -> String {
        if done < 100_000_000 { return "\(amount(bytes: done)) of " + text(gb: gb(bytes: total)) }
        return String(format: "%.1f of %.1f GB", gb(bytes: done), gb(bytes: total))
    }

    /// The button on a model whose download stopped part way: "Resume download · 1.0 of 4.6 GB", or without
    /// the total when it is not known.
    public static func resumeLabel(partialBytes: Int, totalBytes: Int?) -> String {
        guard let total = totalBytes, total > 0 else {
            return "Resume download · \(amount(bytes: partialBytes)) done"
        }
        return "Resume download · " + progress(done: min(partialBytes, total), total: total)
    }

    /// The toast after Remove: "Removed X · freed 4.6 GB", "· freed 12 MB" below 0.1 GB (fix round 4 of
    /// task 19: trashing a 12 MB partial read "freed 0.0 GB"), and no amount when nothing was freed.
    public static func removedMessage(name: String, freedGB: Double) -> String {
        let bytes = Int((freedGB * 1e9).rounded())
        return bytes > 0 ? "Removed \(name) · freed \(amount(bytes: bytes))" : "Removed \(name)"
    }
}

/// How many dropped or picked files can still be attached to the message (task 19).
public struct AttachPlan: Equatable, Sendable {
    public var take: Int
    public var skipped: Int

    /// `current` attachments and `uploading` ones in flight both hold a place.
    public init(files: Int, current: Int, uploading: Int, limit: Int) {
        let free = max(0, limit - current - uploading)
        take = min(files, free)
        skipped = files - take
    }

    public init(files: Int, current: Int, uploading: Int, limit: AttachLimit) {
        self.init(files: files, current: current, uploading: uploading, limit: limit.count)
    }

    /// What the toast says when some files were left out, nil when all fit.
    public func message(_ limit: AttachLimit) -> String? {
        guard skipped > 0 else { return nil }
        if take == 0 { return "Not attached: \(limit.reason)." }
        return "\(skipped) file\(skipped == 1 ? " was" : "s were") not attached: \(limit.reason)."
    }
}

/// How many images one message can carry: the edit model's own limit (`max_references` in the engine's
/// model list), never more than the app's 3. The toast names whichever limit applies, so it doesn't say
/// "this model accepts up to 3" for Qwen Image 2.1, which accepts 10 (task 19).
public struct AttachLimit: Equatable, Sendable {
    public static let app = 3

    public var count: Int
    public var reason: String

    /// `model` is the edit model the message will most likely use (its name and `max_references`), nil when
    /// none is known yet.
    public init(modelName: String?, modelMax: Int?) {
        if let name = modelName, let max = modelMax, max > 0, max <= Self.app {
            count = max
            reason = "\(name) accepts \(Self.images(max))"
        } else {
            count = Self.app
            reason = "dMLX attaches \(Self.images(Self.app)) to a message"
        }
    }

    static func images(_ n: Int) -> String { n == 1 ? "1 image" : "up to \(n) images" }
}

/// Files the engine can't read as they are, converted on this Mac before upload (task 19): iPhone photos
/// (HEIC/HEIF) become PNG with ImageIO, upright, at full size.
public enum UploadImage {
    /// Whether the file must be converted before upload: HEIC/HEIF (by its type, or its extension when the
    /// type is unknown), or an RGB image in another colour space than sRGB, such as an iPhone's Display P3
    /// JPEG. The engine reads pixels as sRGB, so a P3 photo would edit with muted colours (task 19).
    public static func needsConversion(_ url: URL) -> Bool {
        let ext = url.pathExtension.lowercased()
        if let type = UTType(filenameExtension: ext) {
            if type.conforms(to: .heic) || type.conforms(to: .heif) { return true }
        } else if ["heic", "heif", "hif"].contains(ext) {
            return true
        }
        return hasWideColourProfile(url)
    }

    /// An RGB image whose embedded profile is not sRGB (only the file's header is read).
    static func hasWideColourProfile(_ url: URL) -> Bool {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              (props[kCGImagePropertyColorModel] as? String) == (kCGImagePropertyColorModelRGB as String),
              let profile = props[kCGImagePropertyProfileName] as? String else { return false }
        return !isSRGB(profileName: profile)
    }

    /// "sRGB IEC61966-2.1", "sRGB built-in" and the like.
    public static func isSRGB(profileName: String) -> Bool {
        profileName.lowercased().hasPrefix("srgb")
    }

    public enum ConversionError: LocalizedError {
        case unreadable(String)
        public var errorDescription: String? {
            switch self {
            case .unreadable(let name): "\(name) could not be read as an image."
            }
        }
    }

    /// A PNG copy of the photo in the temporary folder. The caller deletes it after the upload.
    public static func convertToPNG(_ url: URL) throws -> URL {
        let name = url.lastPathComponent
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = props[kCGImagePropertyPixelWidth] as? Int,
              let height = props[kCGImagePropertyPixelHeight] as? Int else {
            throw ConversionError.unreadable(name)
        }
        // The thumbnail call at full size applies the photo's orientation, so a portrait iPhone shot stays upright.
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: max(width, height),
        ]
        guard let upright = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary),
              let image = toSRGB(upright) else {
            throw ConversionError.unreadable(name)
        }
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("dmlx-upload-\(UUID().uuidString)")
            .appendingPathComponent(url.deletingPathExtension().lastPathComponent + ".png")
        try FileManager.default.createDirectory(at: out.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard let dest = CGImageDestinationCreateWithURL(out as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw ConversionError.unreadable(name)
        }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw ConversionError.unreadable(name) }
        return out
    }

    /// The image drawn into 8-bit sRGB (the colour space the models expect), so a Display P3 or 10-bit
    /// HDR photo keeps its look. Transparency is kept.
    static func toSRGB(_ image: CGImage) -> CGImage? {
        guard let srgb = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        let alpha: CGImageAlphaInfo
        switch image.alphaInfo {
        case .none, .noneSkipFirst, .noneSkipLast: alpha = .noneSkipLast
        default: alpha = .premultipliedLast
        }
        guard let context = CGContext(data: nil, width: image.width, height: image.height, bitsPerComponent: 8,
                                      bytesPerRow: 0, space: srgb, bitmapInfo: alpha.rawValue) else { return nil }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        return context.makeImage()
    }

    /// The toast for a failed upload: the engine's message names the file (task 19), so it is not named twice.
    public static func failureMessage(file: String, error: String) -> String {
        error.contains(file) ? "Couldn't attach: \(error)" : "Couldn't attach \(file): \(error)"
    }
}

/// When the first-run sheet opens by itself (task 19): once per app launch, the first time the engine is
/// ready, while there is no image model (or setup was never finished and the agent is missing). An engine
/// restart does not open it again; sending with no model still opens the setup step.
public enum OnboardingRule {
    public static func opensByItself(
        alreadyOfferedThisLaunch: Bool, hasImageModel: Bool, setupCompleted: Bool, agentInstalled: Bool
    ) -> Bool {
        guard !alreadyOfferedThisLaunch else { return false }
        return !hasImageModel || (!setupCompleted && !agentInstalled)
    }
}

/// The setup sheet's height on small screens (task 19): the sheet hangs from the window's title bar, so it
/// must end above the bottom of the visible screen (the Dock). The middle scrolls; title and buttons stay.
public enum SheetFit {
    /// The tallest the sheet can be: from the top of the window's content down to the visible screen's
    /// bottom, less a margin.
    public static func maxSheetHeight(contentTop: CGFloat, visibleBottom: CGFloat, margin: CGFloat = 12) -> CGFloat {
        max(0, contentTop - visibleBottom - margin)
    }

    /// The scrolling middle's height: what it wants, but no more than the room the fixed parts leave,
    /// and never less than a usable minimum.
    public static func choicesHeight(wanted: CGFloat, maxSheet: CGFloat, fixed: CGFloat, minimum: CGFloat = 140) -> CGFloat {
        max(minimum, min(wanted, maxSheet - fixed))
    }
}

/// What a toast says when one drop gives several notes (task 19): one toast slot, so a later note must not
/// hide an earlier one ("2 files were not attached" and "Couldn't attach: notes.txt …").
public enum ToastText {
    public static func merged(_ earlier: String?, _ later: String) -> String {
        guard let earlier, !earlier.isEmpty, earlier != later else { return later }
        return earlier + " " + later
    }

    /// Long enough to read: 3 s, plus about a second for every 15 characters past 45, at most 10 s.
    public static func seconds(for message: String) -> Double {
        min(10, max(3, 3 + Double(message.count - 45) / 15))
    }
}

/// The question before quitting during a download (task 19): image models and the agent model alike.
public enum QuitQuestion {
    public static let title = "A download is running. Quit anyway?"

    /// nil when nothing is downloading.
    public static func detail(imageDownloads: [String], agentDownload: String?) -> String? {
        let names = imageDownloads + (agentDownload.map { [$0] } ?? [])
        switch names.count {
        case 0: return nil
        case 1: return "\(names[0]) stops now. Download it again and it resumes where it stopped."
        default: return "\(names.count) downloads stop now. Download them again and they resume where they stopped."
        }
    }
}

/// What the composer keeps when a sent message goes out (task 19). A message can be planned for seconds
/// (the agent, the memory check); images attached meanwhile, or still attaching, are for the next message,
/// so only the images that went with the sent one leave the composer.
public enum SentAttachments {
    public static func kept<T: Identifiable>(_ current: [T], sent: [T.ID]) -> [T] {
        let gone = Set(sent)
        return current.filter { !gone.contains($0.id) }
    }
}

/// The model named in the composer's route hint (task 19). When sending would load it in place of another
/// model, the hint says so with the memory it needs (the engine's `memory_gb`), never its download size.
public enum LoadHint {
    public static func text(name: String, loads: Bool, memoryGB: Double?) -> String {
        guard loads else { return name }
        guard let gb = memoryGB, gb > 0 else { return "loads \(name)" }
        return "loads \(name) · about \(String(format: "%.1f", gb)) GB"
    }
}

/// One quit question at a time (task 19): a second quit from the Dock, ⌘Q or a logout while "A download is
/// running. Quit anyway?" is open does not stack a second, identical alert on it.
public struct QuitGate: Sendable {
    public enum Step: Equatable, Sendable { case quit, ask, ignore }

    public private(set) var asking = false
    public private(set) var confirmed = false

    public init() {}

    /// What a quit request does: quit at once (nothing downloading, or already agreed), ask, or nothing
    /// because the question is already open.
    public mutating func request(downloading: Bool) -> Step {
        if confirmed || !downloading { return .quit }
        if asking { return .ignore }
        asking = true
        return .ask
    }

    public mutating func answered(quit: Bool) {
        asking = false
        confirmed = quit
    }
}

/// A message sent while images upload waits for them (task 19), but not forever: a hung engine would hold
/// it with no way out. The upload gives up after this long, and Cancel ends the wait at once.
public enum UploadWait {
    public static let timeout: TimeInterval = 120

    /// The composer's hint while images are still attaching: the route is decided once they are there,
    /// so it must not say "Will generate" for a message that will edit them.
    public static func hint(attaching: Int) -> String {
        attaching == 1 ? "Attaching an image · decided when it is attached"
            : "Attaching \(attaching) images · decided when they are attached"
    }
}
