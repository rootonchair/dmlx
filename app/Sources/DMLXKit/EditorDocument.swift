import AppKit
import CoreGraphics
import Foundation

public struct NormalizedPoint: Codable, Hashable, Sendable {
    public var x: Double
    public var y: Double

    public init(x: Double, y: Double) {
        self.x = min(max(x, 0), 1)
        self.y = min(max(y, 0), 1)
    }
}

public struct NormalizedRect: Codable, Hashable, Sendable {
    public var x: Double
    public var y: Double
    public var width: Double
    public var height: Double

    public init(x: Double = 0, y: Double = 0, width: Double = 1, height: Double = 1) {
        let left = min(max(x, 0), 1)
        let top = min(max(y, 0), 1)
        self.x = left
        self.y = top
        self.width = min(max(width, 0.001), 1 - left)
        self.height = min(max(height, 0.001), 1 - top)
    }

    public static let full = NormalizedRect()
}

public struct PaintColor: Codable, Hashable, Sendable {
    public var red: Double
    public var green: Double
    public var blue: Double
    public var alpha: Double

    public init(red: Double, green: Double, blue: Double, alpha: Double = 1) {
        self.red = min(max(red, 0), 1)
        self.green = min(max(green, 0), 1)
        self.blue = min(max(blue, 0), 1)
        self.alpha = min(max(alpha, 0), 1)
    }
}

public struct PaintStroke: Codable, Identifiable, Hashable, Sendable {
    public var id: String
    public var points: [NormalizedPoint]
    /// Width as a fraction of the shorter original image edge.
    public var width: Double
    public var color: PaintColor
    public var erases: Bool

    public init(
        id: String = UUID().uuidString, points: [NormalizedPoint] = [], width: Double = 0.02,
        color: PaintColor = PaintColor(red: 1, green: 0.2, blue: 0.1), erases: Bool = false
    ) {
        self.id = id
        self.points = points
        self.width = min(max(width, 0.001), 0.25)
        self.color = color
        self.erases = erases
    }
}

/// Non-destructive local edit state. Coordinates always refer to the original,
/// orientation-normalized image, so crop and paint remain aligned at any view scale.
public struct EditorDocument: Codable, Hashable, Sendable {
    public var schemaVersion: Int
    public var baseResultId: String
    public var baseWidth: Int
    public var baseHeight: Int
    public var crop: NormalizedRect
    public var strokes: [PaintStroke]
    public var updated: Date

    public init(
        baseResultId: String, baseWidth: Int, baseHeight: Int, crop: NormalizedRect = .full,
        strokes: [PaintStroke] = [], updated: Date = .now, schemaVersion: Int = 1
    ) {
        self.schemaVersion = schemaVersion
        self.baseResultId = baseResultId
        self.baseWidth = max(baseWidth, 1)
        self.baseHeight = max(baseHeight, 1)
        self.crop = crop
        self.strokes = strokes
        self.updated = updated
    }
}

public enum EditorRenderError: Error, LocalizedError {
    case cannotCreateImage

    public var errorDescription: String? { "The edited image could not be rendered." }
}

/// Shared geometry for the canvas and renderer. Values are expressed in the
/// caller's coordinate space, so the mapping is invariant across Retina scale.
public enum EditorGeometry {
    public static func fittedRect(imageWidth: Int, imageHeight: Int, in size: CGSize) -> CGRect {
        let aspect = CGFloat(max(imageWidth, 1)) / CGFloat(max(imageHeight, 1))
        var width = size.width
        var height = width / aspect
        if height > size.height { height = size.height; width = height * aspect }
        return CGRect(
            x: (size.width - width) / 2, y: (size.height - height) / 2,
            width: width, height: height)
    }

    public static func normalize(_ point: CGPoint, in imageRect: CGRect) -> NormalizedPoint? {
        guard imageRect.width > 0, imageRect.height > 0, imageRect.contains(point) else { return nil }
        return NormalizedPoint(
            x: (point.x - imageRect.minX) / imageRect.width,
            y: (point.y - imageRect.minY) / imageRect.height)
    }

    public static func pixelCrop(_ crop: NormalizedRect, sourceWidth: Int, sourceHeight: Int) -> CGRect {
        CGRect(
            x: round(crop.x * Double(sourceWidth)),
            y: round(crop.y * Double(sourceHeight)),
            width: max(1, round(crop.width * Double(sourceWidth))),
            height: max(1, round(crop.height * Double(sourceHeight))))
            .intersection(CGRect(x: 0, y: 0, width: sourceWidth, height: sourceHeight))
    }
}

public enum EditorRenderer {
    public static func render(_ document: EditorDocument, source: CGImage) throws -> Data {
        let sourceWidth = source.width
        let sourceHeight = source.height
        let crop = document.crop
        let pixelCrop = EditorGeometry.pixelCrop(crop, sourceWidth: sourceWidth, sourceHeight: sourceHeight)
        guard let cropped = source.cropping(to: pixelCrop) else { throw EditorRenderError.cannotCreateImage }
        let width = cropped.width
        let height = cropped.height
        guard let base = bitmap(width: width, height: height), let overlay = bitmap(width: width, height: height) else {
            throw EditorRenderError.cannotCreateImage
        }

        base.draw(cropped, in: CGRect(x: 0, y: 0, width: width, height: height))
        withTopLeftCoordinates(overlay, height: height) {
            for stroke in document.strokes where !stroke.points.isEmpty {
                let points = stroke.points.map {
                    CGPoint(
                        x: ($0.x - crop.x) / crop.width * Double(width),
                        y: ($0.y - crop.y) / crop.height * Double(height))
                }
                overlay.setBlendMode(stroke.erases ? .clear : .normal)
                overlay.setStrokeColor(CGColor(
                    red: stroke.color.red, green: stroke.color.green,
                    blue: stroke.color.blue, alpha: stroke.color.alpha))
                overlay.setLineWidth(max(1, stroke.width * Double(min(sourceWidth, sourceHeight))))
                overlay.setLineCap(.round)
                overlay.setLineJoin(.round)
                overlay.beginPath()
                overlay.move(to: points[0])
                if points.count == 1 { overlay.addLine(to: points[0]) }
                for point in points.dropFirst() { overlay.addLine(to: point) }
                overlay.strokePath()
            }
        }
        guard let overlayImage = overlay.makeImage() else { throw EditorRenderError.cannotCreateImage }
        base.draw(overlayImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let output = base.makeImage(),
              let png = NSBitmapImageRep(cgImage: output).representation(using: .png, properties: [:]) else {
            throw EditorRenderError.cannotCreateImage
        }
        return png
    }

    private static func bitmap(width: Int, height: Int) -> CGContext? {
        CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    }

    private static func withTopLeftCoordinates(_ context: CGContext, height: Int, _ body: () -> Void) {
        context.saveGState()
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1, y: -1)
        body()
        context.restoreGState()
    }
}

public struct EditorDocumentStore: Sendable {
    public var root: URL

    public init(root: URL) { self.root = root }

    public static func `default`() -> Self {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return Self(root: base.appendingPathComponent("dMLX/editor"))
    }

    public func load(sessionId: String, resultId: String) -> EditorDocument? {
        try? JSONDecoder().decode(EditorDocument.self, from: Data(contentsOf: url(sessionId, resultId)))
    }

    public func save(_ document: EditorDocument, sessionId: String) throws {
        let destination = url(sessionId, document.baseResultId)
        try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONEncoder().encode(document).write(to: destination, options: .atomic)
    }

    public func delete(sessionId: String) throws {
        let directory = root.appendingPathComponent(safe(sessionId), isDirectory: true)
        if FileManager.default.fileExists(atPath: directory.path) { try FileManager.default.removeItem(at: directory) }
    }

    private func url(_ sessionId: String, _ resultId: String) -> URL {
        root.appendingPathComponent(safe(sessionId), isDirectory: true)
            .appendingPathComponent(safe(resultId) + ".json")
    }

    private func safe(_ value: String) -> String {
        value.replacingOccurrences(of: "[^A-Za-z0-9._-]", with: "_", options: .regularExpression)
    }
}
