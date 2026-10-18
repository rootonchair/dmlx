// Draws the dMLX app icon and writes app/Resources/AppIcon.icns (via iconutil).
//   swift tools/dev/make_icon.swift
import AppKit
import Foundation

func drawIcon(size: CGFloat) -> NSImage {
    let image = NSImage(size: NSSize(width: size, height: size))
    image.lockFocus()
    guard let ctx = NSGraphicsContext.current?.cgContext else { fatalError() }
    let s = size / 1024

    // macOS icon grid: 824pt squircle centered in 1024 canvas.
    let inset = 100 * s
    let rect = CGRect(x: inset, y: inset, width: size - 2 * inset, height: size - 2 * inset)
    let shape = NSBezierPath(roundedRect: rect, xRadius: 185 * s, yRadius: 185 * s)
    ctx.saveGState()
    shape.addClip()

    // Sky bands (flat, warm dusk).
    let bands: [(CGFloat, NSColor)] = [
        (0.00, NSColor(srgbRed: 0.13, green: 0.12, blue: 0.13, alpha: 1)),
        (0.42, NSColor(srgbRed: 0.20, green: 0.16, blue: 0.15, alpha: 1)),
        (0.58, NSColor(srgbRed: 0.33, green: 0.22, blue: 0.17, alpha: 1)),
    ]
    for (i, band) in bands.enumerated() {
        let top = rect.maxY - rect.height * band.0
        let bottom = i + 1 < bands.count ? rect.maxY - rect.height * bands[i + 1].0 : rect.minY
        band.1.setFill()
        CGRect(x: rect.minX, y: bottom, width: rect.width, height: top - bottom).fill()
    }

    // Sun.
    NSColor(srgbRed: 0.93, green: 0.65, blue: 0.24, alpha: 1).setFill()
    NSBezierPath(ovalIn: CGRect(x: rect.midX - 40 * s, y: rect.minY + rect.height * 0.40, width: 260 * s, height: 260 * s)).fill()

    // Hills.
    func hill(_ points: [(CGFloat, CGFloat)], _ color: NSColor) {
        let p = NSBezierPath()
        p.move(to: CGPoint(x: rect.minX, y: rect.minY))
        for (x, y) in points { p.line(to: CGPoint(x: rect.minX + rect.width * x, y: rect.minY + rect.height * y)) }
        p.line(to: CGPoint(x: rect.maxX, y: rect.minY))
        p.close()
        color.setFill()
        p.fill()
    }
    hill([(0, 0.42), (0.22, 0.56), (0.45, 0.40), (0.70, 0.52), (1, 0.36)], NSColor(srgbRed: 0.42, green: 0.27, blue: 0.20, alpha: 1))
    hill([(0, 0.28), (0.30, 0.38), (0.58, 0.24), (0.82, 0.32), (1, 0.22)], NSColor(srgbRed: 0.17, green: 0.13, blue: 0.13, alpha: 1))

    // Chat bubble tail hint: a small cream speech mark, bottom-left.
    let bubble = NSBezierPath(roundedRect: CGRect(x: rect.minX + 90 * s, y: rect.minY + 90 * s, width: 230 * s, height: 140 * s),
                              xRadius: 60 * s, yRadius: 60 * s)
    NSColor(srgbRed: 0.97, green: 0.94, blue: 0.88, alpha: 1).setFill()
    bubble.fill()
    let tail = NSBezierPath()
    tail.move(to: CGPoint(x: rect.minX + 120 * s, y: rect.minY + 100 * s))
    tail.line(to: CGPoint(x: rect.minX + 95 * s, y: rect.minY + 55 * s))
    tail.line(to: CGPoint(x: rect.minX + 175 * s, y: rect.minY + 95 * s))
    tail.close()
    tail.fill()
    NSColor(srgbRed: 0.17, green: 0.13, blue: 0.13, alpha: 1).setFill()
    for i in 0..<3 {
        NSBezierPath(ovalIn: CGRect(x: rect.minX + (140 + CGFloat(i) * 55) * s, y: rect.minY + 140 * s, width: 30 * s, height: 30 * s)).fill()
    }
    ctx.restoreGState()
    image.unlockFocus()
    return image
}

let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let iconset = FileManager.default.temporaryDirectory.appendingPathComponent("AppIcon.iconset")
try? FileManager.default.removeItem(at: iconset)
try FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)
for base in [16, 32, 128, 256, 512] {
    for scale in [1, 2] {
        let px = CGFloat(base * scale)
        let img = drawIcon(size: px)
        let rep = NSBitmapImageRep(data: img.tiffRepresentation!)!
        let name = scale == 1 ? "icon_\(base)x\(base).png" : "icon_\(base)x\(base)@2x.png"
        try rep.representation(using: .png, properties: [:])!.write(to: iconset.appendingPathComponent(name))
    }
}
let out = root.appendingPathComponent("app/Resources/AppIcon.icns")
let task = Process()
task.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
task.arguments = ["-c", "icns", iconset.path, "-o", out.path]
try task.run()
task.waitUntilExit()
let preview = NSBitmapImageRep(data: drawIcon(size: 512).tiffRepresentation!)!
try preview.representation(using: .png, properties: [:])!.write(to: root.appendingPathComponent("docs/icon.png"))
print("wrote \(out.path)")
