// Draws the dMLX app icon ("Still Water") and writes app/Resources/AppIcon.icns (via iconutil)
// and docs/icon.png. The vector sources are in docs/brand/.
//   swift tools/dev/make_icon.swift
import AppKit
import Foundation

let tile = NSColor(srgbRed: 0x1d / 255, green: 0x2a / 255, blue: 0x36 / 255, alpha: 1)   // #1d2a36
let sun = NSColor(srgbRed: 0x7d / 255, green: 0xb7 / 255, blue: 0xe8 / 255, alpha: 1)    // #7db7e8
let water = NSColor(srgbRed: 0xe8 / 255, green: 0xee / 255, blue: 0xf3 / 255, alpha: 1)  // #e8eef3

/// A superellipse (n = 5), close to the macOS icon shape.
func squircle(in rect: CGRect) -> CGPath {
    let path = CGMutablePath()
    let n: CGFloat = 5
    for i in 0..<256 {
        let t = 2 * CGFloat.pi * CGFloat(i) / 256
        let c = cos(t), s = sin(t)
        let x = rect.midX + rect.width / 2 * copysign(pow(abs(c), 2 / n), c)
        let y = rect.midY + rect.height / 2 * copysign(pow(abs(s), 2 / n), s)
        i == 0 ? path.move(to: CGPoint(x: x, y: y)) : path.addLine(to: CGPoint(x: x, y: y))
    }
    path.closeSubpath()
    return path
}

/// The mark on a 64-unit grid with y pointing down (the same numbers as docs/brand/*.svg).
/// The small version (for 16 and 32 px) has two thicker lines so it stays clear.
func drawMark(_ ctx: CGContext, small: Bool) {
    func pill(_ x: CGFloat, _ y: CGFloat, _ w: CGFloat, _ h: CGFloat) {
        ctx.addPath(CGPath(roundedRect: CGRect(x: x, y: y, width: w, height: h), cornerWidth: h / 2, cornerHeight: h / 2, transform: nil))
    }
    let r: CGFloat = small ? 26 : 24
    let baseY: CGFloat = 30
    ctx.setFillColor(sun.cgColor)
    ctx.move(to: CGPoint(x: 32 - r, y: baseY))
    ctx.addArc(center: CGPoint(x: 32, y: baseY), radius: r, startAngle: .pi, endAngle: 0, clockwise: false)
    ctx.closePath()
    ctx.fillPath()

    ctx.setFillColor(water.cgColor)
    if small {
        pill(6, 36, 52, 9)
        pill(18, 50, 28, 9)
    } else {
        pill(8, 34, 48, 6)
        pill(13, 43, 17, 6)
        pill(34, 43, 17, 6)
        pill(22, 52, 20, 6)
    }
    ctx.fillPath()
}

func drawIcon(px: Int) -> NSBitmapImageRep {
    let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: px, pixelsHigh: px, bitsPerSample: 8, samplesPerPixel: 4,
                               hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
    let ctx = NSGraphicsContext.current!.cgContext
    let size = CGFloat(px), s = size / 1024

    // macOS icon grid: an 824 pt shape centered on a 1024 canvas.
    ctx.setFillColor(tile.cgColor)
    ctx.addPath(squircle(in: CGRect(x: 100 * s, y: 100 * s, width: 824 * s, height: 824 * s)))
    ctx.fillPath()

    // The mark: 64 grid at 10x, centered (grid point 32 -> 512). Flip to y-down for the grid.
    ctx.saveGState()
    ctx.translateBy(x: 0, y: size)
    ctx.scaleBy(x: 1, y: -1)
    ctx.translateBy(x: 192 * s, y: 192 * s)
    ctx.scaleBy(x: 10 * s, y: 10 * s)
    drawMark(ctx, small: px <= 64)
    ctx.restoreGState()

    NSGraphicsContext.restoreGraphicsState()
    return rep
}

let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let iconset = FileManager.default.temporaryDirectory.appendingPathComponent("AppIcon.iconset")
try? FileManager.default.removeItem(at: iconset)
try FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)
for base in [16, 32, 128, 256, 512] {
    for scale in [1, 2] {
        let name = scale == 1 ? "icon_\(base)x\(base).png" : "icon_\(base)x\(base)@2x.png"
        try drawIcon(px: base * scale).representation(using: .png, properties: [:])!.write(to: iconset.appendingPathComponent(name))
    }
}
let out = root.appendingPathComponent("app/Resources/AppIcon.icns")
let task = Process()
task.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
task.arguments = ["-c", "icns", iconset.path, "-o", out.path]
try task.run()
task.waitUntilExit()
try drawIcon(px: 1024).representation(using: .png, properties: [:])!.write(to: root.appendingPathComponent("docs/icon.png"))
print("wrote \(out.path)")
