import AppKit
import DMLXKit
import Foundation
import SwiftUI

/// Development-only UI driver. Set DMLX_HARNESS=/path/script.json to run scripted steps and
/// write snapshots of the app's own window (no Screen Recording permission needed).
///
/// Script: {"out": "/dir", "steps": [{"newSession": true}, {"send": "a boat"}, {"waitIdle": 300},
///          {"snapshot": "name"}, {"variation": true}, {"edit": true}, {"draft": "text"}, {"sleep": 1},
///          {"onboarding": true|false}, {"settings": true}, {"quit": true}]}
@MainActor
enum DevHarness {
    /// When no SwiftUI window is on screen (e.g. the Mac is locked), render RootView in our own window.
    private static var hostWindow: NSWindow?

    private static func ensureHostWindow(_ model: AppModel) {
        guard hostWindow == nil, !NSApp.windows.contains(where: { $0.isVisible && $0.frame.width > 300 }) else { return }
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1280, height: 860),
            styleMask: [.titled, .closable, .resizable, .fullSizeContentView], backing: .buffered, defer: false)
        window.title = "dMLX (harness)"
        window.contentView = NSHostingView(rootView: RootView().environment(model).tint(Theme.accent))
        window.orderFrontRegardless()
        hostWindow = window
        log("created host window (no visible app window)")
    }

    static func startIfRequested(_ model: AppModel) {
        guard let path = ProcessInfo.processInfo.environment["DMLX_HARNESS"],
              let data = FileManager.default.contents(atPath: path),
              let script = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let steps = script["steps"] as? [[String: Any]] else { return }
        let out = URL(fileURLWithPath: script["out"] as? String ?? NSTemporaryDirectory())
        try? FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        log("script \(path), \(steps.count) steps")
        _Concurrency.Task { @MainActor in
            await waitForEngine(model)
            ensureHostWindow(model)
            log("windows: " + NSApp.windows.map { "\($0.title)|visible=\($0.isVisible)|\(Int($0.frame.width))x\(Int($0.frame.height))" }.joined(separator: ", "))
            for step in steps {
                await run(step, model: model, out: out)
            }
        }
    }

    private static func log(_ message: String) {
        FileHandle.standardError.write(Data("[harness] \(message)\n".utf8))
    }

    private static func waitForEngine(_ model: AppModel) async {
        for _ in 0..<600 {
            if model.isEngineRunning && !model.models.isEmpty { return }
            try? await _Concurrency.Task.sleep(for: .milliseconds(200))
        }
        log("engine did not become ready")
    }

    private static func run(_ step: [String: Any], model: AppModel, out: URL) async {
        log("step \(step)")
        if step["newSession"] != nil { model.newSession() }
        if let text = step["draft"] as? String { model.draft = text }
        if let text = step["send"] as? String { model.send(text) }
        if step["variation"] != nil, let t = model.latestImageTurn { model.variation(t.id) }
        if step["edit"] != nil, let t = model.latestImageTurn { model.edit(t.id) }
        if let show = step["onboarding"] as? Bool { model.showOnboarding = show }
        if step["settings"] != nil { NSApp.sendAction(Selector(("showSettingsWindow:")), to: nil, from: nil) }
        if let seconds = step["sleep"] as? Double { try? await _Concurrency.Task.sleep(for: .seconds(seconds)) }
        if let timeout = step["waitIdle"] as? Double {
            let deadline = Date().addingTimeInterval(timeout)
            while Date() < deadline {
                let busy = model.turns.contains { $0.versions.contains { $0.isActive } }
                if !busy { break }
                try? await _Concurrency.Task.sleep(for: .milliseconds(300))
            }
            try? await _Concurrency.Task.sleep(for: .milliseconds(600))
        }
        if let name = step["snapshot"] as? String { snapshot(name: name, out: out, all: step["allWindows"] != nil) }
        if step["dumpState"] != nil { dump(model, out: out) }
        if step["quit"] != nil { NSApp.terminate(nil) }
    }

    static func snapshot(name: String, out: URL, all: Bool) {
        var windows = NSApp.windows.filter { $0.isVisible && $0.contentView != nil && $0.frame.width > 100 }
        if let host = hostWindow {
            windows = [host] + NSApp.windows.filter { $0 !== host && $0.contentView != nil && $0.frame.width > 100 && ($0.isVisible || $0.isSheet) }
        }
        for (i, window) in (all ? windows : Array(windows.prefix(1))).enumerated() {
            window.contentView?.layoutSubtreeIfNeeded()
            window.displayIfNeeded()
            guard let view = window.contentView else { continue }
            let file = out.appendingPathComponent(all ? "\(name)-\(i).png" : "\(name).png")
            if let png = renderLayers(view) ?? renderCache(view) {
                try? png.write(to: file)
                log("snapshot \(file.path) \(Int(view.bounds.width))x\(Int(view.bounds.height))")
            }
        }
    }

    /// Renders the view's layer tree (captures SwiftUI's layer-backed lists and scroll views).
    private static func renderLayers(_ view: NSView) -> Data? {
        guard let layer = view.layer else { return nil }
        let scale = view.window?.backingScaleFactor ?? 2
        let size = CGSize(width: view.bounds.width * scale, height: view.bounds.height * scale)
        guard let ctx = CGContext(
            data: nil, width: Int(size.width), height: Int(size.height), bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }
        ctx.setFillColor(NSColor.windowBackgroundColor.cgColor)
        ctx.fill(CGRect(origin: .zero, size: size))
        ctx.scaleBy(x: scale, y: scale)
        if !view.isFlipped {
            // CALayer.render draws in layer coordinates; flip to match AppKit's bottom-left origin.
            ctx.translateBy(x: 0, y: view.bounds.height)
            ctx.scaleBy(x: 1, y: -1)
        }
        layer.render(in: ctx)
        guard let image = ctx.makeImage() else { return nil }
        return NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:])
    }

    private static func renderCache(_ view: NSView) -> Data? {
        guard let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) else { return nil }
        view.cacheDisplay(in: view.bounds, to: rep)
        return rep.representation(using: .png, properties: [:])
    }

    static func dump(_ model: AppModel, out: URL) {
        let lines = model.turns.map { t -> String in
            let v = t.current
            return "\(t.kind.rawValue) #\(t.number) route=\(t.route) status=\(v?.status.rawValue ?? "-") result=\(v?.resultId ?? "-") loads=\(v?.loads ?? []) strategy=\(v?.strategy ?? "-") err=\(v?.errorMessage ?? "-") text=\(t.text.prefix(60))"
        }
        try? lines.joined(separator: "\n").write(to: out.appendingPathComponent("state.txt"), atomically: true, encoding: .utf8)
    }
}
