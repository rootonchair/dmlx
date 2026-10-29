import AppKit
import DMLXKit
import Foundation
import SwiftUI

/// Development-only UI driver. Set DMLX_HARNESS=/path/script.json to run scripted steps and
/// write snapshots of the app's own window (no Screen Recording permission needed).
///
/// Script: {"out": "/dir", "steps": [{"newSession": true}, {"send": "a boat"}, {"waitIdle": 300},
///          {"snapshot": "name"}, {"variation": true}, {"edit": true}, {"draft": "text"}, {"sleep": 1},
///          {"onboarding": true|false}, {"settings": true}, {"selectSession": 0}, {"renderChat": "name"},
///          {"modelChoice": "auto"|"<model id>"}, {"logPreview": true},
///          {"windowSize": [w, h]}, {"measure": true}, {"fakeSessions": n},
///          {"ask": {"message": "…", "question": "…"}} (as if the agent asked), {"waitRouted": 60},
///          {"flags": "label"} (logs whether this or another chat is making an image, and a waiting message),
///          {"quit": true}]}
@MainActor
enum DevHarness {
    /// When no SwiftUI window is on screen (e.g. the Mac is locked), render RootView in our own window.
    private static var hostWindow: NSWindow?
    private static var addModelWindow: NSWindow?

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
            if model.isEngineRunning && !model.models.isEmpty {
                try? await _Concurrency.Task.sleep(for: .seconds(1))  // let onboarding logic run
                return
            }
            try? await _Concurrency.Task.sleep(for: .milliseconds(200))
        }
        log("engine did not become ready")
    }

    private static func run(_ step: [String: Any], model: AppModel, out: URL) async {
        log("step \(step)")
        if step["newSession"] != nil { model.newSession() }
        if let index = step["selectSession"] as? Int, model.sessions.indices.contains(index) {
            model.select(model.sessions[index].id)
        }
        if let name = step["renderChat"] as? String { renderChat(model, name: name, out: out) }
        if let text = step["draft"] as? String { model.draft = text }
        if let choice = step["modelChoice"] as? String { model.modelChoice = choice }
        if step["suggestions"] != nil {
            do { log("suggestions: \(try await model.catalogSuggestions().count)") } catch { log("suggestions error: \(error.localizedDescription)") }
        }
        if let repo = step["inspect"] as? String {
            do {
                let i = try await model.inspectModel(repo)
                log("inspect \(repo): \(i.name) | \(i.tasks) | \(i.license) permissive=\(i.permissive_license) | \(i.size_gb) GB | added=\(i.already_added ?? "-")")
            } catch {
                log("inspect \(repo): error \(error.localizedDescription)")
            }
        }
        if step["addModelSheet"] != nil {
            let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 560, height: 600), styleMask: [.titled], backing: .buffered, defer: false)
            window.contentView = NSHostingView(rootView: AddModelView().environment(model))
            window.orderFrontRegardless()
            addModelWindow = window
        }
        if step["logPreview"] != nil { log("preview [\(model.modelChoice)] \(model.draft.isEmpty ? "(empty)" : model.draft): \(model.routePreview)") }
        if let ask = step["ask"] as? [String: String], let message = ask["message"], let question = ask["question"] {
            model.harnessAsk(message: message, question: question)
        }
        if let timeout = step["waitRouted"] as? Double {
            let deadline = Date().addingTimeInterval(timeout)
            try? await _Concurrency.Task.sleep(for: .milliseconds(300))
            while Date() < deadline && model.isResolvingRoute {
                try? await _Concurrency.Task.sleep(for: .milliseconds(300))
            }
            log("routed: pendingAsk=\(model.pendingAsk ?? "-") draft=\(model.draft)")
        }
        if let text = step["send"] as? String { model.send(text) }
        if step["variation"] != nil, let t = model.latestImageTurn { model.variation(t.id) }
        if step["edit"] != nil, let t = model.latestImageTurn { model.attachResult(t.id) }
        if let show = step["onboarding"] as? Bool { model.showOnboarding = show }
        if step["settings"] != nil { NSApp.sendAction(Selector(("showSettingsWindow:")), to: nil, from: nil) }
        if let seconds = step["sleep"] as? Double { try? await _Concurrency.Task.sleep(for: .seconds(seconds)) }
        if let timeout = step["waitIdle"] as? Double {
            let deadline = Date().addingTimeInterval(timeout)
            try? await _Concurrency.Task.sleep(for: .milliseconds(500))  // a send plans before its turn exists
            while Date() < deadline {
                let busy = model.turns.contains { $0.versions.contains { $0.isActive } }
                    || model.isResolvingRoute || !model.activeRuns.isEmpty
                if !busy { break }
                try? await _Concurrency.Task.sleep(for: .milliseconds(300))
            }
            try? await _Concurrency.Task.sleep(for: .milliseconds(600))
        }
        if let name = step["snapshot"] as? String { snapshot(name: name, out: out, all: step["allWindows"] != nil) }
        if step["dumpState"] != nil { dump(model, out: out) }
        if let label = step["flags"] as? String {
            let last = model.turns.last.map { "\($0.kind.rawValue)#\($0.number) \($0.current?.status.rawValue ?? "-") planner=\($0.current?.workflowPrompt?.compiler ?? "-")" } ?? "-"
            log("flags[\(label)] chat=\(model.currentSession?.title ?? "new") generatingHere=\(model.isGeneratingHere) "
                + "elsewhere=\(model.generatingElsewhere?.title ?? "-") waiting=\(model.isWaitingForImage) "
                + "resolving=\(model.isResolvingRoute) turns=\(model.turns.count) last=\(last) draft=\(model.draft)")
        }
        // In-memory only (never written to the store): fill or empty the sidebar list.
        if let count = step["fakeSessions"] as? Int {
            model.sessions = (0..<count).map { Session(title: "Harness chat \($0 + 1)") }
        }
        if let size = step["windowSize"] as? [Double], size.count == 2,
           let window = hostWindow ?? NSApp.windows.first(where: { $0.isVisible && $0.frame.width > 300 }) {
            window.setFrame(NSRect(origin: window.frame.origin, size: CGSize(width: size[0], height: size[1])), display: true)
        }
        if step["measure"] != nil, let window = hostWindow ?? NSApp.windows.first(where: { $0.isVisible && $0.frame.width > 300 }),
           let content = window.contentView {
            content.layoutSubtreeIfNeeded()
            let root = content.subviews.first
            log("measure window=\(window.frame.size) fitting=\(root?.fittingSize ?? .zero) rootFrame=\(root?.frame ?? .zero)")
            func find(_ view: NSView, _ match: (NSView) -> Bool) -> NSView? {
                if match(view) { return view }
                for sub in view.subviews { if let hit = find(sub, match) { return hit } }
                return nil
            }
            if let split = find(content, { $0 is NSSplitView }) as? NSSplitView {
                for (i, column) in split.arrangedSubviews.enumerated() {
                    let host = find(column, { String(describing: type(of: $0)).contains("NSHostingView") })
                    log("measure column\(i) frame=\(column.frame.size) host=\(host?.frame.size ?? .zero) hostFitting=\(host?.fittingSize ?? .zero)")
                }
            }
        }
        if step["killEngine"] != nil {
            let kill = Process()
            kill.executableURL = URL(fileURLWithPath: "/usr/bin/pkill")
            kill.arguments = ["-9", "-f", "dmlx(\\.cli)? serve"]
            try? kill.run()
            kill.waitUntilExit()
            log("killed engine (exit \(kill.terminationStatus))")
        }
        if let timeout = step["waitEngine"] as? Double {
            let deadline = Date().addingTimeInterval(timeout)
            try? await _Concurrency.Task.sleep(for: .seconds(1.5))
            while Date() < deadline && !model.isEngineRunning {
                try? await _Concurrency.Task.sleep(for: .milliseconds(300))
            }
            log("engine running: \(model.isEngineRunning)")
        }
        if let id = step["download"] as? String, let m = model.models.first(where: { $0.id == id }) {
            log("onboarding visible before download: \(model.showOnboarding)")
            model.download(m)  // same action as the onboarding Download button
        }
        if let id = step["waitDownloaded"] as? String {
            let deadline = Date().addingTimeInterval(step["timeout"] as? Double ?? 3600)
            var lastLog = Date.distantPast
            while Date() < deadline {
                await model.refreshModels()
                guard let m = model.models.first(where: { $0.id == id }) else { break }
                if m.downloaded { log("downloaded \(id)"); break }
                if m.download.status == "error" { log("download error: \(m.download.error ?? "?")"); break }
                if Date().timeIntervalSince(lastLog) > 30 {
                    log(String(format: "download %@ %.1f%% (%.2f GB)", id, m.download.progress * 100, Double(m.download.done_bytes) / 1e9))
                    lastLog = Date()
                }
                try? await _Concurrency.Task.sleep(for: .seconds(2))
            }
        }
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
            // A real on-screen window draws correctly through AppKit; the off-screen host window needs layers.
            let png = window === hostWindow ? renderLayers(view) ?? renderCache(view) : renderCache(view)
            if let png {
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

    /// Draws the chat turns and status card with ImageRenderer. AppKit-backed parts (the sidebar list,
    /// scroll view, text field) do not appear in window snapshots, so this is how the chat itself is reviewed.
    static func renderChat(_ model: AppModel, name: String, out: URL) {
        let content = HStack(alignment: .top, spacing: 24) {
            VStack(alignment: .leading, spacing: 8) {
                ForEach(model.sessions.prefix(8)) { s in
                    Text(s.title).lineLimit(1).fontWeight(s.id == model.currentSessionId ? .semibold : .regular)
                }
                Spacer(minLength: 24)
                StatusCard()
            }
            .frame(width: 220)
            .padding(12)
            VStack(alignment: .leading, spacing: 22) {
                if model.turns.isEmpty { EmptyStateView(starters: ["A fox logo, flat vector, two colors"]) }
                ForEach(model.turns) { TurnView(turn: $0) }
            }
            .frame(width: 780)
        }
        .padding(24)
        .background(Color(nsColor: .windowBackgroundColor))
        .environment(model)
        .tint(Theme.accent)
        .environment(\.colorScheme, NSApp.effectiveAppearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? .dark : .light)
        let renderer = ImageRenderer(content: content)
        renderer.scale = 2
        guard let image = renderer.cgImage,
              let png = NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:]) else {
            log("renderChat failed"); return
        }
        let file = out.appendingPathComponent("\(name).png")
        try? png.write(to: file)
        log("renderChat \(file.path) \(image.width)x\(image.height)")
    }

    private static func renderCache(_ view: NSView) -> Data? {
        guard let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) else { return nil }
        view.cacheDisplay(in: view.bounds, to: rep)
        return rep.representation(using: .png, properties: [:])
    }

    static func dump(_ model: AppModel, out: URL) {
        let lines = model.turns.map { t -> String in
            let v = t.current
            return "\(t.kind.rawValue) #\(t.number) route=\(t.route) status=\(v?.status.rawValue ?? "-") result=\(v?.resultId ?? "-") loads=\(v?.loads ?? []) strategy=\(v?.strategy ?? "-") err=\(v?.errorMessage ?? "-") inputs=\(v?.mediaInputs?.map(\.id) ?? []) text=\(t.text.prefix(60))"
        }
        let header = "onboarding=\(model.showOnboarding) engine=\(model.engineStatusText) models=\(model.models.map { "\($0.id):\($0.downloaded)" })"
        try? ([header] + lines).joined(separator: "\n").write(to: out.appendingPathComponent("state.txt"), atomically: true, encoding: .utf8)
    }
}
