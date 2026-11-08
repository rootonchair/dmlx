import AppKit
import DMLXKit
import Foundation
import SwiftUI

/// Development-only UI driver. Set DMLX_HARNESS=/path/script.json to run scripted steps and
/// write snapshots of the app's own window (no Screen Recording permission needed).
///
/// Script: {"out": "/dir", "steps": [{"newSession": true}, {"send": "a boat"}, {"waitIdle": 300},
///          {"snapshot": "name"}, {"variation": true}, {"attachFile": "/path.png"}, {"draft": "text"}, {"sleep": 1},
///          {"onboarding": true|false}, {"onboardingStep": "welcome"|"setup"}, {"onboardingDownload": true} (presses Download or Start Creating), {"agentService": {"service": "ollama", "model": "…"}}, {"guideNext": "chat"|"image"|"editor"} (the tip's Next; DMLX_RESET_TIPS=1 shows the guide again), {"settings": true}, {"selectSession": 0}, {"renderChat": "name"} (add "chatWidth": 384 to draw the turns at that width, e.g. the
///          detail column of the smallest window),
///          {"modelChoice": "auto"|"<model id>"} (both tasks), {"generateModel": "…"}, {"editModel": "…"}, {"logPreview": true},
///          {"windowSize": [w, h]}, {"measure": true}, {"fakeSessions": n},
///          {"ask": {"message": "…", "question": "…"}} (as if the agent asked), {"planning": "message"} (as if just sent; "" clears), {"waitRouted": 60},
///          {"flags": "label"} (logs whether this or another chat is making an image, and a waiting message),
///          {"appearance": "system"|"light"|"dark"}, {"settingsTab": "intelligence"} (with {"settings": true}, which hosts SettingsView in its own window), {"closeSettings": true},
///          {"addModel": ""|"<group>"} (Add Model over Settings → Models), {"openModelMenu": "name", "keys": [125]},
///          Layout (task 17): {"layoutBoxes": [<LayoutBox json>]} (the composer's chip), {"openLayout": true} (▭ Layout,
///          a blank canvas), {"layoutTool": true} (with openEditor: open on Layout),
///          {"editorTool": "Crop"|"Paint"|"Erase"|"Layout"} (with openEditor: open on that tool), {"selectBox": "<label>"|"*"} (in the
///          open editor's Layout tool: select that box, "*" the first, as a click does), {"sheetSize": [w, h]}, {"pickVersion": 0}, {"moveBoxes": {"<label>": [y0, x0, y1, x1]}}, {"applyLayout": {"move": {"<label>":
///          [y0, x0, y1, x1]}, "describe": {"<label>": "text"}}} (what Redraw with layout sends), {"logLayout": true},
///          {"capture": "name", "window": "main"|"settings"|"settingsSheet"} (the real window through screencapture, native
///          controls included; needs Screen Recording for the launching terminal),
///          {"scroll": "top"|"bottom", "overscroll": 60} (the chat transcript, to capture a chat taller than the
///          window in parts),
///          {"fakeTurnState": {"status": "error"|"rendering"|"cancelled", "code": "insufficient_memory", "message": "…",
///          "phase": "Step 3 of 4", "step": 3, "total": 4, "versions": 5}} (in memory only, never saved: the latest
///          image turn's current version in that state; "versions": n repeats it as v1…vn; an error drops the image),
///          {"quit": true}]}
extension Notification.Name {
    /// The harness's selectBox: the open editor selects the box with this label (the object).
    static let dmlxHarnessSelectBox = Notification.Name("dmlx.harness.selectBox")
}

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
        if let name = step["renderChat"] as? String {
            renderChat(model, name: name, out: out, width: step["chatWidth"] as? Double ?? 780)
        }
        if let text = step["draft"] as? String { model.draft = text }
        if let choice = step["modelChoice"] as? String { model.setModelChoice(choice) }
        if let choice = step["generateModel"] as? String { model.setModelChoice(choice, for: [.t2i]) }
        if let choice = step["editModel"] as? String { model.setModelChoice(choice, for: [.edit]) }
        if let name = step["aspect"] as? String, let aspect = Aspect(rawValue: name) { model.aspect = aspect }
        // The "Try anyway" button on the latest image turn (a job refused for memory).
        if step["tryAnyway"] != nil, let t = model.latestImageTurn, t.current?.errorMessage != nil {
            log("try anyway after: \(t.current?.errorMessage ?? "")")
            model.retry(t.id, force: true)
        }
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
        if step["logPreview"] != nil { log("preview [\(model.generateChoice)|\(model.editChoice)] \(model.draft.isEmpty ? "(empty)" : model.draft): \(model.routePreview)") }
        if let text = step["planning"] as? String { model.harnessPlanning(text.isEmpty ? nil : text) }
        if let ask = step["ask"] as? [String: String], let message = ask["message"], let question = ask["question"] {
            model.harnessAsk(message: message, question: question)
        }
        if let timeout = step["waitRouted"] as? Double {
            let deadline = Date().addingTimeInterval(timeout)
            try? await _Concurrency.Task.sleep(for: .milliseconds(300))
            while Date() < deadline && model.planningMessage != nil {
                try? await _Concurrency.Task.sleep(for: .milliseconds(300))
            }
            log("routed: pendingAsk=\(model.pendingAsk ?? "-") draft=\(model.draft)")
        }
        if let text = step["send"] as? String {
            model.send(text)
            if model.planningMessage != text && model.turns.last?.text != text, let toast = model.toast {
                log("send not taken: \(toast)")
            }
        }
        if step["variation"] != nil, let t = model.latestImageTurn { model.variation(t.id) }
        if let spec = step["fakeTurnState"] as? [String: Any], let id = model.latestImageTurn?.id,
           let t = model.turns.firstIndex(where: { $0.id == id }) {
            var turn = model.turns[t]
            if let n = spec["versions"] as? Int, n > 1, let v = turn.current {
                turn.versions = (0..<n).map { _ in var c = v; c.id = UUID().uuidString; return c }
                turn.activeVersion = n - 1
            }
            var v = turn.versions[turn.activeVersion]
            if let raw = spec["status"] as? String, let status = VersionStatus(rawValue: raw) {
                v.status = status
                if status == .error || status == .cancelled { v.resultId = nil }
            }
            v.errorCode = spec["code"] as? String ?? v.errorCode
            v.errorMessage = spec["message"] as? String ?? v.errorMessage
            v.phase = spec["phase"] as? String ?? v.phase
            v.step = spec["step"] as? Int ?? v.step
            v.totalSteps = spec["total"] as? Int ?? v.totalSteps
            turn.versions[turn.activeVersion] = v
            model.turns[t] = turn
        }
        if let path = step["attachFile"] as? String { model.attach(files: [URL(fileURLWithPath: path)]) }
        if let name = step["onboardingStep"] as? String, let value = OnboardingStep(rawValue: name) { model.onboardingStep = value }
        if let show = step["onboarding"] as? Bool { model.showOnboarding = show }
        if step["onboardingDownload"] != nil { model.harnessOnboardingDownload += 1 }
        if let agent = step["agentService"] as? [String: String], let service = AgentService(rawValue: agent["service"] ?? "") {
            model.harnessAgent = AgentChoice(service: service, model: agent["model"] ?? "", address: agent["address"] ?? "")
        }
        if step["guideLog"] != nil {
            log("guide ready=\(GuideState.ready) hasImage=\(GuideState.hasImage) chat=\(Guide.shared.chat.currentTip?.id ?? "-") image=\(Guide.shared.image.currentTip?.id ?? "-") t2i=\(model.defaultModel(for: .t2i)?.id ?? "-") onboarding=\(model.showOnboarding)")
        }
        if let group = step["guideNext"] as? String {
            // Press Next on the current tip of the "chat" or "image" group.
            if group == "editor" { Guide.shared.next(Guide.shared.editor) }
            else if let tip = (group == "image" ? Guide.shared.image : Guide.shared.chat).currentTip { Guide.shared.next(tip) }
        }
        // {"settings": true}, or {"settings": {"height": 760}} for a taller window (README screenshots).
        if let value = step["settings"] {
            showSettings(model, height: ((value as? [String: Any])?["height"] as? Double) ?? 460)
        }
        if let tab = step["settingsTab"] as? String { model.settingsTab = tab }
        if let name = step["appearance"] as? String, let a = AppAppearance(rawValue: name) {
            UserDefaults.standard.set(a.rawValue, forKey: AppAppearance.key)
            a.apply()
        }
        if step["closeSettings"] != nil { settingsWindow()?.close() }
        // {"overrideRoute": "image.generate"|"image.edit"}: as if picked in the composer's workflow menu.
        if let operation = step["overrideRoute"] as? String { model.overrideRoute(operation) }
        // With Settings → Models open: Add Model, at the top ("") or at one model's versions ("flux-2-klein-9b").
        if let group = step["addModel"] as? String { model.harnessAddModel = group }
        // {"openModelMenu": "name", "keys": [125, 124]}: opens the composer's model menu, presses the keys (key
        // codes: 125 down, 124 right) while it is open, and captures the screen around the main window.
        // Logs the composer's model menu as AppKit built it: titles, check marks and submenus.
        if step["dumpModelMenu"] != nil, let content = mainWindow()?.contentView {
            func find(_ view: NSView) -> NSPopUpButton? {
                // The aspect menu is the other one in the composer ("Square · 1024²").
                if let popup = view as? NSPopUpButton, !(popup.accessibilityLabel() ?? "").contains("²") { return popup }
                for sub in view.subviews { if let hit = find(sub) { return hit } }
                return nil
            }
            if let popup = find(content), let menu = popup.menu {
                func lines(_ menu: NSMenu, _ indent: String) -> [String] {
                    menu.items.flatMap { item -> [String] in
                        if let sub = item.submenu { sub.delegate?.menuNeedsUpdate?(sub); sub.delegate?.menuWillOpen?(sub); sub.update() }
                        let mark = item.state == .on ? "✓ " : "  "
                        let line = item.isSeparatorItem ? "\(indent)──" : "\(indent)\(mark)\(item.title)\(item.badge.map { "  [\($0.stringValue)]" } ?? "")\(item.isEnabled ? "" : " (disabled)")"
                        return [line] + (item.submenu.map { lines($0, indent + "    ") } ?? [])
                    }
                }
                // SwiftUI builds the items when the menu opens: open it, read it while it tracks, close it.
                let timer = Timer(timeInterval: 0.6, repeats: false) { _ in
                    log("model menu \"\(popup.accessibilityLabel() ?? popup.title)\":\n" + lines(menu, "  ").joined(separator: "\n"))
                    menu.cancelTracking()
                }
                RunLoop.main.add(timer, forMode: .common)
                popup.performClick(nil)
            } else {
                var found: [String] = []
                func walk(_ v: NSView) {
                    if String(describing: type(of: v)).contains("Popup") {
                        var chain: [String] = []
                        var c: AnyClass? = type(of: v)
                        while let k = c { chain.append(String(describing: k)); c = class_getSuperclass(k) }
                        found.append("\(chain.prefix(4).joined(separator: "<")) id=\(v.accessibilityIdentifier()) label=\(v.accessibilityLabel() ?? "-") items=\(v.menu?.items.count ?? -1) popupItems=\((v as? NSPopUpButton)?.numberOfItems ?? -1)")
                    }
                    v.subviews.forEach(walk)
                }
                walk(content)
                log("dumpModelMenu: menu not found; \(found.joined(separator: " | "))")
            }
        }
        // Draws one window in-process (works while the screen is locked, unlike capture).
        if let name = step["snapshotWindow"] as? String {
            let window: NSWindow? = switch step["window"] as? String {
            case "settings": settingsWindow()
            case "settingsSheet": settingsWindow()?.attachedSheet ?? NSApp.windows.first { $0.isSheet && $0.isVisible }
            case "sheet": mainWindow()?.attachedSheet  // the Crop / Draw editor
            default: mainWindow()
            }
            if let view = window?.contentView, let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) {
                // cacheDisplay leaves out the window's own background (vibrancy): draw it under the content.
                view.cacheDisplay(in: view.bounds, to: rep)
                let image = NSImage(size: view.bounds.size)
                image.lockFocus()
                view.effectiveAppearance.performAsCurrentDrawingAppearance {
                    NSColor.windowBackgroundColor.setFill()
                    NSRect(origin: .zero, size: view.bounds.size).fill()
                }
                rep.draw(in: NSRect(origin: .zero, size: view.bounds.size), from: .zero, operation: .sourceOver,
                         fraction: 1, respectFlipped: true, hints: nil)
                image.unlockFocus()
                guard let tiff = image.tiffRepresentation, let png = NSBitmapImageRep(data: tiff)?
                    .representation(using: .png, properties: [:]) else { return }
                try? png.write(to: out.appendingPathComponent("\(name).png"))
                log("snapshotWindow \(name) \(Int(view.bounds.width))x\(Int(view.bounds.height))")
            } else {
                log("snapshotWindow \(name): no window")
            }
        }
        // {"menuFrames": "name"}: opens the composer's model menu and captures only this app's windows (the menu
        // included) every 0.1 s for 1.2 s; no keys are sent, so other apps are untouched.
        // {"modelPicker": "edit"|"t2i"|"close"}: opens or closes the composer's model picker panel.
        if let value = step["modelPicker"] as? String { model.showModelPicker = value != "close" }
        // {"snapshotPopover": "name"}: draws the open popover's window in-process (works with the display asleep).
        if let name = step["snapshotPopover"] as? String {
            let popovers = NSApp.windows.filter { String(describing: type(of: $0)).contains("Popover") && $0.isVisible }
            log("snapshotPopover: \(NSApp.windows.map { String(describing: type(of: $0)) })")
            if let view = popovers.first?.contentView, let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) {
                view.cacheDisplay(in: view.bounds, to: rep)
                let image = NSImage(size: view.bounds.size)
                image.lockFocus()
                // The popover's material is not drawn: fill with a plain color of its appearance.
                let dark = view.effectiveAppearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
                (dark ? NSColor(white: 0.17, alpha: 1) : NSColor(white: 0.96, alpha: 1)).setFill()
                NSRect(origin: .zero, size: view.bounds.size).fill()
                rep.draw(in: NSRect(origin: .zero, size: view.bounds.size))
                image.unlockFocus()
                if let tiff = image.tiffRepresentation, let png = NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:]) {
                    try? png.write(to: out.appendingPathComponent("\(name).png"))
                    log("snapshotPopover \(name) \(Int(view.bounds.width))x\(Int(view.bounds.height))")
                }
            }
        }
        // {"ownWindows": "name"}: captures each on-screen window of this app (popovers and menus included).
        if let name = step["ownWindows"] as? String {
            let pid = ProcessInfo.processInfo.processIdentifier
            let infos = (CGWindowListCopyWindowInfo([.optionOnScreenOnly], kCGNullWindowID) as? [[String: Any]]) ?? []
            for (j, info) in infos.filter({ ($0[kCGWindowOwnerPID as String] as? Int32) == pid }).enumerated() {
                guard let wid = info[kCGWindowNumber as String] as? Int else { continue }
                let shot = Process()
                shot.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
                shot.arguments = ["-x", "-o", "-l\(wid)", out.appendingPathComponent("\(name)-\(j).png").path]
                try? shot.run()
                shot.waitUntilExit()
                log("ownWindows \(name)-\(j) \(info[kCGWindowBounds as String] ?? "")")
            }
        }
        if let name = step["menuFrames"] as? String, let content = mainWindow()?.contentView {
            let aspect = step["aspect"] != nil  // the size menu instead, to compare
            func find(_ view: NSView) -> NSPopUpButton? {
                if let popup = view as? NSPopUpButton, (popup.accessibilityLabel() ?? "").contains("²") == aspect { return popup }
                for sub in view.subviews { if let hit = find(sub) { return hit } }
                return nil
            }
            if let popup = find(content) {
                NSApp.activate(ignoringOtherApps: true)
                mainWindow()?.makeKeyAndOrderFront(nil)
                let pid = ProcessInfo.processInfo.processIdentifier
                for i in 0..<12 {
                    let timer = Timer(timeInterval: 0.05 + Double(i) * 0.1, repeats: false) { _ in
                        let infos = (CGWindowListCopyWindowInfo([.optionOnScreenOnly], kCGNullWindowID) as? [[String: Any]]) ?? []
                        if let w = popup.window { log("menuFrames \(i) button at \(w.convertToScreen(popup.convert(popup.bounds, to: nil))) window \(w.frame)") }
                        let mine = infos.filter { ($0[kCGWindowOwnerPID as String] as? Int32) == pid && ($0[kCGWindowLayer as String] as? Int ?? 0) > 0 }
                        for (j, info) in mine.enumerated() {
                            guard let wid = info[kCGWindowNumber as String] as? Int else { continue }
                            let bounds = info[kCGWindowBounds as String] as? [String: Any] ?? [:]
                            log("menuFrames \(i) window \(wid) layer \(info[kCGWindowLayer as String] ?? "-") bounds \(bounds["X"] ?? 0),\(bounds["Y"] ?? 0) \(bounds["Width"] ?? 0)x\(bounds["Height"] ?? 0)")
                            let shot = Process()
                            shot.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
                            shot.arguments = ["-x", "-o", "-l\(wid)", out.appendingPathComponent("\(name)-\(i)-\(j).png").path]
                            try? shot.run()
                            shot.waitUntilExit()
                        }
                    }
                    RunLoop.main.add(timer, forMode: .common)
                }
                let close = Timer(timeInterval: 1.4, repeats: false) { _ in popup.menu?.cancelTracking() }
                RunLoop.main.add(close, forMode: .common)
                popup.performClick(nil)
            } else { log("menuFrames: no model menu") }
        }
        if let name = step["openModelMenu"] as? String {
            openMenu(labeled: "Models", in: mainWindow(), keys: step["keys"] as? [Int] ?? [], name: name, out: out)
        }
        if let raw = step["layoutBoxes"], let data = try? JSONSerialization.data(withJSONObject: raw),
           let boxes = try? JSONDecoder().decode([LayoutBox].self, from: data) {
            model.pendingLayout = boxes
        }
        if step["openLayout"] != nil { model.openBlankLayout() }
        if step["layoutTool"] != nil { model.editorStartTool = "Layout" }
        if let tool = step["editorTool"] as? String { model.editorStartTool = tool }
        if let label = step["selectBox"] as? String {
            NotificationCenter.default.post(name: .dmlxHarnessSelectBox, object: label)
        }
        // {"pickVersion": n}: shows version n (0-based) of the latest image, as its v1 / v2 buttons do.
        if let n = step["pickVersion"] as? Int, let t = model.latestImageTurn { model.pickVersion(t.id, n) }
        if step["openEditor"] != nil, let t = model.latestImageTurn { model.openEditor(t.id) }
        // {"moveBoxes": {"<label>": [y0, x0, y1, x1]}} (with openEditor): the boxes as if dragged, before the sheet draws.
        if let moves = step["moveBoxes"] as? [String: [Int]], var boxes = model.editorLayout {
            for i in boxes.indices { if let box = moves[boxes[i].label] { boxes[i].box = box; boxes[i].byUser = true } }
            model.editorLayout = boxes
        }
        if let spec = step["applyLayout"] as? [String: Any], var boxes = model.editorLayout {
            for (label, box) in spec["move"] as? [String: [Int]] ?? [:] {
                if let i = boxes.firstIndex(where: { $0.label == label }), box.count == 4 {
                    boxes[i].place(NormalizedRect(x: Double(box[1]) / 1000, y: Double(box[0]) / 1000,
                                                  width: Double(box[3] - box[1]) / 1000, height: Double(box[2] - box[0]) / 1000))
                }
            }
            for (label, text) in spec["describe"] as? [String: String] ?? [:] {
                if let i = boxes.firstIndex(where: { $0.label == label }) {
                    boxes[i].desc = text; boxes[i].descByUser = true; boxes[i].byUser = true
                }
            }
            model.applyLayout(LayoutBox.sendable(boxes))
        }
        if step["logLayout"] != nil {
            let v = model.latestImageTurn?.current
            log("layout v\((model.latestImageTurn?.activeVersion ?? 0) + 1) note=\(v?.note ?? "-") sent=\(v?.layoutRequest?.map { "\($0.label)\($0.box)" } ?? []) "
                + "saved=\(v?.layout?.map { "\($0.label)\($0.box)\($0.byUser ? "*" : "")" } ?? []) pending=\(model.pendingLayout?.count ?? 0)")
        }
        if step["closeEditor"] != nil { model.closeEditor() }
        // {"sheetSize": [w, h]}: the Crop / Draw sheet at its ideal size (it opens at its minimum under the harness).
        if let size = step["sheetSize"] as? [Double], size.count == 2, let sheet = mainWindow()?.attachedSheet {
            sheet.setContentSize(NSSize(width: size[0], height: size[1]))
        }
        // {"editorDoc": {"tool": "Paint", "crop": [x, y, w, h], "paint": [[[x, y], ...]], "erase": [[[x, y], ...]],
        //  "width": 0.03, "turn": 1}}: reopen the editor on the latest image with this crop and these strokes (normalized).
        // "turn": n (1-based image number) reopens an older image instead of the latest.
        if let spec = step["editorDoc"] as? [String: Any],
           let t = (spec["turn"] as? Int).flatMap({ n in model.turns.first { $0.kind == .image && $0.number == n } })
               ?? model.latestImageTurn {
            if model.editorDocument != nil {
                // Let the sheet go away first, or SwiftUI keeps the old editor (its tool and strokes).
                model.closeEditor()
                try? await _Concurrency.Task.sleep(for: .seconds(1.2))
            }
            model.editorStartTool = spec["tool"] as? String
            model.openEditor(t.id)
            if var doc = model.editorDocument {
                if let c = spec["crop"] as? [Double], c.count == 4 {
                    doc.crop = NormalizedRect(x: c[0], y: c[1], width: c[2], height: c[3])
                }
                let width = spec["width"] as? Double ?? 0.03
                func strokes(_ key: String, erases: Bool) -> [PaintStroke] {
                    (spec[key] as? [[[Double]]] ?? []).map { line in
                        PaintStroke(points: line.map { NormalizedPoint(x: $0[0], y: $0[1]) }, width: erases ? width * 1.6 : width,
                                    erases: erases)
                    }
                }
                doc.strokes = strokes("paint", erases: false) + strokes("erase", erases: true)
                model.editorDocument = doc
            }
        }
        if step["applyEditor"] != nil, let doc = model.editorDocument { model.applyEditor(doc) }
        if let name = step["capture"] as? String {
            let window: NSWindow? = switch step["window"] as? String {
            case "settings": settingsWindow()
            case "settingsSheet": settingsWindow()?.attachedSheet  // Add Model over Settings → Models
            case "sheet": mainWindow()?.attachedSheet  // the Crop / Draw editor
            case "addModel": addModelWindow
            default: mainWindow()
            }
            capture(name: name, window: window, out: out, region: step["region"] != nil)
        }
        if let edge = step["scroll"] as? String { scrollChat(toTop: edge == "top", overscroll: step["overscroll"] as? Double ?? 0) }
        if let seconds = step["sleep"] as? Double { try? await _Concurrency.Task.sleep(for: .seconds(seconds)) }
        if let timeout = step["waitIdle"] as? Double {
            let deadline = Date().addingTimeInterval(timeout)
            try? await _Concurrency.Task.sleep(for: .milliseconds(500))  // a send plans before its turn exists
            while Date() < deadline {
                let busy = model.turns.contains { $0.versions.contains { $0.isActive } }
                    || model.planningMessage != nil || !model.activeRuns.isEmpty
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
                + "planning=\(model.planningMessage ?? "-") turns=\(model.turns.count) last=\(last) draft=\(model.draft)"
                + " decision=\(model.routeDecision.map { "\($0.intent) \($0.input_result_ids ?? []) \($0.provider)" } ?? "-")")
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

    /// The Settings scene cannot be opened from here (showSettingsWindow: is gone); host its view instead.
    private static func showSettings(_ model: AppModel, height: Double = 460) {
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 620, height: height),
            styleMask: [.titled, .closable], backing: .buffered, defer: false)
        window.title = "Settings"
        window.identifier = NSUserInterfaceItemIdentifier("HarnessSettings")
        window.isReleasedWhenClosed = false
        window.contentView = NSHostingView(rootView: SettingsView().environment(model)
            .frame(width: 620, height: height).tint(Theme.accent))
        window.center()
        window.orderFrontRegardless()
    }

    private static func settingsWindow() -> NSWindow? {
        NSApp.windows.first { $0.isVisible && $0.identifier?.rawValue.contains("Settings") == true }
            ?? NSApp.windows.first { $0.isVisible && ["Models", "Memory", "Intelligence", "Engine"].contains($0.title) }
    }

    private static func mainWindow() -> NSWindow? {
        NSApp.windows.filter { $0.isVisible && $0 !== settingsWindow() && $0 !== addModelWindow && $0.sheetParent == nil }
            .max { $0.frame.width < $1.frame.width }
    }

    /// Opens a SwiftUI menu (an NSPopUpButton underneath) found by its accessibility label, captures the
    /// screen around the window while the menu tracks, then closes it.
    private static func openMenu(labeled label: String, in window: NSWindow?, keys: [Int], name: String, out: URL) {
        guard let window, let content = window.contentView else { return log("openMenu: no window") }
        func find(_ view: NSView) -> NSView? {
            if view.accessibilityLabel() == label || (view as? NSPopUpButton)?.accessibilityLabel() == label { return view }
            for sub in view.subviews { if let hit = find(sub) { return hit } }
            return nil
        }
        guard let button = find(content) else {
            var classes: [String] = []
            func walk(_ v: NSView) {
                if v is NSButton || v.accessibilityLabel() != nil { classes.append("\(type(of: v))[\(v.accessibilityLabel() ?? "-")]") }
                v.subviews.forEach(walk)
            }
            walk(content)
            return log("openMenu: no view labeled \(label); candidates: \(classes.prefix(60).joined(separator: ", "))")
        }
        log("openMenu: \(type(of: button))")
        NSApp.activate(ignoringOtherApps: true)
        window.makeKeyAndOrderFront(nil)
        let file = out.appendingPathComponent("\(name).png")
        let shot = Process()
        shot.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
        let f = window.frame
        let top = (NSScreen.screens.first?.frame.maxY ?? f.maxY) - f.maxY
        shot.arguments = ["-x", "-T", "2", "-R\(Int(f.minX)),\(Int(top)),\(Int(f.width)),\(Int(f.height))", file.path]
        try? shot.run()
        for (i, code) in keys.enumerated() {
            let timer = Timer(timeInterval: 0.6 + Double(i) * 0.4, repeats: false) { _ in
                for type in [CGEventType.keyDown, .keyUp] {
                    CGEvent(keyboardEventSource: nil, virtualKey: CGKeyCode(code), keyDown: type == .keyDown)?.post(tap: .cghidEventTap)
                }
            }
            RunLoop.main.add(timer, forMode: .common)
        }
        let close = Timer(timeInterval: 3.2, repeats: false) { _ in
            (button as? NSPopUpButton)?.menu?.cancelTracking()
            // Escape closes any SwiftUI menu that is not an NSPopUpButton.
            for type in [CGEventType.keyDown, .keyUp] {
                CGEvent(keyboardEventSource: nil, virtualKey: 53, keyDown: type == .keyDown)?.post(tap: .cghidEventTap)
            }
        }
        RunLoop.main.add(close, forMode: .common)
        if let popup = button as? NSPopUpButton { popup.performClick(nil) } else {
            let point = button.convert(NSPoint(x: button.bounds.midX, y: button.bounds.midY), to: nil)
            let screen = window.convertPoint(toScreen: point)
            let y = (NSScreen.screens.first?.frame.maxY ?? 0) - screen.y
            for type in [CGEventType.leftMouseDown, .leftMouseUp] {
                CGEvent(mouseEventSource: nil, mouseType: type, mouseCursorPosition: CGPoint(x: screen.x, y: y),
                        mouseButton: .left)?.post(tap: .cghidEventTap)
            }
        }
        RunLoop.main.run(until: Date().addingTimeInterval(3.6))
        shot.waitUntilExit()
        log("openMenu capture \(file.path) status \(shot.terminationStatus)")
    }

    /// Scrolls the chat transcript (the widest scroll view in the main window) to its top or bottom, so a chat
    /// taller than the window can be captured in parts.
    /// `overscroll` pulls the transcript that many points past the edge, as a rubber-band scroll would, so the
    /// first message clears the fade under the title bar.
    private static func scrollChat(toTop: Bool, overscroll: Double) {
        guard let content = mainWindow()?.contentView else { return log("scroll: no main window") }
        func scrollViews(_ view: NSView) -> [NSScrollView] {
            (view as? NSScrollView).map { [$0] } ?? [] + view.subviews.flatMap(scrollViews)
        }
        guard let scroll = scrollViews(content).max(by: { $0.frame.width < $1.frame.width }),
              let document = scroll.documentView else { return log("scroll: no chat scroll view") }
        let clip = scroll.contentView
        let maxY = max(0, document.frame.height - clip.bounds.height)
        let pastEdge = document.isFlipped == toTop ? -overscroll : overscroll
        let y = (document.isFlipped == toTop ? 0 : maxY) + pastEdge
        // setBoundsOrigin, not scroll(to:), which clamps to the document.
        clip.setBoundsOrigin(NSPoint(x: clip.bounds.origin.x, y: y))
        scroll.reflectScrolledClipView(clip)
        log("scroll \(toTop ? "top" : "bottom") document=\(Int(document.frame.height)) visible=\(Int(clip.bounds.height))")
    }

    /// The window as it is on screen, through screencapture (ImageRenderer cannot draw native controls).
    /// `region`: the window's area of the screen, so popovers over it (the guide's tips) are included.
    static func capture(name: String, window: NSWindow?, out: URL, region: Bool = false) {
        guard let window else {
            return log("capture \(name): no such window; " + NSApp.windows.map {
                "\($0.title)|\($0.identifier?.rawValue ?? "-")|visible=\($0.isVisible)" }.joined(separator: ", "))
        }
        // An inactive window draws its sidebar and composer late or not at all; bring the app forward first.
        NSApp.activate(ignoringOtherApps: true)
        window.makeKeyAndOrderFront(nil)
        RunLoop.main.run(until: Date().addingTimeInterval(0.8))
        let file = out.appendingPathComponent("\(name).png")
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
        if region, let screen = window.screen ?? NSScreen.main {
            // screencapture -R counts from the top-left of the main display; AppKit from the bottom-left.
            let f = window.frame.insetBy(dx: -40, dy: -40)
            let top = (NSScreen.screens.first?.frame.maxY ?? screen.frame.maxY) - f.maxY
            process.arguments = ["-x", "-R\(Int(f.minX)),\(Int(top)),\(Int(f.width)),\(Int(f.height))", file.path]
        } else {
            process.arguments = ["-x", "-o", "-l\(window.windowNumber)", file.path]
        }
        do {
            try process.run()
            process.waitUntilExit()
            log("capture \(file.path) status \(process.terminationStatus)")
        } catch {
            log("capture \(name) failed: \(error)")
        }
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
    static func renderChat(_ model: AppModel, name: String, out: URL, width: Double = 780) {
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
                if model.turns.isEmpty && model.planningMessage == nil && model.heldMessage == nil {
                    EmptyStateView(starters: ["A fox logo, flat vector, two colors"])
                }
                ForEach(model.turns) { TurnView(turn: $0) }
                PendingMessages()
            }
            .frame(width: width)
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
