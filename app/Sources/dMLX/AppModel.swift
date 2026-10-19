import AppKit
import DMLXKit
import Foundation
import Observation

enum Aspect: String, CaseIterable, Identifiable {
    case square = "Square"
    case landscape = "Landscape"
    case portrait = "Portrait"

    var id: String { rawValue }
    var size: String {
        switch self {
        case .square: return "1024x1024"
        case .landscape: return "1216x832"
        case .portrait: return "832x1216"
        }
    }
    var label: String {
        switch self {
        case .square: return "Square · 1024²"
        case .landscape: return "Landscape · 3:2"
        case .portrait: return "Portrait · 2:3"
        }
    }
}

struct Attachment: Identifiable, Hashable {
    var id: String  // result id on the engine
    var label: String
}

@MainActor
@Observable
final class AppModel {
    // Engine
    let engine = EngineProcess()
    var engineState: EngineProcess.State = .stopped
    var models: [EngineModel] = []
    var memory: MemoryStatus?

    // History
    private var store: Store?
    var sessions: [Session] = []
    var currentSessionId: String?
    var turns: [Turn] = []

    // Composer
    var draft = ""
    var targetTurnId: String?
    var attachments: [Attachment] = []
    var aspect: Aspect = .square
    var preferredGenerateModel: String = UserDefaults.standard.string(forKey: "generateModel") ?? "flux2-klein-4b" {
        didSet { UserDefaults.standard.set(preferredGenerateModel, forKey: "generateModel") }
    }

    // UI
    var toast: String?
    var showOnboarding = false
    var storeError: String?

    private var streams: [String: _Concurrency.Task<Void, Never>] = [:]
    private var poller: _Concurrency.Task<Void, Never>?
    private var toastTask: _Concurrency.Task<Void, Never>?
    let images = ImageCache()

    init() {
        do {
            store = try Store(path: Store.defaultURL())
            sessions = try store?.sessions() ?? []
        } catch {
            storeError = "\(error)"
        }
        engine.onStateChange = { [weak self] state in
            guard let self else { return }
            self.engineState = state
            switch state {
            case .running: self.onEngineReady()
            case .starting, .failed: self.failActiveJobs()
            case .stopped: break
            }
        }
        engine.start()
        if let first = sessions.first {
            select(first.id)
        }
    }

    var client: EngineClient? { engine.client }

    // MARK: engine

    private func onEngineReady() {
        poller?.cancel()
        poller = _Concurrency.Task { [weak self] in
            var first = true
            while !_Concurrency.Task.isCancelled {
                await self?.refreshModels(openOnboarding: first)
                await self?.refreshMemory()
                first = false
                try? await _Concurrency.Task.sleep(for: .seconds(2))
            }
        }
    }

    /// Jobs in flight die with the engine; close their streams so the UI shows Retry.
    private func failActiveJobs() {
        for (_, task) in streams { task.cancel() }
        for t in turns.indices {
            for v in turns[t].versions.indices where turns[t].versions[v].isActive {
                turns[t].versions[v].status = .error
                turns[t].versions[v].errorCode = "engine_stopped"
                turns[t].versions[v].errorMessage = "The engine stopped unexpectedly and was restarted. Retry to run it again."
            }
            persist(t)
        }
        streams = [:]
    }

    func refreshModels(openOnboarding: Bool = false) async {
        guard let client else { return }
        if let list = try? await client.models() {
            models = list
            if openOnboarding && defaultModel(for: .t2i) == nil {
                showOnboarding = true
            }
        }
    }

    func refreshMemory() async {
        guard let client else { return }
        memory = try? await client.memory()
    }

    func download(_ model: EngineModel) {
        _Concurrency.Task {
            do {
                try await client?.pull(model: model.id)
                await refreshModels()
            } catch {
                show("Download failed: \(error.localizedDescription)")
            }
        }
    }

    var isEngineRunning: Bool {
        if case .running = engineState { return true }
        return false
    }

    var engineStatusText: String {
        switch engineState {
        case .stopped: return "Engine stopped"
        case .starting: return "Starting engine…"
        case .running: return memory.map { readyText($0) } ?? "Ready"
        case .failed(let message): return message
        }
    }

    private func readyText(_ m: MemoryStatus) -> String {
        let resident = Set(m.memory.resident.map(\.model))
        let names = models.filter { resident.contains($0.id) }.map(\.name)
        let warm = names.isEmpty ? "No models in memory" : names.joined(separator: ", ") + " in memory"
        return "\(warm) · \(String(format: "%.1f", max(m.memory.headroom_gb, 0))) GB free to use"
    }

    // MARK: model choice

    func defaultModel(for task: GenTask) -> EngineModel? {
        let downloaded = models.filter { $0.downloaded && $0.supports(task) && $0.fits_this_mac }
        // One model for both tasks keeps generate/edit turns warm; others are opt-in.
        let preference = task == .t2i ? [preferredGenerateModel, "flux2-klein-4b", "z-image-turbo"] : ["flux2-klein-4b"]
        for id in preference {
            if let m = downloaded.first(where: { $0.id == id }) { return m }
        }
        return downloaded.first
    }

    func modelName(_ id: String) -> String {
        models.first(where: { $0.id == id })?.name ?? id
    }

    // MARK: sessions

    func newSession() {
        currentSessionId = nil
        turns = []
        clearComposerContext()
    }

    func select(_ id: String) {
        currentSessionId = id
        turns = (try? store?.turns(session: id)) ?? []
        // Jobs that were running when the app quit cannot be resumed; mark them.
        for i in turns.indices {
            for v in turns[i].versions.indices where turns[i].versions[v].isActive && streams[turns[i].versions[v].id] == nil {
                turns[i].versions[v].status = .cancelled
                turns[i].versions[v].errorMessage = "Stopped when the app closed."
            }
        }
        clearComposerContext()
    }

    func delete(session id: String) {
        try? store?.delete(session: id)
        sessions.removeAll { $0.id == id }
        if currentSessionId == id { newSession() }
    }

    private func ensureSession(title: String) -> String {
        if let id = currentSessionId { return id }
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        let session = Session(title: String(trimmed.prefix(48)))
        try? store?.save(session)
        sessions.insert(session, at: 0)
        currentSessionId = session.id
        return session.id
    }

    private func touchSession() {
        guard let id = currentSessionId, let i = sessions.firstIndex(where: { $0.id == id }) else { return }
        sessions[i].updated = .now
        try? store?.save(sessions[i])
        let s = sessions.remove(at: i)
        sessions.insert(s, at: 0)
    }

    // MARK: routing

    var latestImageTurn: Turn? {
        turns.last(where: { $0.kind == .image && $0.current?.resultId != nil })
    }

    var targetTurn: Turn? {
        targetTurnId.flatMap { id in turns.first(where: { $0.id == id }) }
    }

    var routingContext: RoutingContext {
        RoutingContext(
            targetResultId: targetTurn?.current?.resultId, targetTurnId: targetTurnId,
            latestResultId: latestImageTurn?.current?.resultId, latestTurnId: latestImageTurn?.id,
            attachmentIds: attachments.map(\.id))
    }

    var routePreview: String {
        switch Router.route(draft, routingContext) {
        case .none:
            return "Model picked for you"
        case .generate:
            return defaultModel(for: .t2i).map { "Will generate · \($0.name)" } ?? "No image model downloaded"
        case .edit(_, let fromTurn):
            let name = defaultModel(for: .edit)?.name ?? "no edit model"
            if let fromTurn, let t = turns.first(where: { $0.id == fromTurn }) {
                return "Will edit image \(t.number) · \(name)"
            }
            return "Will edit the attached image · \(name)"
        case .video:
            return "Video isn't in this beta yet"
        }
    }

    private func clearComposerContext() {
        targetTurnId = nil
        attachments = []
    }

    // MARK: sending

    func send(_ text: String? = nil) {
        let message = (text ?? draft).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !message.isEmpty else { return }
        let intent = Router.route(message, routingContext)
        let sessionId = ensureSession(title: message)
        let attachedIds = attachments.map(\.id)
        appendTurn(Turn(sessionId: sessionId, index: turns.count, kind: .user, text: message, attachments: attachedIds))
        draft = ""

        switch intent {
        case .none:
            return
        case .video:
            appendTurn(Turn(
                sessionId: sessionId, index: turns.count, kind: .notice,
                text: "Video generation isn't in this beta yet. Images and edits are; video is next on the roadmap."))
        case .generate:
            guard let model = defaultModel(for: .t2i) else { return noModel(sessionId) }
            startImageTurn(sessionId: sessionId, prompt: message, model: model, task: .t2i, sources: [], route: "Generated",
                           size: aspect.size)
        case .edit(let sources, let fromTurn):
            guard let model = defaultModel(for: .edit) else { return noModel(sessionId) }
            var route = "Edited from the attached image"
            if let fromTurn, let t = turns.first(where: { $0.id == fromTurn }) {
                route = "Edited from image \(t.number) · v\(t.activeVersion + 1)"
            }
            if sources.count > 1 { route += " + \(sources.count - 1) reference\(sources.count > 2 ? "s" : "")" }
            startImageTurn(sessionId: sessionId, prompt: message, model: model, task: .edit, sources: sources, route: route,
                           size: nil)
        }
        clearComposerContext()
        touchSession()
    }

    private func noModel(_ sessionId: String) {
        appendTurn(Turn(sessionId: sessionId, index: turns.count, kind: .notice,
                        text: "No suitable model is downloaded yet. Open Models to download one."))
        showOnboarding = true
    }

    private func appendTurn(_ turn: Turn) {
        turns.append(turn)
        try? store?.save(turn)
    }

    private func persist(_ index: Int) {
        guard turns.indices.contains(index) else { return }
        try? store?.save(turns[index])
    }

    private var nextImageNumber: Int {
        (turns.filter { $0.kind == .image }.map(\.number).max() ?? 0) + 1
    }

    private func startImageTurn(
        sessionId: String, prompt: String, model: EngineModel, task: GenTask, sources: [String], route: String, size: String?
    ) {
        let version = ImageVersion(prompt: prompt, model: model.id, task: task, size: size ?? "auto", sourceIds: sources)
        let turn = Turn(sessionId: sessionId, index: turns.count, kind: .image, text: prompt, number: nextImageNumber,
                        route: route, versions: [version], activeVersion: 0)
        appendTurn(turn)
        run(turnId: turn.id, versionId: version.id)
    }

    func run(turnId: String, versionId: String, force: Bool = false) {
        guard let client else {
            update(turnId, versionId) { $0.status = .error; $0.errorMessage = "The engine is not running yet." }
            return
        }
        guard let version = find(turnId, versionId) else { return }
        update(turnId, versionId) {
            $0.status = .queued; $0.phase = "queued"; $0.errorCode = nil; $0.errorMessage = nil; $0.step = 0
        }
        let task = _Concurrency.Task { [weak self] in
            do {
                let stream = client.run(
                    task: version.task, prompt: version.prompt, model: version.model,
                    size: version.size == "auto" ? nil : version.size, seed: version.seed,
                    sourceIds: version.sourceIds, force: force)
                for try await event in stream {
                    self?.apply(event, turnId: turnId, versionId: versionId)
                }
            } catch is CancellationError {
            } catch {
                let err = error as? EngineClientError
                self?.update(turnId, versionId) {
                    $0.status = .error
                    $0.errorCode = err?.code
                    $0.errorMessage = error.localizedDescription
                }
            }
            // The stream can end without a terminal event (engine crashed or restarted).
            self?.update(turnId, versionId) {
                if $0.isActive {
                    $0.status = .error
                    $0.errorCode = "engine_stopped"
                    $0.errorMessage = "The engine stopped unexpectedly and was restarted. Retry to run it again."
                }
            }
            self?.streams[versionId] = nil
            self?.persistTurn(turnId)
        }
        streams[versionId] = task
    }

    private func apply(_ event: EngineEvent, turnId: String, versionId: String) {
        let p = event.payload
        update(turnId, versionId) { v in
            if let job = p.job_id { v.jobId = job }
            switch event.name {
            case "queued":
                v.status = .queued
                v.phase = (p.position ?? 1) > 1 ? "Waiting for \(p.position! - 1) job(s)" : "Starting"
                v.seed = p.seed ?? v.seed
                v.totalSteps = p.steps ?? v.totalSteps
                if let w = p.width, let h = p.height { v.size = "\(w)x\(h)" }
            case "plan":
                v.status = .rendering
                v.strategy = p.strategy
            case "loading":
                v.status = .rendering
                v.phase = "Loading \(Self.componentName(p.component))"
                if let c = p.component, !v.loads.contains(c) { v.loads.append(c) }
            case "encoding":
                v.phase = "Reading your prompt"
            case "step":
                v.status = .rendering
                v.step = p.step ?? v.step
                v.totalSteps = p.total ?? v.totalSteps
                v.phase = "Step \(v.step) of \(v.totalSteps)"
            case "decoding":
                v.phase = "Finishing"
            case "image":
                v.resultId = p.result_id
            case "done":
                v.status = .done
                v.resultId = p.result_ids?.first ?? v.resultId
                v.loads = p.report?.loads ?? v.loads
                v.strategy = p.report?.strategy ?? v.strategy
                v.seconds = p.report?.total_seconds
                v.phase = "Done"
            case "error":
                v.status = .error
                v.errorCode = p.code
                v.errorMessage = p.message
            case "cancelled":
                v.status = .cancelled
                v.errorMessage = "Cancelled"
            default:
                break
            }
        }
        if ["done", "error", "cancelled"].contains(event.name) {
            persistTurn(turnId)
            if event.name == "done" { _Concurrency.Task { await refreshMemory() } }
        }
    }

    static func componentName(_ c: String?) -> String {
        switch c {
        case "text_encoder": return "text encoder"
        case "transformer": return "image model"
        case "vae": return "decoder"
        default: return "model"
        }
    }

    /// Shown in a result's details when the engine had to save memory; nothing for the normal mode.
    static func strategyLabel(_ strategy: String?) -> String? {
        switch strategy {
        case "phase_free": return "memory-saving mode"
        case "sequential": return "low-memory mode"
        case "streamed": return "lowest-memory mode"
        default: return nil
        }
    }

    private func find(_ turnId: String, _ versionId: String) -> ImageVersion? {
        turns.first(where: { $0.id == turnId })?.versions.first(where: { $0.id == versionId })
    }

    private func update(_ turnId: String, _ versionId: String, _ change: (inout ImageVersion) -> Void) {
        guard let t = turns.firstIndex(where: { $0.id == turnId }),
              let v = turns[t].versions.firstIndex(where: { $0.id == versionId }) else { return }
        change(&turns[t].versions[v])
    }

    private func persistTurn(_ turnId: String) {
        if let i = turns.firstIndex(where: { $0.id == turnId }) { persist(i) }
    }

    // MARK: result actions

    func pickVersion(_ turnId: String, _ index: Int) {
        guard let t = turns.firstIndex(where: { $0.id == turnId }) else { return }
        turns[t].activeVersion = index
        persist(t)
    }

    func variation(_ turnId: String) {
        guard let t = turns.firstIndex(where: { $0.id == turnId }), let base = turns[t].current else { return }
        var v = ImageVersion(prompt: base.prompt, model: base.model, task: base.task, size: base.size,
                             sourceIds: base.sourceIds, label: "Variation")
        v.seed = nil
        turns[t].versions.append(v)
        turns[t].activeVersion = turns[t].versions.count - 1
        persist(t)
        run(turnId: turnId, versionId: v.id)
    }

    func retry(_ turnId: String, force: Bool) {
        guard let t = turns.firstIndex(where: { $0.id == turnId }), let v = turns[t].current else { return }
        run(turnId: turnId, versionId: v.id, force: force)
    }

    func cancel(_ turnId: String) {
        guard let v = turns.first(where: { $0.id == turnId })?.current else { return }
        if let job = v.jobId {
            _Concurrency.Task { try? await client?.cancel(job: job) }
        } else {
            streams[v.id]?.cancel()
            update(turnId, v.id) { $0.status = .cancelled; $0.errorMessage = "Cancelled" }
        }
    }

    func edit(_ turnId: String) {
        targetTurnId = turnId
    }

    func useAsReference(_ turnId: String) {
        guard let t = turns.first(where: { $0.id == turnId }), let rid = t.current?.resultId else { return }
        if !attachments.contains(where: { $0.id == rid }) {
            attachments.append(Attachment(id: rid, label: "Image \(t.number)"))
        }
    }

    func removeAttachment(_ id: String) {
        attachments.removeAll { $0.id == id }
    }

    func attach(files: [URL]) {
        guard let client else { return show("The engine is still starting.") }
        _Concurrency.Task {
            for url in files.prefix(3) {
                do {
                    let id = try await client.upload(file: url)
                    attachments.append(Attachment(id: id, label: url.lastPathComponent))
                } catch {
                    show("Couldn't attach \(url.lastPathComponent): \(error.localizedDescription)")
                }
            }
        }
    }

    func save(_ turnId: String) {
        guard let t = turns.first(where: { $0.id == turnId }), let v = t.current, let rid = v.resultId else { return }
        let src = ImageCache.fileURL(rid)
        let dir = FileManager.default.urls(for: .picturesDirectory, in: .userDomainMask)[0].appendingPathComponent("dMLX")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let slug = v.prompt.lowercased().components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }.prefix(6).joined(separator: "-")
        let dest = dir.appendingPathComponent("\(slug.isEmpty ? "image" : slug)-\(rid.suffix(6)).png")
        do {
            if !FileManager.default.fileExists(atPath: dest.path) {
                try FileManager.default.copyItem(at: src, to: dest)
            }
            NSWorkspace.shared.activateFileViewerSelecting([dest])
            show("Saved to Pictures/dMLX with its prompt and seed")
        } catch {
            show("Couldn't save: \(error.localizedDescription)")
        }
    }

    func show(_ message: String) {
        toast = message
        toastTask?.cancel()
        toastTask = _Concurrency.Task {
            try? await _Concurrency.Task.sleep(for: .seconds(3))
            if !_Concurrency.Task.isCancelled { toast = nil }
        }
    }

    func shutdown() {
        engine.stop()
    }
}

/// Loads result PNGs straight from the engine's results folder (same machine), with a small cache.
@MainActor
final class ImageCache {
    private var cache: [String: NSImage] = [:]
    private var order: [String] = []

    static func fileURL(_ resultId: String) -> URL {
        let base = ProcessInfo.processInfo.environment["DMLX_HOME"].map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("dMLX")
        return base.appendingPathComponent("results/\(resultId).png")
    }

    func image(_ resultId: String) -> NSImage? {
        if let img = cache[resultId] { return img }
        guard let img = NSImage(contentsOf: Self.fileURL(resultId)) else { return nil }
        cache[resultId] = img
        order.append(resultId)
        if order.count > 64 { cache[order.removeFirst()] = nil }
        return img
    }
}
