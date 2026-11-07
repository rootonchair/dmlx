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
    var role: InputRole? = nil
    var metadata: [String: String] = [:]

    func mediaInput(order: Int) -> MediaInput {
        MediaInput(id: id, role: role ?? (order == 0 ? .source : .reference), order: order, metadata: metadata)
    }
}

@MainActor
@Observable
final class AppModel {
    // Engine
    let engine = EngineProcess()
    var engineState: EngineProcess.State = .stopped
    var models: [EngineModel] = []
    var capabilityCatalog: CapabilityCatalog?
    var memory: MemoryStatus?

    // History
    private var store: Store?
    var sessions: [Session] = []
    var currentSessionId: String?
    var turns: [Turn] = []

    // Composer
    var draft = ""
    var attachments: [Attachment] = []
    var aspect: Aspect = .square
    var routeDecision: WorkflowDecision?
    var isResolvingRoute = false
    var promptEnhancementMode: PromptEnhancementMode = AppModel.savedPromptMode() {
        didSet { UserDefaults.standard.set(promptEnhancementMode.rawValue, forKey: "promptEnhancementMode") }
    }
    var promptSuggestion: CompiledPromptResponse?
    var promptSuggestionApplied = false
    var isEnhancingPrompt = false
    var editorDocument: EditorDocument?
    var editorImage: NSImage?
    var editorTurnNumber = 0
    var isExportingEditor = false
    /// "auto" or a model id; chosen from the composer's model menu (and mirrored in Settings).
    var modelChoice: String = AppModel.savedModelChoice() {
        didSet { UserDefaults.standard.set(modelChoice, forKey: "modelChoice") }
    }

    private static func savedModelChoice() -> String {
        if let saved = UserDefaults.standard.string(forKey: "modelChoice") { return saved }
        // Earlier builds kept a "model for new images" setting; carry a non-default pick over.
        if let old = UserDefaults.standard.string(forKey: "generateModel"), old != "flux2-klein-4b" { return old }
        return "auto"
    }

    private static func savedPromptMode() -> PromptEnhancementMode {
        UserDefaults.standard.string(forKey: "promptEnhancementMode")
            .flatMap(PromptEnhancementMode.init(rawValue:)) ?? .off
    }

    // UI
    var toast: String?
    var showOnboarding = false
    var onboardingStep: OnboardingStep = .welcome
    static let onboardingCompletedKey = "onboardingCompleted"
    var storeError: String?

    private var streams: [String: _Concurrency.Task<Void, Never>] = [:]
    /// Image jobs in flight, by version id: the chat each one belongs to. The engine makes one image at a time.
    private(set) var activeRuns: [String: String] = [:]
    /// Turns with a job in flight whose chat is not on screen: their progress and result still land and are saved.
    private var offscreenTurns: [String: Turn] = [:]
    /// A message from another chat is waiting for the running image before the agent plans it.
    private(set) var isWaitingForImage = false
    private var poller: _Concurrency.Task<Void, Never>?
    private var toastTask: _Concurrency.Task<Void, Never>?
    private var routeTask: _Concurrency.Task<Void, Never>?
    private var routeRequest: RouteRequestIdentity?
    private var routeDecisionKey: String?
    private var pendingRoute: PendingRoute?
    /// The agent's question when it could not tell what a sent message means.
    private(set) var pendingAsk: String?
    /// A sent message while it is planned: out of the composer, shown in the chat, never dropped silently.
    private(set) var planningMessage: String?
    /// The open Settings tab.
    var settingsTab = "models"
    private var planTask: _Concurrency.Task<Void, Never>?
    /// The local prompt assistant is downloaded: sent messages are planned by it.
    var promptAssistantInstalled = false
    private let editorStore = EditorDocumentStore.default()
    let images = ImageCache()

    private struct PendingRoute {
        var message: String
        var attachmentInputs: [MediaInput]
        var latestResultId: String?
        var latestTurnId: String?
        var promptSpec: PromptSpec
        /// False for an answer to the agent's question: the attachments are shown on the message it answers.
        var showsAttachments = true

        var attachmentIds: [String] { attachmentInputs.map(\.id) }
    }

    /// The chat image a prepared image (Crop / Draw) was made from, by its base result id.
    private func preparedFrom(_ input: MediaInput) -> Turn? {
        guard let base = input.metadata["base_result_id"] else { return nil }
        return turns.first { $0.versions.contains { $0.resultId == base } }
    }

    /// What the agent should know about each attachment it cannot tell from the pixels alone.
    private func attachmentNotes(_ inputs: [MediaInput]) -> [String: String] {
        var notes: [String: String] = [:]
        for input in inputs where input.metadata["base_result_id"] != nil {
            let source = preparedFrom(input).map { "image \($0.number)" } ?? "a chat image"
            notes[input.id] = "\(source) prepared in Crop / Draw: the user cropped it and may have painted marks on it "
                + "to show what to change"
        }
        return notes
    }

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
        scheduleRoutePreview()
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
        activeRuns = [:]
    }

    func refreshModels(openOnboarding: Bool = false) async {
        guard let client else { return }
        if let list = try? await client.models() {
            models = list
            capabilityCatalog = try? await client.capabilities()
            promptAssistantInstalled = (try? await client.promptAssistant())?.installed ?? false
            // First run, or no image model yet. Someone who finished setup without the agent is not asked again.
            let completed = UserDefaults.standard.bool(forKey: Self.onboardingCompletedKey)
            if openOnboarding && (defaultModel(for: .t2i) == nil || (!completed && !promptAssistantInstalled)) {
                onboardingStep = .welcome
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

    // MARK: adding and removing models

    func catalogSuggestions() async throws -> [CatalogSuggestion] {
        guard let client else { throw EngineClientError.notRunning }
        return try await client.catalogSuggestions()
    }

    func inspectModel(_ text: String) async throws -> ModelInspection {
        guard let client else { throw EngineClientError.notRunning }
        return try await client.inspect(repo: text)
    }

    /// Adds the model and starts its download; progress shows in Settings → Models.
    func addModel(_ repo: String) async throws {
        guard let client else { throw EngineClientError.notRunning }
        let added = try await client.addModel(repo: repo)
        await refreshModels()
        show(added.downloaded ? "\(added.name) added (already on this Mac)" : "Downloading \(added.name)…")
    }

    /// Built-in models keep their entry (and can be downloaded again); added models leave the list.
    func remove(_ m: EngineModel) {
        _Concurrency.Task {
            do {
                guard let client else { throw EngineClientError.notRunning }
                let freed = m.isUserAdded ? try await client.removeModel(model: m.id) : try await client.deleteFiles(model: m.id)
                if modelChoice == m.id && m.isUserAdded { modelChoice = "auto" }
                await refreshModels()
                show(freed > 0 ? "Removed \(m.name) · freed \(String(format: "%.1f", freed)) GB" : "Removed \(m.name)")
            } catch {
                show("Couldn't remove \(m.name): \(error.localizedDescription)")
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

    /// Every downloaded model is visible for explicit selection. Auto still uses
    /// only models that pass this Mac's safety tier.
    var choosableModels: [EngineModel] {
        models.filter(\.downloaded)
    }

    /// The model the menu's choice resolves to, or nil for Auto (or a choice that is no longer available).
    var chosenModel: EngineModel? {
        modelChoice == "auto" ? nil : choosableModels.first(where: { $0.id == modelChoice })
    }

    func defaultModel(for task: GenTask) -> EngineModel? {
        defaultModel(for: task == .edit ? "image.edit" : "image.generate")
    }

    /// On Auto, the model the agent picked for this message when it can run it; otherwise the usual choice.
    private func model(for operation: String, decision: WorkflowDecision) -> EngineModel? {
        if chosenModel == nil, let picked = decision.model,
           let model = models.first(where: { $0.id == picked && $0.downloaded }),
           capabilityCatalog?.operations[operation]?.contains(where: { $0.model == picked }) == true {
            return model
        }
        return defaultModel(for: operation)
    }

    func defaultModel(for operation: String) -> EngineModel? {
        let capabilities = capabilityCatalog?.operations[operation, default: []] ?? []
        if let chosen = chosenModel,
           capabilities.contains(where: { $0.model == chosen.id && $0.downloaded }) {
            return chosen
        }
        let capableIds = Set(capabilities
            .filter { $0.downloaded && $0.fits_this_mac }.map(\.model))
        let downloaded = models.filter { capableIds.contains($0.id) && $0.autoSelectable }
        // Auto: one model for both tasks keeps generate/edit turns warm.
        for id in ["flux2-klein-4b", "z-image-turbo"] {
            if let m = downloaded.first(where: { $0.id == id }) { return m }
        }
        return downloaded.first
    }

    /// One line for the model menu: what the model does and its size.
    static func modelSummary(_ m: EngineModel) -> String {
        let does = m.supports(.edit) ? "New images and edits" : "New images only"
        let license = m.requiresLicenseAcceptance ? " · Research-only" : ""
        let maturity = m.isExperimental ? " · Experimental" : ""
        let compatibility = m.fits_this_mac ? "" : " · May not fit this Mac"
        return m.size_gb.map {
            "\(does) · \(String(format: "%.1f", $0)) GB\(license)\(maturity)\(compatibility)"
        } ?? does + license + maturity + compatibility
    }

    /// Appended to the route preview when sending would swap another model out of memory.
    private func loadHint(_ m: EngineModel) -> String {
        guard !m.resident, models.contains(where: { $0.resident && $0.id != m.id }), let gb = m.size_gb else { return "" }
        return " · loads \(String(format: "%.1f", gb)) GB"
    }

    func modelName(_ id: String) -> String {
        models.first(where: { $0.id == id })?.name ?? id
    }

    // MARK: sessions

    func newSession() {
        abandonPlanning()
        keepActiveTurnsOffscreen()
        currentSessionId = nil
        turns = []
        clearComposerContext()
    }

    func select(_ id: String) {
        abandonPlanning()
        keepActiveTurnsOffscreen()
        currentSessionId = id
        turns = (try? store?.turns(session: id)) ?? []
        // Jobs still running from this chat come back with their live state.
        for i in turns.indices {
            if let live = offscreenTurns.removeValue(forKey: turns[i].id) { turns[i] = live }
        }
        // Jobs that were running when the app quit cannot be resumed; mark them.
        for i in turns.indices {
            for v in turns[i].versions.indices where turns[i].versions[v].isActive && streams[turns[i].versions[v].id] == nil {
                turns[i].versions[v].status = .cancelled
                turns[i].versions[v].errorMessage = "Stopped when the app closed."
            }
        }
        clearComposerContext()
    }

    var currentSession: Session? {
        guard let currentSessionId else { return nil }
        return sessions.first { $0.id == currentSessionId }
    }

    /// An image is being made in this chat: its next message may refer to it, so sending waits for it.
    var isGeneratingHere: Bool {
        guard let currentSessionId else { return false }
        return activeRuns.values.contains(currentSessionId)
    }

    /// The other chat whose image is being made, if any; a message sent here is planned when it is done.
    var generatingElsewhere: Session? {
        guard let id = activeRuns.values.first(where: { $0 != currentSessionId }) else { return nil }
        return sessions.first { $0.id == id }
    }

    /// Stops the image being made in this chat (the send button turns into Stop).
    func stopGeneratingHere() {
        guard let currentSessionId else { return }
        for (versionId, sessionId) in activeRuns where sessionId == currentSessionId {
            if let turn = turns.first(where: { $0.versions.contains { $0.id == versionId } }) { cancel(turn.id) }
        }
    }

    func delete(session id: String) {
        guard let index = sessions.firstIndex(where: { $0.id == id }) else { return }
        let deletingCurrent = currentSessionId == id
        let turnsToCancel = deletingCurrent ? turns : []

        do {
            try store?.delete(session: id)
        } catch {
            show("Couldn't delete this chat: \(error.localizedDescription)")
            return
        }

        for turn in turnsToCancel where turn.current?.isActive == true {
            cancel(turn.id)
        }
        do {
            try editorStore.delete(sessionId: id)
        } catch {
            show("Chat deleted, but its editor drafts couldn't be removed.")
        }
        sessions.removeAll { $0.id == id }
        guard deletingCurrent else { return }

        if sessions.isEmpty {
            newSession()
        } else {
            select(sessions[min(index, sessions.count - 1)].id)
        }
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

    var needsRouteChoice: Bool { pendingRoute != nil }

    var pendingRouteChoices: [String] {
        var choices = defaultModel(for: "image.generate") == nil ? [] : ["image.generate"]
        if pendingRoute?.latestResultId != nil || !(pendingRoute?.attachmentInputs.isEmpty ?? true) {
            for operation in ["image.edit", "image.upscale", "image.restore", "image.enhance"]
                where defaultModel(for: operation) != nil {
                choices.append(operation)
            }
        }
        return choices
    }

    static func operationLabel(_ operation: String) -> String {
        switch operation {
        case "image.generate": "Create new"
        case "image.edit": "Edit current"
        case "image.upscale": "Upscale 2×"
        case "image.restore": "Restore"
        case "image.enhance": "Enhance"
        default: operation
        }
    }

    var routePreview: String {
        let message = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !message.isEmpty else {
            return chosenModel == nil ? "Model picked for you" : ""  // the model menu already names the choice
        }
        guard routeDecisionKey == routingKey(message: message), let decision = routeDecision else {
            return isResolvingRoute ? "Determining workflow…"
                : promptAssistantInstalled ? "Planned by the agent when you send" : "Ready to route"
        }
        switch decision.intent {
        case "image.generate":
            return defaultModel(for: .t2i).map { "Will generate · \($0.name)\(loadHint($0))" } ?? "No image model downloaded"
        case "image.edit":
            guard let m = defaultModel(for: .edit) else { return "No model that can edit is downloaded" }
            // The chosen model may not edit; say which one is used instead.
            let fallback = chosenModel.map { $0.supports(.edit) ? "" : " (\($0.name) can't edit)" } ?? ""
            let what = attachments.isEmpty ? latestImageTurn.map { "image \($0.number)" } ?? "the current image" : "the attached image\(attachments.count == 1 ? "" : "s")"
            return "Will edit \(what) · \(m.name)\(fallback)\(loadHint(m))"
        case "image.upscale", "image.restore", "image.enhance":
            guard let m = defaultModel(for: decision.intent) else {
                return "No installed executor for \(Self.operationLabel(decision.intent).lowercased())"
            }
            return "Will \(Self.operationLabel(decision.intent).lowercased()) · \(m.name)\(loadHint(m))"
        default:
            return "Choose whether to edit or create new"
        }
    }

    /// Refreshes the non-binding route hint. Structural cases resolve locally; only
    /// a continuation with a current image needs the configured decision provider.
    func scheduleRoutePreview() {
        // While a sent message is held (the agent's question, or a workflow to choose), what the user types
        // is the answer or the next message, and must not drop the held one.
        if pendingRoute != nil { return }
        routeTask?.cancel()
        pendingRoute = nil
        pendingAsk = nil
        let message = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        if promptSuggestion?.user != message {
            promptSuggestion = nil
            promptSuggestionApplied = false
        }
        let key = routingKey(message: message)
        guard !message.isEmpty else {
            routeDecision = nil
            routeDecisionKey = nil
            routeRequest = nil
            isResolvingRoute = false
            return
        }
        if let decision = structuralDecision() {
            routeDecision = decision
            routeDecisionKey = key
            routeRequest = nil
            isResolvingRoute = false
            return
        }
        // With the agent, a sent message is planned by it; a preview would only spend a router call.
        if promptAssistantInstalled {
            routeDecision = nil
            routeDecisionKey = nil
            routeRequest = nil
            isResolvingRoute = false
            return
        }
        routeDecision = nil
        routeDecisionKey = nil
        isResolvingRoute = true
        let currentResultId = latestImageTurn?.current?.resultId
        let currentPrompt = latestImageTurn?.current?.prompt
        let attachmentIds = attachments.map(\.id)
        let request = RouteRequestIdentity(context: routingContext(message: message))
        routeRequest = request
        routeTask = _Concurrency.Task { [weak self] in
            do {
                try await _Concurrency.Task.sleep(for: .milliseconds(300))
                guard let self, !_Concurrency.Task.isCancelled,
                      self.routeRequest?.accepts(
                        context: self.routingContext(message: message), requestID: request.id) == true else { return }
                guard let client = self.client else {
                    self.routeDecision = Self.ambiguousDecision(provider: "unavailable")
                    self.routeDecisionKey = key
                    self.isResolvingRoute = false
                    return
                }
                let decision = try await client.routeWorkflow(
                    message: message, currentResultId: currentResultId,
                    currentPrompt: currentPrompt, attachments: attachmentIds,
                    history: self.routingHistory, selectedModel: self.chosenModel?.id)
                guard !_Concurrency.Task.isCancelled,
                      self.routeRequest?.accepts(
                        context: self.routingContext(message: message), requestID: request.id) == true else { return }
                self.routeDecision = decision
                self.routeDecisionKey = key
                self.isResolvingRoute = false
            } catch is CancellationError {
            } catch {
                guard let self, self.routeRequest?.accepts(
                    context: self.routingContext(message: message), requestID: request.id) == true else { return }
                self.routeDecision = Self.ambiguousDecision(provider: "unavailable")
                self.routeDecisionKey = key
                self.isResolvingRoute = false
            }
        }
    }

    private func routingKey(message: String) -> String {
        routingContext(message: message).key
    }

    private func routingContext(message: String) -> RoutingContext {
        RoutingContext(
            message: message, currentResultId: latestImageTurn?.current?.resultId,
            attachmentIds: attachments.map(\.id), history: routingHistory,
            selectedModel: chosenModel?.id)
    }

    /// Keep routing context useful and bounded. The first few turns preserve
    /// references such as "the first image" while the recent suffix carries
    /// the active conversational intent.
    private var routingHistory: [RoutingHistoryItem] {
        let history = turns.map { turn in
            let version = turn.current
            return RoutingHistoryItem(
                sequence: turn.index, kind: turn.kind.rawValue, text: turn.text,
                imageNumber: turn.kind == .image ? turn.number : nil,
                operation: version?.operation, resultId: version?.resultId,
                inputIds: version?.mediaInputs?.map(\.id)
                    ?? turn.attachmentInputs?.map(\.id)
                    ?? version?.sourceIds
                    ?? turn.attachments)
        }
        guard history.count > 24 else { return history }
        return Array(history.prefix(4)) + Array(history.suffix(20))
    }

    private func structuralDecision() -> WorkflowDecision? {
        let count = attachments.count
        let hasCurrent = latestImageTurn != nil
        let operations = capabilityCatalog?.operations.compactMap { operation, capabilities -> String? in
            let available = capabilities.contains { capability in
                guard capability.downloaded else { return false }
                if let chosen = chosenModel {
                    guard capability.model == chosen.id else { return false }
                } else {
                    guard capability.fits_this_mac else { return false }
                }
                if let constraint = capability.inputs["image"] {
                    let inputs = count > 0 ? count : (hasCurrent ? 1 : 0)
                    return constraint.min <= inputs && inputs <= constraint.max
                }
                return count == 0
            }
            return available ? operation : nil
        } ?? []
        guard operations.count == 1, let operation = operations.first else { return nil }
        return WorkflowDecision(
            intent: operation, confidence: 1, probabilities: [operation: 1], provider: "deterministic")
    }

    private static func ambiguousDecision(provider: String) -> WorkflowDecision {
        WorkflowDecision(intent: "ambiguous", confidence: 0, probabilities: [:], provider: provider)
    }

    private func clearComposerContext() {
        attachments = []
        routeTask?.cancel()
        routeDecision = nil
        routeDecisionKey = nil
        routeRequest = nil
        pendingRoute = nil
        pendingAsk = nil
        isResolvingRoute = false
        promptSuggestion = nil
        promptSuggestionApplied = false
    }

    // MARK: sending

    func send(_ text: String? = nil) {
        let message = (text ?? draft).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !message.isEmpty else { return }
        // No image model yet (for example right after "Continue in Background"): nothing could make the image,
        // and the engine would hold the message with no workflow to choose. Keep it in the box and say why.
        if !models.isEmpty, !models.contains(where: \.downloaded) {
            if let loading = models.first(where: { $0.download.status == "downloading" }) {
                let percent = Int((loading.download.progress * 100).rounded())
                return show("Not sent yet: \(loading.name) is still downloading (\(percent)%). Send it when the download is done.")
            }
            onboardingStep = .setup
            showOnboarding = true
            return show("Not sent: download an image model first.")
        }
        if let planning = planningMessage { return show("Not sent yet: “\(planning)” is still being planned.") }
        guard !isGeneratingHere else { return show("Not sent: wait for the image to finish, or stop it.") }
        if let pending = pendingRoute, let question = pendingAsk, promptAssistantInstalled {
            if text == nil { draft = "" }
            return answer(pending, question: question, with: message)
        }
        if let held = pendingRoute?.message { return show("Choose what to do with “\(held)” first, or cancel it.") }
        let snapshot = PendingRoute(
            message: message, attachmentInputs: attachments.enumerated().map { $0.element.mediaInput(order: $0.offset) },
            latestResultId: latestImageTurn?.current?.resultId, latestTurnId: latestImageTurn?.id,
            promptSpec: selectedPromptSpec(for: message))
        let key = routingKey(message: message)
        // With the agent, every sent message is planned by it; the rules only check its answer (in the engine).
        let planned = promptAssistantInstalled
        if !planned, let decision = structuralDecision() {
            execute(snapshot, decision: decision)
            return
        }
        if !planned, routeDecisionKey == key, let decision = routeDecision {
            handle(snapshot, decision: decision)
            return
        }
        guard let client else { return show("The engine is still starting.") }
        // The composer's hint is not this message: typing, attaching or changing the model while it is
        // planned must not cancel it.
        routeTask?.cancel()
        routeRequest = nil
        isResolvingRoute = false
        planningMessage = message
        if text == nil { draft = "" }
        let waits = planned && generatingElsewhere != nil
        isWaitingForImage = waits
        let history = routingHistory
        let notes = attachmentNotes(snapshot.attachmentInputs)
        let currentPrompt = latestImageTurn?.current?.prompt
        planTask = _Concurrency.Task { [weak self] in
            defer { self?.isWaitingForImage = false }
            do {
                let decision = try await client.routeWorkflow(
                    message: message, currentResultId: snapshot.latestResultId, currentPrompt: currentPrompt,
                    attachments: snapshot.attachmentIds, attachmentNotes: notes, history: history,
                    selectedModel: self?.chosenModel?.id,
                    plan: planned, wait: waits)
                guard let self, !_Concurrency.Task.isCancelled else { return }
                self.planningMessage = nil
                self.routeDecision = decision
                self.routeDecisionKey = key
                self.handle(snapshot, decision: decision)
            } catch is CancellationError {
            } catch {
                guard let self, !_Concurrency.Task.isCancelled else { return }
                self.planningMessage = nil
                self.giveBack(message, because: "Could not plan it: \(error.localizedDescription)")
            }
        }
    }

    /// A sent message waiting on the user (the agent's question, or a workflow to choose), shown in the chat.
    var heldMessage: String? { pendingRoute?.message }

    /// A message that was not sent goes back to the composer, with the reason, so it is never lost.
    private func giveBack(_ message: String, because reason: String) {
        if draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            draft = message
            show("\(reason) Your message is back in the message box.")
        } else {
            show("\(reason) Not sent: “\(message)”.")
        }
    }

    /// Leaving the chat while a message is planned, or while the agent's question about it is open.
    private func abandonPlanning() {
        if let message = planningMessage {
            planTask?.cancel()
            planningMessage = nil
            isWaitingForImage = false
            giveBack(message, because: "Not sent: you changed chats while it was being planned.")
        } else if let held = heldMessage {
            giveBack(held, because: "Not sent: you changed chats before answering.")
        }
    }

    private func handle(_ pending: PendingRoute, decision: WorkflowDecision) {
        guard !decision.requiresConfirmation() else {
            pendingRoute = pending
            pendingAsk = decision.ask
            // The message is held with the question; the composer is free for the answer.
            if decision.ask != nil { draft = "" }
            return
        }
        execute(pending, decision: decision)
    }

    /// The user wrote again while the agent's question was open. The agent sees the original message and
    /// its question and decides whether this answers it or is a new request; both stay in the chat.
    private func answer(_ pending: PendingRoute, question: String, with reply: String) {
        guard let client else { return show("The engine is still starting.") }
        let history = routingHistory + [
            RoutingHistoryItem(sequence: turns.count, kind: TurnKind.user.rawValue, text: pending.message,
                               inputIds: pending.attachmentIds),
            RoutingHistoryItem(sequence: turns.count + 1, kind: TurnKind.reply.rawValue, text: question),
        ]
        planningMessage = reply
        let waits = generatingElsewhere != nil
        isWaitingForImage = waits
        planTask = _Concurrency.Task { [weak self] in
            defer { self?.isWaitingForImage = false }
            do {
                let decision = try await client.routeWorkflow(
                    message: reply, currentResultId: pending.latestResultId,
                    currentPrompt: self?.latestImageTurn?.current?.prompt, attachments: pending.attachmentIds,
                    attachmentNotes: self?.attachmentNotes(pending.attachmentInputs) ?? [:],
                    history: history, selectedModel: self?.chosenModel?.id, plan: true, wait: waits)
                guard let self, !_Concurrency.Task.isCancelled else { return }
                self.planningMessage = nil
                guard self.pendingAsk == question else { return }
                let sessionId = self.ensureSession(title: pending.message)
                self.appendTurn(Turn(
                    sessionId: sessionId, index: self.turns.count, kind: .user, text: pending.message,
                    attachments: pending.attachmentIds, attachmentInputs: pending.attachmentInputs))
                self.appendTurn(Turn(sessionId: sessionId, index: self.turns.count, kind: .reply, text: question))
                self.pendingRoute = nil
                self.pendingAsk = nil
                var answering = pending
                answering.message = reply
                answering.promptSpec = self.selectedPromptSpec(for: reply)
                answering.showsAttachments = false
                self.handle(answering, decision: decision)
            } catch is CancellationError {
            } catch {
                guard let self, !_Concurrency.Task.isCancelled else { return }
                self.planningMessage = nil
                self.giveBack(reply, because: "Could not plan the answer: \(error.localizedDescription)")
            }
        }
    }

    /// Dev harness only: the state right after the agent asked `question` about `message`.
    /// Dev harness only: shows a message as being planned, as right after Send.
    func harnessPlanning(_ message: String?) { planningMessage = message }
    /// Dev harness: each change presses the first-run setup's Download button.
    var harnessOnboardingDownload = 0

    func harnessAsk(message: String, question: String) {
        pendingRoute = PendingRoute(
            message: message, attachmentInputs: [], latestResultId: latestImageTurn?.current?.resultId,
            latestTurnId: latestImageTurn?.id, promptSpec: selectedPromptSpec(for: message))
        pendingAsk = question
        draft = ""
    }

    func choosePendingRoute(_ operation: String) {
        guard let pending = pendingRoute else { return }
        pendingRoute = nil
        pendingAsk = nil
        execute(pending, decision: WorkflowDecision(
            intent: operation, confidence: 1, probabilities: [operation: 1], provider: "user"))
    }

    func cancelPendingRoute() {
        if let pending = pendingRoute, draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            draft = pending.message  // give the held message back
        }
        routeTask?.cancel()
        isResolvingRoute = false
        if planningMessage != nil {
            planTask?.cancel()
            planningMessage = nil
            isWaitingForImage = false
        }
        pendingRoute = nil
        pendingAsk = nil
    }

    func overrideRoute(_ operation: String) {
        let message = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !message.isEmpty else { return }
        routeTask?.cancel()
        routeRequest = nil
        pendingRoute = nil
        isResolvingRoute = false
        routeDecision = WorkflowDecision(
            intent: operation, confidence: 1, probabilities: [operation: 1], provider: "user")
        routeDecisionKey = routingKey(message: message)
    }

    private func selectedPromptSpec(for message: String) -> PromptSpec {
        if promptSuggestionApplied, let suggestion = promptSuggestion, suggestion.user == message {
            return PromptSpec(
                user: message, positive: suggestion.positive, negative: suggestion.negative,
                enhancementMode: .ask, compiler: suggestion.compiler)
        }
        return PromptSpec(user: message, enhancementMode: promptEnhancementMode == .auto ? .auto : .off)
    }

    func enhanceDraft() {
        let message = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !message.isEmpty, !isEnhancingPrompt, let client else { return }
        let routedOperation = routeDecision?.intent
        let operation = routedOperation != nil && routedOperation != "ambiguous"
            ? routedOperation! : (!attachments.isEmpty ? "image.edit" : "image.generate")
        // The images an edit would run on, so the prompt assistant can look at them.
        let imageIds = operation == "image.edit"
            ? (!attachments.isEmpty ? attachments.map(\.id)
               : routeDecision?.input_result_ids?.isEmpty == false ? routeDecision!.input_result_ids!
               : latestImageTurn?.current?.resultId.map { [$0] } ?? [])
            : []
        isEnhancingPrompt = true
        _Concurrency.Task {
            defer { isEnhancingPrompt = false }
            do {
                let suggestion = try await client.compilePrompt(
                    prompt: message, operation: operation, model: defaultModel(for: operation)?.id, imageIds: imageIds)
                guard draft.trimmingCharacters(in: .whitespacesAndNewlines) == message else { return }
                promptSuggestion = suggestion
                promptSuggestionApplied = false
            } catch {
                show("Prompt enhancement failed: \(error.localizedDescription)")
            }
        }
    }

    func applyPromptSuggestion() {
        guard promptSuggestion != nil else { return }
        promptSuggestionApplied = true
    }

    func dismissPromptSuggestion() {
        promptSuggestion = nil
        promptSuggestionApplied = false
    }

    private func execute(_ pending: PendingRoute, decision: WorkflowDecision) {
        var pending = pending
        let message = pending.message
        // The agent's prompt is used as written: it decides how much to expand the message (a detailed
        // prompt is kept as it is). It is marked with the agent's name so the job does not rewrite it again;
        // an accepted ✨ suggestion wins.
        if pending.promptSpec.compiler == "passthrough@1", let written = decision.prompt {
            pending.promptSpec = PromptSpec(
                user: message, positive: written, enhancementMode: pending.promptSpec.enhancement_mode,
                compiler: decision.provider)
        }
        let sessionId = ensureSession(title: message)
        appendTurn(Turn(
            sessionId: sessionId, index: turns.count, kind: .user, text: message,
            attachments: pending.showsAttachments ? pending.attachmentIds : [],
            attachmentInputs: pending.showsAttachments ? pending.attachmentInputs : nil))
        draft = ""

        switch decision.intent {
        case "image.generate":
            guard let model = model(for: "image.generate", decision: decision) else { return noModel(sessionId) }
            startImageTurn(sessionId: sessionId, prompt: message, model: model, task: .t2i, inputs: [], route: "Generated",
                           size: aspect.size, decision: decision, promptSpec: pending.promptSpec, operation: decision.intent)
        case "image.edit":
            guard let model = model(for: "image.edit", decision: decision) else { return noModel(sessionId) }
            let inputs = routedInputs(pending, decision: decision)
            guard !inputs.isEmpty else { return noModel(sessionId) }
            var route = "Edited from the attached image"
            if let source = inputs.first {
                if let attached = pending.attachmentInputs.first(where: { $0.id == source.id }) {
                    if let base = preparedFrom(attached) { route = "Edited from the prepared image \(base.number)" }
                } else if let t = turns.first(where: { $0.versions.contains { $0.resultId == source.id } }) {
                    let version = t.versions.firstIndex { $0.resultId == source.id } ?? t.activeVersion
                    route = "Edited from image \(t.number) · v\(version + 1)"
                }
            }
            if inputs.count > 1 { route += " + \(inputs.count - 1) reference\(inputs.count > 2 ? "s" : "")" }
            startImageTurn(sessionId: sessionId, prompt: message, model: model, task: .edit, inputs: inputs, route: route,
                           size: nil, decision: decision, promptSpec: pending.promptSpec, operation: decision.intent)
        default:
            let operation = decision.intent
            guard operation.hasPrefix("image."), let model = model(for: operation, decision: decision) else {
                appendTurn(Turn(sessionId: sessionId, index: turns.count, kind: .notice,
                                text: "That workflow isn't available yet."))
                break
            }
            let inputs = routedInputs(pending, decision: decision)
            guard !inputs.isEmpty else { return noModel(sessionId) }
            startImageTurn(
                sessionId: sessionId, prompt: message, model: model, task: .edit, inputs: inputs,
                route: Self.operationLabel(operation), size: nil, decision: decision,
                promptSpec: pending.promptSpec, operation: operation)
        }
        clearComposerContext()
        touchSession()
    }

    private func routedInputs(_ pending: PendingRoute, decision: WorkflowDecision) -> [MediaInput] {
        let selected = decision.input_result_ids ?? []
        // The agent may add chat images to the attachments ("redo the boat in the style of this painting");
        // the engine has already checked that its list keeps every attachment.
        if decision.provider.hasPrefix("agent:"), !selected.isEmpty {
            return selected.prefix(3).enumerated().map { index, id in
                var input = pending.attachmentInputs.first(where: { $0.id == id })
                    ?? MediaInput(id: id, role: .reference, order: index)
                input.role = index == 0 ? .source : .reference
                input.order = index
                return input
            }
        }
        if !pending.attachmentInputs.isEmpty { return pending.attachmentInputs }
        let ids = selected.isEmpty ? pending.latestResultId.map { [$0] } ?? [] : selected
        return ids.prefix(3).enumerated().map { index, id in
            MediaInput(id: id, role: index == 0 ? .source : .reference, order: index)
        }
    }

    private func noModel(_ sessionId: String) {
        appendTurn(Turn(sessionId: sessionId, index: turns.count, kind: .notice,
                        text: "No suitable model is downloaded yet. Open Models to download one."))
        onboardingStep = .setup
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
        sessionId: String, prompt: String, model: EngineModel, task: GenTask, inputs: [MediaInput], route: String, size: String?,
        decision: WorkflowDecision, promptSpec: PromptSpec, operation: String? = nil
    ) {
        let workflowRoute = WorkflowRoute(
            intent: decision.intent, confidence: decision.confidence, router: decision.provider,
            probabilities: decision.probabilities, modelVersion: decision.model_version,
            latencyMS: decision.latency_ms, memoryMB: decision.memory_mb)
        let version = ImageVersion(
            prompt: prompt, model: model.id, task: task,
            operation: operation ?? (task == .edit ? "image.edit" : "image.generate"),
            size: size ?? "auto", sourceIds: inputs.map(\.id),
            workflowRoute: workflowRoute, mediaInputs: inputs, workflowPrompt: promptSpec)
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
                    sourceIds: version.sourceIds, mediaInputs: version.mediaInputs,
                    force: force, route: version.workflowRoute, promptSpec: version.workflowPrompt,
                    operation: version.operation)
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
            self?.activeRuns[versionId] = nil
            self?.persistTurn(turnId)
        }
        streams[versionId] = task
        if let currentSessionId { activeRuns[versionId] = currentSessionId }
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
            case "rewriting":
                v.phase = v.task == .edit ? "Planning the edit" : "Writing a detailed prompt"
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
        case "prompt_assistant": return "prompt assistant"
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
        if let t = turns.firstIndex(where: { $0.id == turnId }),
           let v = turns[t].versions.firstIndex(where: { $0.id == versionId }) {
            change(&turns[t].versions[v])
        } else if var turn = offscreenTurns[turnId], let v = turn.versions.firstIndex(where: { $0.id == versionId }) {
            let wasActive = turn.versions[v].isActive
            change(&turn.versions[v])
            offscreenTurns[turnId] = turn
            if wasActive != turn.versions[v].isActive { try? store?.save(turn) }
        }
    }

    private func persistTurn(_ turnId: String) {
        if let i = turns.firstIndex(where: { $0.id == turnId }) {
            persist(i)
        } else if let turn = offscreenTurns[turnId] {
            try? store?.save(turn)
            if !turn.versions.contains(where: \.isActive) { offscreenTurns[turnId] = nil }
        }
    }

    /// Before another chat is shown, turns with a job in flight move aside so their updates are not lost.
    private func keepActiveTurnsOffscreen() {
        let running = Set(streams.keys)
        for turn in turns where turn.versions.contains(where: { running.contains($0.id) }) {
            offscreenTurns[turn.id] = turn
        }
    }

    // MARK: result actions

    func pickVersion(_ turnId: String, _ index: Int) {
        guard let t = turns.firstIndex(where: { $0.id == turnId }) else { return }
        turns[t].activeVersion = index
        persist(t)
    }

    func variation(_ turnId: String) {
        guard let t = turns.firstIndex(where: { $0.id == turnId }), let base = turns[t].current else { return }
        var v = ImageVersion(prompt: base.prompt, model: base.model, task: base.task, operation: base.operation, size: base.size,
                             sourceIds: base.sourceIds, label: "Variation", workflowRoute: base.workflowRoute,
                             mediaInputs: base.mediaInputs, workflowPrompt: base.workflowPrompt)
        v.seed = nil
        turns[t].versions.append(v)
        turns[t].activeVersion = turns[t].versions.count - 1
        persist(t)
        run(turnId: turnId, versionId: v.id)
    }

    /// The image a message produced: the first image turn after it, before the next message.
    func imageTurn(answering userTurnId: String) -> Turn? {
        guard let i = turns.firstIndex(where: { $0.id == userTurnId }) else { return nil }
        for turn in turns[(i + 1)...] {
            if turn.kind == .user { return nil }
            if turn.kind == .image { return turn }
        }
        return nil
    }

    /// Replaces the image a message produced with a run of the edited prompt; the old result and its
    /// Variations are dropped. Model, inputs, size and seed stay the same, so only the prompt changes.
    func editPrompt(_ userTurnId: String, to text: String) {
        let prompt = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty, let u = turns.firstIndex(where: { $0.id == userTurnId }),
              let target = imageTurn(answering: userTurnId),
              let t = turns.firstIndex(where: { $0.id == target.id }), let base = turns[t].current,
              !base.isActive, prompt != base.prompt else { return }
        // A prompt suggestion accepted for the old text does not describe the new one.
        let spec = PromptSpec(user: prompt, enhancementMode: base.workflowPrompt?.enhancement_mode == .auto ? .auto : .off)
        var v = ImageVersion(prompt: prompt, model: base.model, task: base.task, operation: base.operation, size: base.size,
                             sourceIds: base.sourceIds, workflowRoute: base.workflowRoute,
                             mediaInputs: base.mediaInputs, workflowPrompt: spec)
        v.seed = base.seed
        turns[u].text = prompt
        turns[t].text = prompt
        turns[t].versions = [v]
        turns[t].activeVersion = 0
        persist(u)
        persist(t)
        run(turnId: target.id, versionId: v.id)
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

    func openEditor(_ turnId: String) {
        guard let sessionId = currentSessionId,
              let turn = turns.first(where: { $0.id == turnId }),
              let resultId = turn.current?.resultId,
              let image = images.image(resultId) else { return }
        var rect = CGRect(origin: .zero, size: image.size)
        guard let cgImage = image.cgImage(forProposedRect: &rect, context: nil, hints: nil) else {
            return show("Couldn't open this image for editing.")
        }
        editorDocument = editorStore.load(sessionId: sessionId, resultId: resultId)
            ?? EditorDocument(baseResultId: resultId, baseWidth: cgImage.width, baseHeight: cgImage.height)
        editorImage = image
        editorTurnNumber = turn.number
    }

    /// Dev harness only: the tool the editor opens on ("Crop", "Paint" or "Erase").
    var editorStartTool: String?

    func closeEditor() {
        guard !isExportingEditor else { return }
        editorDocument = nil
        editorImage = nil
    }

    func applyEditor(_ document: EditorDocument) {
        guard !isExportingEditor, let sessionId = currentSessionId, let image = editorImage,
              let client else { return }
        guard attachments.count < 3 else {
            return show("Remove an attachment before adding the prepared image.")
        }
        var rect = CGRect(origin: .zero, size: image.size)
        guard let source = image.cgImage(forProposedRect: &rect, context: nil, hints: nil) else {
            return show("Couldn't render this image.")
        }
        isExportingEditor = true
        _Concurrency.Task {
            let temporary = FileManager.default.temporaryDirectory
                .appendingPathComponent("dmlx-composite-\(UUID().uuidString).png")
            defer { try? FileManager.default.removeItem(at: temporary) }
            do {
                let png = try EditorRenderer.render(document, source: source)
                try png.write(to: temporary, options: .atomic)
                try editorStore.save(document, sessionId: sessionId)
                let resultId = try await client.upload(file: temporary)
                attachments.insert(Attachment(
                    id: resultId, label: "Prepared image \(editorTurnNumber)", role: .composite,
                    metadata: ["base_result_id": document.baseResultId, "editor_schema": String(document.schemaVersion)]), at: 0)
                editorDocument = nil
                editorImage = nil
                show("Prepared image attached as image 1")
            } catch {
                show("Couldn't prepare the image: \(error.localizedDescription)")
            }
            isExportingEditor = false
        }
    }

    func removeAttachment(_ id: String) {
        attachments.removeAll { $0.id == id }
    }

    func moveAttachment(_ id: String, before targetId: String) {
        guard id != targetId,
              let from = attachments.firstIndex(where: { $0.id == id }),
              let target = attachments.firstIndex(where: { $0.id == targetId }) else { return }
        let item = attachments.remove(at: from)
        let destination = from < target ? target - 1 : target
        attachments.insert(item, at: destination)
    }

    func attach(files: [URL]) {
        guard let client else { return show("The engine is still starting.") }
        _Concurrency.Task {
            let available = max(0, 3 - attachments.count)
            if available == 0 { return show("This model accepts up to 3 images.") }
            for url in files.prefix(available) {
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
        Store.dataDirectory().appendingPathComponent("results/\(resultId).png")
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

enum OnboardingStep: String {
    case welcome, setup
}
