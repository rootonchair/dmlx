import AppKit
import DMLXKit
import SwiftUI

struct SettingsView: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        @Bindable var app = app
        TabView(selection: $app.settingsTab) {
            GeneralSettings().tabItem { Label("General", systemImage: "gearshape") }.tag("general")
            ModelsSettings().tabItem { Label("Models", systemImage: "square.stack.3d.up") }.tag("models")
            MemorySettings().tabItem { Label("Memory", systemImage: "memorychip") }.tag("memory")
            IntelligenceSettings().tabItem { Label("Intelligence", systemImage: "sparkles") }.tag("intelligence")
            EngineSettings().tabItem { Label("Engine", systemImage: "gearshape.2") }.tag("engine")
        }
        .padding(16)
    }
}

struct GeneralSettings: View {
    @Environment(AppModel.self) private var model
    @AppStorage(AppAppearance.key) private var appearance = AppAppearance.system.rawValue

    var body: some View {
        Form {
            Picker("Appearance", selection: $appearance) {
                ForEach(AppAppearance.allCases) { Text($0.title).tag($0.rawValue) }
            }
            .pickerStyle(.segmented)
            .fixedSize()
            Text("System matches the appearance you set in macOS.")
                .font(.caption)
                .foregroundStyle(.secondary)
            Section {
                LabeledContent("Uninstall") {
                    Button("Uninstall dMLX…", role: .destructive) { model.uninstall() }
                }
                Text("Deletes all of dMLX's models, chats, images and settings, then moves dMLX to the Trash. "
                    + "Dragging dMLX to the Trash keeps the models on your Mac.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .onChange(of: appearance) { _, value in (AppAppearance(rawValue: value) ?? .system).apply() }
    }
}

/// The models the agent can run on: pick one, download it, see which is ready.
struct PromptAssistantRow: View {
    @Environment(AppModel.self) private var app
    @State private var status: PromptAssistantStatus?
    @State private var error: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let models = status?.models, !models.isEmpty {
                ForEach(models) { model in row(model) }
            } else {
                Text(error ?? (app.isEngineRunning ? "Checking…" : "Engine is not running"))
                    .font(.caption).foregroundStyle(.secondary)
            }
            if let error, status?.models != nil {
                Text(error).font(.caption).foregroundStyle(.red)
            }
        }
        .task(id: app.isEngineRunning) { await poll() }
    }

    private func row(_ model: PromptAssistantStatus.Model) -> some View {
        let selected = model.id == status?.selected
        return HStack(alignment: .firstTextBaseline) {
            Image(systemName: selected ? "largecircle.fill.circle" : "circle")
                .foregroundStyle(selected ? Color.accentColor : .secondary)
            VStack(alignment: .leading, spacing: 2) {
                Text(model.name)
                Text(detail(model, selected: selected)).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            if model.isDownloading {
                ProgressView(value: model.download.progress).frame(width: 120)
            } else if !model.installed {
                Button("Download \(String(format: "%.1f", model.size_gb)) GB") { pull(model.id) }
                    .disabled(!app.isEngineRunning)
            }
        }
        .contentShape(Rectangle())
        .onTapGesture { if !selected { select(model.id) } }
    }

    private func detail(_ model: PromptAssistantStatus.Model, selected: Bool) -> String {
        var parts = [String(format: "%.1f GB", model.size_gb), model.license == "apache-2.0" ? "Apache-2.0" : model.license]
        if !model.note.isEmpty { parts.append(model.note) }
        if model.isDownloading { parts.insert("Downloading… \(Int(model.download.progress * 100))%", at: 0) }
        else if model.download.status == "error" { parts.insert("Download failed: \(model.download.error ?? "unknown error")", at: 0) }
        else if model.installed { parts.insert(selected ? "In use" : "Ready", at: 0) }
        else if selected { parts.insert("Selected · not downloaded", at: 0) }
        return parts.joined(separator: " · ")
    }

    private func pull(_ id: String) {
        _Concurrency.Task {
            do {
                guard let client = app.client else { return }
                status = try await client.pullPromptAssistant(model: id)
                error = nil
            } catch {
                self.error = error.localizedDescription
            }
        }
    }

    /// Picking a model that is not downloaded yet also starts its download.
    private func select(_ id: String) {
        _Concurrency.Task {
            do {
                guard let client = app.client else { return }
                var next = try await client.selectPromptAssistant(model: id)
                if next.models?.first(where: { $0.id == id })?.installed == false {
                    next = try await client.pullPromptAssistant(model: id)
                }
                status = next
                app.promptAssistantInstalled = next.installed
                error = nil
            } catch {
                self.error = error.localizedDescription
            }
        }
    }

    /// Refreshes while the settings are open; every second during a download.
    private func poll() async {
        while !_Concurrency.Task.isCancelled {
            if let client = app.client {
                status = try? await client.promptAssistant()
                if let status { app.promptAssistantInstalled = status.installed }
            }
            let downloading = status?.models?.contains(where: \.isDownloading) ?? status?.isDownloading ?? false
            try? await _Concurrency.Task.sleep(for: .seconds(downloading ? 1 : 5))
        }
    }
}

struct IntelligenceSettings: View {
    @Environment(AppModel.self) private var app
    @State private var routerKind: RouterProviderKind
    @State private var baseURL: String
    @State private var providerModel: String
    @State private var apiKey: String
    @State private var timeout: Double
    @State private var localModelPath: String
    @State private var localRevision: String
    @State private var status = ""
    @State private var promptBaseURL: String
    @State private var promptModel: String
    @State private var promptAPIKey: String
    @State private var promptTimeout: Double

    init() {
        let settings = RouterProviderSettings.load()
        _routerKind = State(initialValue: settings.kind)
        _baseURL = State(initialValue: settings.baseURL)
        _providerModel = State(initialValue: settings.model)
        _apiKey = State(initialValue: RouterSecretStore.load())
        _timeout = State(initialValue: settings.timeout)
        _localModelPath = State(initialValue: settings.localModelPath)
        _localRevision = State(initialValue: settings.localRevision)
        let prompt = PromptProviderSettings.load()
        _promptBaseURL = State(initialValue: prompt.baseURL)
        _promptModel = State(initialValue: prompt.model)
        _promptAPIKey = State(initialValue: PromptSecretStore.load())
        _promptTimeout = State(initialValue: prompt.timeout)
    }

    var body: some View {
        @Bindable var app = app
        Form {
            Section("Semantic workflow routing") {
                Picker("Provider", selection: $routerKind) {
                    Text("Ask me when ambiguous").tag(RouterProviderKind.disabled)
                    Text("OpenAI-compatible").tag(RouterProviderKind.openAICompatible)
                    Text("Local Qwen MLX").tag(RouterProviderKind.localLLM)
                }
                if routerKind == .openAICompatible {
                    TextField("Base URL", text: $baseURL, prompt: Text("http://localhost:1234/v1"))
                    TextField("Model ID", text: $providerModel, prompt: Text("small-router-model"))
                    SecureField("API key (optional)", text: $apiKey)
                    HStack {
                        Text("Timeout")
                        TextField("Seconds", value: $timeout, format: .number)
                            .frame(width: 80)
                        Text("seconds").foregroundStyle(.secondary)
                    }
                } else if routerKind == .localLLM {
                    TextField("Local checkpoint folder", text: $localModelPath, prompt: Text("/Users/me/models/router"))
                    TextField("Pinned revision or checksum", text: $localRevision)
                    Text("The checkpoint and optional runtime must already be installed. dMLX will not download them automatically.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Text("Ambiguous follow-ups include bounded thread text and image-result metadata. Image pixels stay on this Mac.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Section("Agent") {
                AgentWhereSettings()
                Text("The agent looks at your images, decides what each message asks for, and rewrites every edit into a detailed description, which image editors follow far better than a short command. On this Mac, larger models plan better but need more memory; the agent is loaded only while it works.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Prompt enhancement") {
                Picker("Mode", selection: $app.promptEnhancementMode) {
                    Text("Off").tag(PromptEnhancementMode.off)
                    Text("Ask — preview before use").tag(PromptEnhancementMode.ask)
                    Text("Auto — enhance every submitted prompt").tag(PromptEnhancementMode.auto)
                }
                TextField("Base URL", text: $promptBaseURL, prompt: Text("http://localhost:1234/v1"))
                TextField("Model ID", text: $promptModel, prompt: Text("prompt-model"))
                SecureField("API key (optional)", text: $promptAPIKey)
                HStack {
                    Text("Timeout")
                    TextField("Seconds", value: $promptTimeout, format: .number).frame(width: 80)
                    Text("seconds").foregroundStyle(.secondary)
                }
                Text("✨ and Auto use the prompt assistant when it is downloaded. Otherwise they use this provider, which receives prompt text only; images remain local.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            HStack {
                Button("Save and restart engine") { save() }
                Button("Test current connection") { test() }
                    .disabled(!app.isEngineRunning)
                if !status.isEmpty { Text(status).font(.caption).foregroundStyle(.secondary) }
            }
        }
        .formStyle(.grouped)
    }

    private func save() {
        do {
            let settings = RouterProviderSettings(
                kind: routerKind,
                baseURL: baseURL.trimmingCharacters(in: .whitespacesAndNewlines),
                model: providerModel.trimmingCharacters(in: .whitespacesAndNewlines), timeout: timeout,
                localModelPath: localModelPath.trimmingCharacters(in: .whitespacesAndNewlines),
                localRevision: localRevision.trimmingCharacters(in: .whitespacesAndNewlines))
            try settings.save(apiKey: apiKey)
            let prompt = PromptProviderSettings(
                baseURL: promptBaseURL.trimmingCharacters(in: .whitespacesAndNewlines),
                model: promptModel.trimmingCharacters(in: .whitespacesAndNewlines), timeout: promptTimeout)
            try prompt.save(apiKey: promptAPIKey)
            status = settings.kind == .disabled && prompt.baseURL.isEmpty ? "Intelligence providers disabled" : "Saved"
            app.engine.restart()
        } catch {
            status = error.localizedDescription
        }
    }

    private func test() {
        status = "Testing…"
        _Concurrency.Task {
            do {
                guard let client = app.client else { throw EngineClientError.notRunning }
                let result = try await client.testRouter()
                status = "Connected in \(String(format: "%.0f", result.latency_ms)) ms"
            } catch {
                status = error.localizedDescription
            }
        }
    }
}

/// Settings → Models: the model for each task, then one card per model (like `ollama list`: the versions you
/// have), with Download, progress and Remove per version. Other versions and models are added from the catalog.
struct ModelsSettings: View {
    @Environment(AppModel.self) private var model
    @State private var adding: AddModelRequest?

    /// Recommended first, then the models on this Mac, then the rest (built in, not downloaded).
    private var groups: [ModelGroup] {
        ModelCatalog.groups(models: model.models, suggestions: []).sorted { a, b in
            let rank = { (g: ModelGroup) in g.id == ModelHints.recommendedGroup ? 0 : g.isDownloaded ? 1 : 2 }
            return (rank(a), a.name) < (rank(b), b.name)
        }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                if model.hasModelChoice {
                    TaskModelChoices()
                        .padding(.bottom, 4)
                }
                HStack(alignment: .firstTextBaseline) {
                    Text("Your Models").font(.headline)
                        .accessibilityAddTraits(.isHeader)
                    Spacer()
                    Button("Add Model…") { adding = AddModelRequest() }
                        .disabled(!model.isEngineRunning)
                }
                if model.models.isEmpty {
                    Text("Waiting for the engine…").foregroundStyle(.secondary)
                }
                ForEach(groups) { group in
                    ModelGroupCard(group: group) { adding = AddModelRequest(group: group.id) }
                }
                Text("More models, such as FLUX.3 Klein, ERNIE Image, Krea 2 or Iris, are in Add Model.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 4)
            .padding(.bottom, 8)
        }
        .task { await model.refreshModels() }
        .sheet(item: $adding) { request in
            AddModelView(initialGroup: request.group).environment(model)
        }
        .onChange(of: model.harnessAddModel) { adding = model.harnessAddModel.map { AddModelRequest(group: $0.isEmpty ? nil : $0) } }
    }
}

/// Opens Add Model, at one model's versions or at the top.
struct AddModelRequest: Identifiable {
    var id = UUID()
    var group: String?
}

struct MemorySettings: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        Form {
            if let m = model.memory {
                LabeledContent("Memory tier", value: "\(m.ram_tier_gb) GB")
                LabeledContent("Available now", value: String(format: "%.1f GB", m.memory.system.available_gb))
                LabeledContent("dMLX may use", value: String(format: "%.1f GB", max(m.memory.headroom_gb, 0)))
                LabeledContent("Kept free for macOS", value: String(format: "%.1f GB", m.memory.reserve_gb))
                LabeledContent("Memory pressure", value: m.memory.system.pressure.capitalized)
                LabeledContent("Swap in use", value: String(format: "%.1f GB", m.memory.system.swap_used_gb))
                if m.memory.degraded {
                    Text("Recently under memory pressure: using low-memory mode for a few minutes.")
                        .font(.caption).foregroundStyle(.orange)
                }
                Section("In memory") {
                    if m.memory.resident.isEmpty {
                        Text("Nothing loaded").foregroundStyle(.secondary)
                    }
                    ForEach(m.memory.resident, id: \.self) { r in
                        LabeledContent("\(model.modelName(r.model)) · \(AppModel.componentName(r.component))",
                                       value: String(format: "%.2f GB", r.gb))
                    }
                }
            } else {
                Text("Waiting for the engine…").foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .task {
            while !_Concurrency.Task.isCancelled {
                await model.refreshMemory()
                try? await _Concurrency.Task.sleep(for: .seconds(1))
            }
        }
    }
}

struct EngineSettings: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(model.engineStatusText).font(.callout)
                Spacer()
                Button("Restart engine") { model.engine.restart() }
                Button("Show log file") { NSWorkspace.shared.activateFileViewerSelecting([model.engine.logFile]) }
            }
            ScrollView {
                Text(model.engine.logLines.suffix(300).joined(separator: "\n"))
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 8))
        }
    }
}

/// Settings → Intelligence → Agent: on this Mac (its models), or on an AI service the user has.
struct AgentWhereSettings: View {
    @Environment(AppModel.self) private var app
    @State private var useService = false
    @State private var service: AgentService = .anthropic
    @State private var model = AgentService.claudeModels[0].id
    @State private var address = ""
    @State private var apiKey = ""
    @State private var status = ""
    @State private var working = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Picker("Where the agent runs", selection: $useService) {
                Text("On this Mac").tag(false)
                Text("On your AI service").tag(true)
            }
            .pickerStyle(.segmented)

            if useService {
                AgentServiceForm(service: $service, model: $model, address: $address, apiKey: $apiKey)
            } else {
                PromptAssistantRow()
            }

            HStack {
                if useService || app.agentChoice.service != .local {
                    Button(useService ? "Use \(service.title)" : "Use the Agent on This Mac") { apply() }
                        .disabled(working || !app.isEngineRunning)
                }
                if working { ProgressView().controlSize(.small) }
                Text(currentLine).font(.caption).foregroundStyle(.secondary)
            }
            if !status.isEmpty { Text(status).font(.caption).foregroundStyle(status.hasPrefix("Could not") ? .red : .secondary) }
        }
        .onAppear {
            let choice = app.agentChoice
            useService = choice.service != .local
            if useService {
                service = choice.service
                model = choice.model
                address = choice.address
                apiKey = AgentKeyStore.load(choice.service)
            }
        }
    }

    private var currentLine: String {
        let choice = app.agentChoice
        return choice.service == .local ? "Now: on this Mac" : "Now: \(choice.service.title) · \(choice.model)"
    }

    private func apply() {
        working = true
        status = ""
        let choice = useService ? AgentChoice(service: service, model: model, address: address) : AgentChoice()
        _Concurrency.Task {
            defer { working = false }
            do {
                try await app.connectAgent(choice, apiKey: apiKey, test: useService)
                status = useService ? "Connected. The agent now plans on \(service.title)." : "The agent runs on this Mac again."
            } catch {
                status = "Could not use \(service.title): \(error.localizedDescription)"
            }
        }
    }
}
