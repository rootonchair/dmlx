import AppKit
import DMLXKit
import SwiftUI

struct SettingsView: View {
    var body: some View {
        TabView {
            ModelsSettings().tabItem { Label("Models", systemImage: "square.stack.3d.up") }
            MemorySettings().tabItem { Label("Memory", systemImage: "memorychip") }
            IntelligenceSettings().tabItem { Label("Intelligence", systemImage: "sparkles") }
            EngineSettings().tabItem { Label("Engine", systemImage: "gearshape.2") }
        }
        .padding(16)
    }
}

/// Status of the local prompt assistant, with its download.
struct PromptAssistantRow: View {
    @Environment(AppModel.self) private var app
    @State private var status: PromptAssistantStatus?
    @State private var error: String?

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Qwen3-VL-8B")
                Text(detail).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            if let status, status.isDownloading {
                ProgressView(value: status.download.progress).frame(width: 120)
            } else if let status, !status.installed {
                Button("Download \(String(format: "%.1f", status.size_gb)) GB") { pull() }
                    .disabled(!app.isEngineRunning)
            }
        }
        .task(id: app.isEngineRunning) { await poll() }
    }

    private var detail: String {
        if let error { return error }
        guard let status else { return app.isEngineRunning ? "Checking…" : "Engine is not running" }
        if status.installed { return "Ready · Apache-2.0 · loaded only while it writes, then freed" }
        if status.isDownloading { return "Downloading… \(Int(status.download.progress * 100))%" }
        if status.download.status == "error" { return "Download failed: \(status.download.error ?? "unknown error")" }
        return "Not downloaded · \(String(format: "%.1f", status.size_gb)) GB · Apache-2.0"
    }

    private func pull() {
        _Concurrency.Task {
            do {
                guard let client = app.client else { return }
                status = try await client.pullPromptAssistant()
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
            try? await _Concurrency.Task.sleep(for: .seconds(status?.isDownloading == true ? 1 : 5))
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
            Section("Prompt assistant") {
                PromptAssistantRow()
                Text("Runs on this Mac and looks at your images. Every edit is rewritten into a detailed description first, which instruction-based editors follow far better than a short command. Without it, edits run as written.")
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

struct ModelsSettings: View {
    @Environment(AppModel.self) private var model
    @State private var adding = false

    var body: some View {
        @Bindable var model = model
        ScrollView {
            VStack(spacing: 10) {
                if model.choosableModels.count > 1 {
                    Picker("Model", selection: $model.modelChoice) {
                        Text("Auto").tag("auto")
                        ForEach(model.choosableModels) { m in Text(m.name).tag(m.id) }
                    }
                    .help("Also in the message box. Auto uses one model for new images and edits, which keeps follow-ups fastest.")
                    .padding(.bottom, 6)
                }
                ForEach(model.models) { m in
                    ModelRow(model: m, removable: true)
                }
                HStack {
                    Text("Add FLUX.2 Klein, Z-Image, or Qwen Image 2.1 Base models from Hugging Face.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Add Model…") { adding = true }
                        .disabled(!model.isEngineRunning)
                }
                .padding(.top, 4)
                if model.models.isEmpty {
                    Text("Waiting for the engine…").foregroundStyle(.secondary)
                }
            }
        }
        .task { await model.refreshModels() }
        .sheet(isPresented: $adding) {
            AddModelView().environment(model)
        }
    }
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
