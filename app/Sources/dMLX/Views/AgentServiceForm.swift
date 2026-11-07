import DMLXKit
import Foundation
import Security
import SwiftUI

/// Where the planning agent runs: a model on this Mac, or an AI service the user already has.
enum AgentService: String, CaseIterable, Identifiable, Codable {
    case local
    case anthropic
    case openai
    case ollama
    case other = "openai-compatible"

    var id: String { rawValue }
    static let services: [AgentService] = [.anthropic, .openai, .ollama, .other]

    var title: String {
        switch self {
        case .local: "On this Mac"
        case .anthropic: "Claude"
        case .openai: "ChatGPT"
        case .ollama: "Ollama"
        case .other: "Other service"
        }
    }

    /// Who receives the chat and its images.
    var company: String {
        switch self {
        case .local: "this Mac"
        case .anthropic: "Anthropic"
        case .openai: "OpenAI"
        case .ollama: "Ollama on this Mac"
        case .other: "that service"
        }
    }

    var hint: String {
        switch self {
        case .local: "Private and free. Downloads a model that runs on this Mac."
        case .anthropic: "Claude, by Anthropic. Very good at understanding what you mean. Needs an API key; Anthropic charges your account for each message."
        case .openai: "OpenAI's models, the ones behind ChatGPT. Needs an API key; OpenAI charges your account for each message. A ChatGPT subscription is not an API key."
        case .ollama: "Free. Uses a model you run with Ollama on this Mac. Pick one that can see images, such as Gemma or Qwen-VL."
        case .other: "Any service that works like the OpenAI API, such as LM Studio or OpenRouter. Enter its address, and a key if it needs one."
        }
    }

    var needsKey: Bool { self == .anthropic || self == .openai }
    var takesKey: Bool { needsKey || self == .other }
    var takesAddress: Bool { self == .ollama || self == .other }
    var defaultAddress: String { self == .ollama ? "http://localhost:11434/v1" : "" }
    /// Paid by the user's own account.
    var charges: Bool { self == .anthropic || self == .openai }

    var keyPage: URL? {
        switch self {
        case .anthropic: URL(string: "https://console.anthropic.com/settings/keys")
        case .openai: URL(string: "https://platform.openai.com/api-keys")
        default: nil
        }
    }

    /// Shown for Claude before a key is entered (or if the list cannot load). With a key, the list comes from
    /// Anthropic, so new models appear without an app update.
    static let claudeModels: [AgentServiceModel] = [
        AgentServiceModel(id: "claude-opus-5-5", name: "Claude Opus 5.5", line: "opus", recommended: true),
        AgentServiceModel(id: "claude-sonnet-5-5", name: "Claude Sonnet 5.5", line: "sonnet", recommended: true),
        AgentServiceModel(id: "claude-haiku-5-5", name: "Claude Haiku 5.5", line: "haiku", recommended: true),
    ]

    /// A plain hint for a recommended model; nil when there is nothing simple to say.
    func hint(for model: AgentServiceModel, among recommended: [AgentServiceModel]) -> String? {
        guard model.recommended else { return nil }
        switch self {
        case .anthropic:
            switch model.line {
            case "opus": return "Most capable"
            case "sonnet": return "Faster, costs less"
            case "haiku": return "Fastest, costs the least"
            default: return nil
            }
        case .openai:
            if ["mini", "nano", "luna"].contains(where: model.id.contains) { return "Smaller, costs less" }
            return model.id == recommended.first?.id ? "Newest" : nil
        default:
            return nil
        }
    }
}

/// The user's choice, kept between launches. The API key is in the Keychain (AgentKeyStore), never here.
struct AgentChoice: Codable, Equatable {
    var service: AgentService = .local
    var model: String = ""
    var address: String = ""

    private static let key = "agentChoice"

    static func load() -> AgentChoice {
        guard let data = UserDefaults.standard.data(forKey: key),
              let choice = try? JSONDecoder().decode(AgentChoice.self, from: data) else { return AgentChoice() }
        return choice
    }

    func save() {
        if let data = try? JSONEncoder().encode(self) { UserDefaults.standard.set(data, forKey: Self.key) }
    }
}

/// API keys for agent services, one per service, in the login Keychain on this Mac.
enum AgentKeyStore {
    private static let service = "app.dmlx.agent-service"

    static func load(_ agent: AgentService) -> String {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: agent.rawValue,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess, let data = item as? Data else { return "" }
        return String(decoding: data, as: UTF8.self)
    }

    static func save(_ key: String, for agent: AgentService) {
        let identity: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: agent.rawValue,
        ]
        SecItemDelete(identity as CFDictionary)
        guard !key.isEmpty else { return }
        var item = identity
        item[kSecValueData as String] = Data(key.utf8)
        item[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        SecItemAdd(item as CFDictionary, nil)
    }
}

/// Choosing an AI service for the agent: the service, its address and key, and a model. Used in the
/// first-run setup and in Settings → Intelligence.
struct AgentServiceForm: View {
    @Environment(AppModel.self) private var app
    @Binding var service: AgentService
    @Binding var model: String
    @Binding var address: String
    @Binding var apiKey: String

    @State private var listed: [AgentServiceModel] = []
    @State private var listing = false
    @State private var listError: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Picker("Service", selection: $service) {
                ForEach(AgentService.services) { Text($0.title).tag($0) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()

            Text(service.hint)
                .font(.callout).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            if service.takesAddress {
                LabeledContent("Address") {
                    TextField("Address", text: $address, prompt: Text(service == .ollama ? service.defaultAddress : "https://…/v1"))
                        .textFieldStyle(.roundedBorder)
                        .labelsHidden()
                }
            }
            if service.takesKey {
                LabeledContent("API key") {
                    HStack {
                        SecureField("API key", text: $apiKey, prompt: Text(service.needsKey ? "Paste your key" : "Only if the service needs one"))
                            .textFieldStyle(.roundedBorder)
                            .labelsHidden()
                        if let page = service.keyPage {
                            Link("Get a key", destination: page).font(.callout)
                        }
                    }
                }
            }
            LabeledContent("Model") { modelPicker }

            Label(privacyNote, systemImage: "info.circle")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .onChange(of: service) {
            model = service == .anthropic ? AgentService.claudeModels[0].id : ""
            listError = nil
            address = service.defaultAddress
            apiKey = AgentKeyStore.load(service)
            listed = []
        }
        .task(id: "\(service.rawValue)|\(address)|\(apiKey.count)") { await listModels() }
    }

    /// What the menu offers: the service's own list, or for Claude the built-in one until a key is entered.
    private var offered: [AgentServiceModel] {
        if listed.isEmpty, service == .anthropic { return AgentService.claudeModels }
        return listed
    }

    @ViewBuilder
    private var modelPicker: some View {
        HStack {
            if offered.isEmpty {
                TextField("Model", text: $model, prompt: Text(service == .ollama ? "gemma4" : "model name"))
                    .textFieldStyle(.roundedBorder)
                    .labelsHidden()
            } else {
                let recommended = offered.filter(\.recommended)
                let others = offered.filter { !$0.recommended }
                Picker("Model", selection: $model) {
                    if model.isEmpty || !offered.contains(where: { $0.id == model }) { Text("Choose a model").tag(model) }
                    if recommended.isEmpty {
                        ForEach(others) { Text($0.name).tag($0.id) }
                    } else {
                        Section("Recommended") {
                            ForEach(recommended) { m in
                                Text(service.hint(for: m, among: recommended).map { "\(m.name) — \($0)" } ?? m.name).tag(m.id)
                            }
                        }
                        if !others.isEmpty {
                            Section("Other models") { ForEach(others) { Text($0.name).tag($0.id) } }
                        }
                    }
                }
                .labelsHidden()
            }
            if listing { ProgressView().controlSize(.small) }
        }
        if let listError, listed.isEmpty, !(service == .anthropic && apiKey.isEmpty) {
            Text(listError).font(.caption).foregroundStyle(.secondary)
        }
    }

    private var privacyNote: String {
        let sent = "To plan each message, your messages and the images in the chat are sent to \(service.company)."
        let local = " Images are still made on this Mac."
        return sent + local + (service.charges ? " \(service.company) charges your account for this." : "")
    }

    /// The service's own model list, once it can be reached (Claude and ChatGPT need the key first).
    private func listModels() async {
        guard let client = app.client else { return }
        if service.needsKey && apiKey.isEmpty { listed = []; return }
        let base = address.isEmpty ? service.defaultAddress : address
        if service.takesAddress && base.isEmpty { return }
        try? await _Concurrency.Task.sleep(for: .milliseconds(400))  // typing
        guard !_Concurrency.Task.isCancelled else { return }
        listing = true
        defer { listing = false }
        do {
            // The engine leaves out models that cannot chat or see images, and puts recommended ones first.
            let models = try await client.agentServiceModels(kind: service.rawValue, baseURL: base, apiKey: apiKey)
            listed = models
            listError = nil
            if !models.contains(where: { $0.id == model }), let first = models.first { model = first.id }
        } catch {
            listed = []
            listError = error.localizedDescription
        }
    }
}

extension AppModel {
    /// Connects the agent to the chosen service (or back to this Mac) and remembers the choice.
    /// With `test`, a wrong key or address throws here, before anything is saved.
    func connectAgent(_ choice: AgentChoice, apiKey: String, test: Bool) async throws {
        guard let client else { throw EngineClientError.notRunning }
        let status: PromptAssistantStatus
        if choice.service == .local {
            status = try await client.setAgentProvider(kind: "local")
        } else {
            let address = choice.address.isEmpty ? choice.service.defaultAddress : choice.address
            status = try await client.setAgentProvider(
                kind: choice.service.rawValue, model: choice.model.trimmingCharacters(in: .whitespaces),
                baseURL: address, apiKey: apiKey.trimmingCharacters(in: .whitespacesAndNewlines), test: test)
            AgentKeyStore.save(apiKey.trimmingCharacters(in: .whitespacesAndNewlines), for: choice.service)
        }
        choice.save()
        agentChoice = choice
        promptAssistantInstalled = status.installed
    }

    /// At engine start (and restart): run the agent where the user chose. A failure here is not shown;
    /// the first planned message reports it.
    func restoreAgentChoice() async {
        let choice = AgentChoice.load()
        agentChoice = choice
        guard choice.service != .local else { return }
        try? await connectAgent(choice, apiKey: AgentKeyStore.load(choice.service), test: false)
    }
}
