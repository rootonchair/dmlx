import DMLXKit
import SwiftUI

/// First-run setup, in two steps: what dMLX is, then a recommended image model and agent for this Mac.
/// It also opens when a message needs a model that is not downloaded.
struct OnboardingView: View {
    @Environment(AppModel.self) private var app
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        Group {
            switch app.onboardingStep {
            case .welcome: OnboardingWelcome(next: { app.onboardingStep = .setup }, later: finish)
            case .setup: OnboardingSetup(back: { app.onboardingStep = .welcome }, finish: finish)
            }
        }
        .padding(28)
        .frame(width: 600)
    }

    private func finish() {
        UserDefaults.standard.set(true, forKey: AppModel.onboardingCompletedKey)
        dismiss()
    }
}

/// What a Mac has: memory and the free space on the disk that holds the Hugging Face cache.
private struct ThisMac {
    var memoryGB: Int
    var freeGB: Double?

    @MainActor init(_ app: AppModel) {
        memoryGB = app.memory?.ram_tier_gb ?? Int((Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824).rounded())
        let home = URL(fileURLWithPath: NSHomeDirectory())
        let bytes = (try? home.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
            .volumeAvailableCapacityForImportantUsage
        freeGB = bytes.map { Double($0) / 1e9 }
    }
}

private struct OnboardingWelcome: View {
    @Environment(AppModel.self) private var app
    let next: () -> Void
    let later: () -> Void

    var body: some View {
        let mac = ThisMac(app)
        VStack(alignment: .leading, spacing: 22) {
            HStack(spacing: 14) {
                DMLXMark(size: 52)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Welcome to dMLX")
                        .font(.system(size: 28, weight: .regular, design: .serif))
                    Text("Make and edit images with chat, fully on this Mac.")
                        .foregroundStyle(.secondary)
                }
            }

            VStack(alignment: .leading, spacing: 14) {
                point("text.bubble", "Describe an image, then say what to change.",
                      "Write \u{201C}make the boat red\u{201D} or \u{201C}remove the mug\u{201D}. Every result is kept as a version.")
                point("lock", "Private and offline.",
                      "After the models download, everything runs on this Mac. No account, and nothing is uploaded.")
                point("memorychip", "Your Mac stays usable.",
                      "dMLX checks free memory before each image and stops a job before macOS starts to slow down.")
            }

            HStack(spacing: 8) {
                Image(systemName: mac.memoryGB >= 16 ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                    .foregroundStyle(mac.memoryGB >= 16 ? Color.green : Color.orange)
                Text(macLine(mac)).font(.callout)
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 10))

            HStack {
                Button("Set Up Later", action: later)
                    .keyboardShortcut(.cancelAction)
                Spacer()
                Button("Continue", action: next)
                    .keyboardShortcut(.defaultAction)
                    .buttonStyle(.borderedProminent)
            }
        }
    }

    private func point(_ symbol: String, _ title: String, _ detail: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Image(systemName: symbol)
                .foregroundStyle(Theme.accent)
                .frame(width: 20)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.headline)
                Text(detail).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func macLine(_ mac: ThisMac) -> String {
        var line = "This Mac has \(mac.memoryGB) GB of memory"
        if let free = mac.freeGB { line += " and \(Int(free)) GB of free disk space" }
        return line + (mac.memoryGB >= 16 ? "." : ". dMLX is made for 16 GB or more, so large models may not run.")
    }
}

private struct OnboardingSetup: View {
    @Environment(AppModel.self) private var app
    let back: () -> Void
    let finish: () -> Void

    @State private var imageId: String?
    @State private var agentId: String?
    @State private var useAgent = true
    @State private var agent: PromptAssistantStatus?
    @State private var started = false
    @State private var error: String?
    @State private var licenseModel: EngineModel?
    // The agent on an AI service instead of this Mac (AgentServiceForm.swift).
    @State private var useService = false
    @State private var service: AgentService = .anthropic
    @State private var serviceModel = AgentService.claudeModels[0].id
    @State private var serviceAddress = ""
    @State private var serviceKey = ""
    @State private var connecting = false
    @State private var connected = false

    /// Image models this Mac can run, recommended one first. Only models that both make and edit images:
    /// the chat is built on "now change this", so a new-images-only model (Z-Image Turbo) would leave every
    /// edit with nothing to run it. Those stay available in Settings → Models.
    private var imageChoices: [EngineModel] {
        let fitting = app.models.filter { $0.fits_this_mac && $0.supports(.t2i) && $0.supports(.edit) }
        let preferred = ["flux2-klein-4b"]
        return fitting.sorted { a, b in
            (preferred.firstIndex(of: a.id) ?? 99, a.name) < (preferred.firstIndex(of: b.id) ?? 99, b.name)
        }
    }
    private var image: EngineModel? {
        imageChoices.first { $0.id == imageId } ?? imageChoices.first(where: \.downloaded) ?? imageChoices.first
    }
    private var agentChoices: [PromptAssistantStatus.Model] { agent?.models ?? [] }
    private var agentModel: PromptAssistantStatus.Model? {
        agentChoices.first { $0.id == (agentId ?? agent?.selected) } ?? agentChoices.first
    }

    private var needsImage: Bool { !(image?.downloaded ?? true) }
    private var needsAgent: Bool { !useService && useAgent && !(agentModel?.installed ?? true) }
    private var serviceReady: Bool {
        !serviceModel.trimmingCharacters(in: .whitespaces).isEmpty
            && (!service.needsKey || !serviceKey.isEmpty)
            && (!service.takesAddress || !(serviceAddress.isEmpty ? service.defaultAddress : serviceAddress).isEmpty)
    }
    private var downloadGB: Double {
        (needsImage ? image?.size_gb ?? 0 : 0) + (needsAgent ? agentModel?.size_gb ?? 0 : 0)
    }
    private var downloading: Bool {
        image?.download.status == "downloading" || (!useService && useAgent && agentModel?.isDownloading == true)
    }
    private var imageReady: Bool { image?.downloaded ?? false }

    /// The agent model suggested for this Mac's memory (as the engine's own default).
    private func recommendedAgent(_ mac: ThisMac) -> String {
        mac.memoryGB >= 24 ? "qwen3-vl-8b-instruct-4bit" : "ministral-3-3b-instruct-2512-4bit"
    }

    private var choicesLocked: Bool { started && downloading }

    @State private var choicesHeight: CGFloat = 480
    /// Room for the choices: the screen, less the title, the footer and the sheet's margins.
    private var maxChoicesHeight: CGFloat { max(300, (NSScreen.main?.visibleFrame.height ?? 800) - 320) }

    var body: some View {
        let mac = ThisMac(app)
        let enoughSpace = mac.freeGB.map { $0 > downloadGB + 2 } ?? true
        VStack(alignment: .leading, spacing: 22) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Choose your models")
                    .font(.system(size: 26, weight: .regular, design: .serif))
                Text("We picked what suits this Mac. You can change both later in Settings.")
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if !app.isEngineRunning || app.models.isEmpty {
                HStack {
                    ProgressView().controlSize(.small)
                    Text(app.engineStatusText).foregroundStyle(.secondary)
                }
            }

            // The choices scroll only when the sheet would not fit the screen (a 13-inch Mac, more models).
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    if let image {
                        VStack(alignment: .leading, spacing: 10) {
                            sectionHeader("Image model", "Makes the pictures, on this Mac.")
                            VStack(spacing: 6) {
                                ForEach(Array(imageChoices.enumerated()), id: \.element.id) { index, m in
                                    ChoiceRow(
                                        name: m.group_name ?? m.name, hint: ModelHints.image(m), size: gb(m.size_gb),
                                        recommended: index == 0 && imageChoices.count > 1,
                                        // No license badge: the license dialog before the download says it.
                                        tag: nil,
                                        state: .init(downloaded: m.downloaded, status: m.download.status,
                                                     progress: m.download.progress, error: m.download.error),
                                        selected: m.id == image.id
                                    ) { imageId = m.id }
                                    .disabled(choicesLocked)
                                }
                            }
                        }
                    }

                    VStack(alignment: .leading, spacing: 10) {
                        HStack(alignment: .firstTextBaseline) {
                            sectionHeader("Agent", "Reads your messages and images, and decides what to do.")
                            Spacer()
                            if !useService, let agentModel, !agentModel.installed || !useAgent {
                                Toggle("Download the agent", isOn: $useAgent)
                                    .toggleStyle(.switch)
                                    .controlSize(.mini)
                                    .font(.callout)
                                    .disabled(choicesLocked)
                                    .help("Without the agent, dMLX routes messages with simple rules.")
                            }
                        }
                        HStack(spacing: 8) {
                            PlaceCard(symbol: "laptopcomputer", title: "On this Mac",
                                      detail: Text("Private and free"), selected: !useService) { useService = false }
                            PlaceCard(symbol: "cloud", title: "Your AI service",
                                      detail: serviceStatus, selected: useService) { useService = true }
                        }
                        .disabled(choicesLocked)

                        if useService {
                            AgentServiceForm(service: $service, model: $serviceModel, address: $serviceAddress, apiKey: $serviceKey)
                                .padding(14)
                                .background(ChoiceRow.fill, in: RoundedRectangle(cornerRadius: 10))
                                .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(.separator))
                                .onChange(of: "\(service.rawValue)|\(serviceModel)|\(serviceAddress)|\(serviceKey)") { connected = false }
                        } else {
                            let recommended = recommendedAgent(mac)
                            VStack(spacing: 6) {
                                ForEach(agentChoices.sorted { ($0.id == recommended ? 0 : 1) < ($1.id == recommended ? 0 : 1) }) { m in
                                    ChoiceRow(
                                        name: m.name, hint: ModelHints.agent(m), size: gb(m.size_gb),
                                        recommended: m.id == recommended && agentChoices.count > 1, tag: nil,
                                        state: .init(downloaded: m.installed, status: m.download.status,
                                                     progress: m.download.progress, error: m.download.error),
                                        selected: m.id == agentModel?.id
                                    ) { agentId = m.id }
                                    .disabled(!useAgent || choicesLocked)
                                }
                            }
                            .opacity(useAgent ? 1 : 0.5)
                        }
                    }
                }
                .onGeometryChange(for: CGFloat.self) { $0.size.height } action: { choicesHeight = $0 }
            }
            .scrollBounceBehavior(.basedOnSize)
            .frame(height: min(choicesHeight, maxChoicesHeight))

            footer(mac: mac, enoughSpace: enoughSpace)
        }
        .task { await pollAgent() }
        .onAppear { restoreChoice() }
        // Dev harness: the main button (Download, or Start Creating), and filling in an agent service.
        .onChange(of: app.harnessOnboardingDownload) { downloadGB > 0 ? proceed { requestDownloads() } : proceed { finish() } }
        .onChange(of: app.harnessAgent) {
            guard let choice = app.harnessAgent else { return }
            useService = true
            service = choice.service
            serviceModel = choice.model
            serviceAddress = choice.address
        }
        .sheet(item: $licenseModel) { m in
            ModelLicenseView(model: m) {
                ModelLicenseAcceptance.accept(m)
                licenseModel = nil
                startDownloads()
            }
        }
    }

    /// The AI service card's second line: what it is, or how the connection check went.
    private var serviceStatus: Text {
        if connecting { return Text("Connecting…") }
        if connected && useService {
            return Text("\(Image(systemName: "checkmark.circle.fill")) Connected").foregroundStyle(.green)
        }
        return Text("Claude, ChatGPT, Ollama…")
    }

    private func sectionHeader(_ title: String, _ detail: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(title).font(.headline)
            Text(detail).font(.callout).foregroundStyle(.secondary)
        }
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isHeader)
    }

    @ViewBuilder
    private func footer(mac: ThisMac, enoughSpace: Bool) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            if let error {
                Text(error).font(.callout).foregroundStyle(.red)
            }
            if downloadGB > 0 && !started {
                Text(spaceLine(mac: mac, enoughSpace: enoughSpace))
                    .font(.callout)
                    .foregroundStyle(enoughSpace ? Color.secondary : Color.red)
            } else if downloading {
                Text(imageReady
                     ? "The image model is ready. You can start now; the agent keeps downloading."
                     : "You can close this window. The downloads continue, and you can start when the image model is ready.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack {
                Button("Back", action: back).disabled(started && downloading)
                Spacer()
                if downloadGB > 0 && !(started && downloading) {
                    Button("Set Up Later", action: finish)
                    Button("Download \(gb(downloadGB))") { proceed { requestDownloads() } }
                        .keyboardShortcut(.defaultAction)
                        .buttonStyle(.borderedProminent)
                        .disabled(!enoughSpace || image == nil || !app.isEngineRunning || connecting || (useService && !serviceReady))
                } else if !imageReady {
                    Button("Continue in Background", action: finish)
                        .keyboardShortcut(.defaultAction)
                } else {
                    Button("Start Creating") { proceed { finish() } }
                        .keyboardShortcut(.defaultAction)
                        .buttonStyle(.borderedProminent)
                        .disabled(connecting || (useService && !serviceReady))
                }
            }
        }
    }

    private func spaceLine(mac: ThisMac, enoughSpace: Bool) -> String {
        guard let free = mac.freeGB else { return "Total download: \(gb(downloadGB))." }
        let base = "Total download: \(gb(downloadGB)). This Mac has \(Int(free)) GB free."
        return enoughSpace ? base + " Models are saved in your Hugging Face cache." : base + " Free up some space, or pick smaller models."
    }

    /// With an AI service chosen, check it first (one small request): a wrong key or address shows here.
    /// Back on this Mac, the agent runs locally again.
    private func proceed(_ next: @escaping () -> Void) {
        error = nil
        rememberImageChoice()
        guard useService else {
            if app.agentChoice.service != .local {
                _Concurrency.Task { try? await app.connectAgent(AgentChoice(), apiKey: "", test: false) }
            }
            return next()
        }
        if connected { return next() }
        connecting = true
        let choice = AgentChoice(service: service, model: serviceModel, address: serviceAddress)
        _Concurrency.Task {
            defer { connecting = false }
            do {
                try await app.connectAgent(choice, apiKey: serviceKey, test: true)
                connected = true
                next()
            } catch {
                self.error = "Could not use \(service.title): \(error.localizedDescription)"
            }
        }
    }

    private func restoreChoice() {
        let choice = app.agentChoice
        guard choice.service != .local else { return }
        useService = true
        service = choice.service
        serviceModel = choice.model
        serviceAddress = choice.address
        serviceKey = AgentKeyStore.load(choice.service)
        connected = app.promptAssistantInstalled
    }

    /// The picked image model becomes the model menu's choice, so it is the one that runs (Auto never
    /// picks a slow, research-only model like Qwen Image 2.1 by itself). The recommended one stays on Auto.
    private func rememberImageChoice() {
        guard let image else { return }
        app.setModelChoice(image.id == imageChoices.first?.id ? "auto" : image.id, for: [.t2i, .edit])
    }

    private func requestDownloads() {
        if needsImage, let image, image.requiresLicenseAcceptance, !ModelLicenseAcceptance.hasAccepted(image) {
            licenseModel = image
        } else {
            startDownloads()
        }
    }

    private func startDownloads() {
        started = true
        error = nil
        if needsImage, let image { app.download(image) }
        guard !useService, useAgent, let target = agentModel?.id else { return }
        _Concurrency.Task {
            do {
                guard let client = app.client else { return }
                var next = try await client.selectPromptAssistant(model: target)
                if next.models?.first(where: { $0.id == target })?.installed == false {
                    next = try await client.pullPromptAssistant(model: target)
                }
                agent = next
            } catch {
                self.error = "The agent could not start downloading: \(error.localizedDescription)"
            }
        }
    }

    /// Reads the agent models; every second while one downloads.
    private func pollAgent() async {
        while !_Concurrency.Task.isCancelled {
            if let client = app.client, let status = try? await client.promptAssistant() {
                agent = status
                app.promptAssistantInstalled = status.installed
            }
            try? await _Concurrency.Task.sleep(for: .seconds(downloading ? 1 : 3))
        }
    }

    private func gb(_ value: Double?) -> String { String(format: "%.1f GB", value ?? 0) }
}

enum ModelLicenseAcceptance {
    private static let key = "acceptedModelLicenses"

    static func hasAccepted(_ model: EngineModel) -> Bool {
        guard let id = model.license_acceptance_id else { return true }
        return Set(UserDefaults.standard.stringArray(forKey: key) ?? []).contains(id)
    }

    static func accept(_ model: EngineModel) {
        guard let id = model.license_acceptance_id else { return }
        var accepted = Set(UserDefaults.standard.stringArray(forKey: key) ?? [])
        accepted.insert(id)
        UserDefaults.standard.set(Array(accepted).sorted(), forKey: key)
    }
}

struct ModelLicenseView: View {
    @Environment(\.dismiss) private var dismiss
    let model: EngineModel
    let onAccept: () -> Void

    var body: some View {
        let license = ModelLicense.name(model.license)
        VStack(alignment: .leading, spacing: 16) {
            Label(ModelLicense.title(model.license), systemImage: "exclamationmark.triangle.fill")
                .font(.title2.weight(.semibold))
                .foregroundStyle(.orange)
            Text("\(model.name) uses the \(license). \(ModelLicense.terms(model.license))")
                .fixedSize(horizontal: false, vertical: true)
            if let value = model.license_url, let url = URL(string: value) {
                Link("Read the \(license)", destination: url)
            }
            Text("By continuing, you confirm that you understand this restriction.")
                .font(.callout)
                .foregroundStyle(.secondary)
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                Button("I Understand — Download") { onAccept() }
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding(24)
        .frame(width: 500)
    }
}

/// Model licenses in plain words: a readable name, and what a restricted one allows.
enum ModelLicense {
    static func name(_ license: String) -> String {
        switch license.lowercased() {
        case "apache-2.0": "Apache 2.0 License"
        case "mit": "MIT License"
        case "qwen-research": "Qwen Research License"
        case "flux-non-commercial": "FLUX Non-Commercial License"
        default: license
        }
    }

    private static func isResearch(_ license: String) -> Bool { license.lowercased().contains("research") }
    private static func isNonCommercial(_ license: String) -> Bool {
        let l = license.lowercased()
        return l.contains("non-commercial") || l.contains("noncommercial") || l.hasSuffix("-nc")
    }

    /// A short tag for restricted models ("Research use only", "Non-commercial"), nil for open ones.
    static func restriction(_ license: String, needsAcceptance: Bool = false) -> String? {
        if isResearch(license) { return "Research use only" }
        if isNonCommercial(license) { return "Non-commercial" }
        return needsAcceptance ? "Restricted license" : nil
    }

    static func title(_ license: String) -> String {
        if isResearch(license) { return "Research-only model" }
        if isNonCommercial(license) { return "Non-commercial model" }
        return "Restricted license"
    }

    static func terms(_ license: String) -> String {
        switch license.lowercased() {
        case "qwen-research":
            "The model may only be used for non-commercial research or evaluation unless you obtain a separate commercial license from Qwen."
        case "flux-non-commercial":
            "The model may only be used for non-commercial purposes unless you obtain a separate commercial license from Black Forest Labs."
        default:
            isResearch(license) || isNonCommercial(license)
                ? "The model may not be used for commercial purposes. Read the license before you download it."
                : "Its license limits how the model may be used. Read it before you download the model."
        }
    }
}

/// Plain words about each model for the first-run setup and the model menus: what it is good at, not how.
enum ModelHints {
    static func image(_ m: EngineModel) -> String {
        switch m.id {
        case "flux2-klein-4b": "Fast and good at most things. Makes new images and edits yours."
        case "flux2-klein-9b": "The bigger Klein. Usually more detail than Klein 4B, but about twice as slow."
        case "z-image-turbo": "Very realistic photos, but slower. Makes new images only; it can't edit them."
        case "qwen-image-2.1-q4": "Best at writing words in images, like posters, and careful edits. Very slow: minutes per image."
        default: m.supports(.edit) ? "Makes new images and edits yours." : "Makes new images."
        }
    }

    /// One line for the composer's model picker.
    static func picker(_ m: EngineModel) -> String {
        switch m.id {
        case "flux2-klein-4b": "Fast and good at most things"
        case "flux2-klein-9b": "More detail, about twice as slow"
        case "z-image-turbo": "Very realistic photos, but slower"
        case "qwen-image-2.1-q4": "Best at words in images · minutes per image"
        case let id where id.hasPrefix("fibo"): "Good with text · Layout · very slow"
        case let id where id.hasPrefix("ideogram"): "Best at lettering · Layout · very slow"
        case let id where id.hasPrefix("moss"): "Layout · very slow"
        default: m.supports(.edit) ? "Makes new images and edits" : "Makes new images"
        }
    }

    static func imageShort(_ m: EngineModel) -> String {
        switch m.id {
        case "flux2-klein-4b": "fast, makes and edits images (recommended)"
        case "flux2-klein-9b": "more detail, about twice as slow"
        case "z-image-turbo": "realistic photos, slower, new images only"
        case "qwen-image-2.1-q4": "best at words in images, very slow, research use only"
        default: m.supports(.edit) ? "makes and edits images" : "makes new images"
        }
    }

    static func agent(_ m: PromptAssistantStatus.Model) -> String {
        switch m.id {
        case "qwen3-vl-8b-instruct-4bit": "Understands you best. Needs 24 GB of memory or more."
        case "ministral-3-3b-instruct-2512-4bit": "Good and light. Made for Macs with 16 GB of memory."
        case "qwen3-vl-4b-instruct-4bit": "Fastest and lightest, but gets things wrong more often."
        default: m.note
        }
    }

    static func agentShort(_ m: PromptAssistantStatus.Model) -> String {
        switch m.id {
        case "qwen3-vl-8b-instruct-4bit": "understands you best, 24 GB+"
        case "ministral-3-3b-instruct-2512-4bit": "good and light, for 16 GB Macs"
        case "qwen3-vl-4b-instruct-4bit": "fastest, makes more mistakes"
        default: m.note
        }
    }
}

/// One choice in the first-run setup: a model with a radio mark, its hint, size and download state.
struct ChoiceRow: View {
    struct State {
        var downloaded: Bool
        var status: String
        var progress: Double
        var error: String?
    }

    static let fill = Color(nsColor: .controlBackgroundColor)

    let name: String
    let hint: String
    let size: String
    var recommended = false
    var tag: String?
    let state: State
    let selected: Bool
    let action: () -> Void

    @Environment(\.isEnabled) private var isEnabled

    private var downloading: Bool { state.status == "downloading" }
    private var failure: String? { state.status == "error" ? (state.error ?? "unknown error") : nil }

    var body: some View {
        Button(action: action) {
            HStack(alignment: .top, spacing: 12) {
                RadioMark(selected: selected).padding(.top, 1)
                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 6) {
                        Text(name).font(.body.weight(.semibold))
                        if recommended { Badge(text: "Recommended", color: Theme.accent) }
                        if let tag { Badge(text: tag, color: .orange) }
                    }
                    Text(hint)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    if let failure {
                        Text("Download failed: \(failure)")
                            .font(.caption).foregroundStyle(.red).lineLimit(2)
                    }
                }
                Spacer(minLength: 8)
                VStack(alignment: .trailing, spacing: 4) {
                    Text(size).font(.callout).monospacedDigit().foregroundStyle(.secondary)
                    status
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(selected ? Theme.accent.opacity(0.09) : Self.fill, in: RoundedRectangle(cornerRadius: 10))
            .overlay(
                RoundedRectangle(cornerRadius: 10)
                    .strokeBorder(selected ? AnyShapeStyle(Theme.accent) : AnyShapeStyle(.separator), lineWidth: selected ? 1.5 : 1)
            )
            .contentShape(RoundedRectangle(cornerRadius: 10))
        }
        .buttonStyle(.plain)
        .opacity(isEnabled ? 1 : 0.6)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(name)
        .accessibilityValue(accessibilityDetail)
        .accessibilityHint(hint)
        .accessibilityAddTraits(selected ? [.isButton, .isSelected] : .isButton)
    }

    @ViewBuilder
    private var status: some View {
        if downloading {
            HStack(spacing: 6) {
                ProgressView(value: state.progress).frame(width: 64).controlSize(.small)
                Text("\(Int(state.progress * 100))%").font(.caption).monospacedDigit().foregroundStyle(.secondary)
            }
        } else if failure != nil {
            Label("Failed", systemImage: "exclamationmark.circle.fill")
                .font(.caption).foregroundStyle(.red)
        } else if state.downloaded {
            HStack(spacing: 3) {
                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                Text("Downloaded").foregroundStyle(.secondary)
            }
            .font(.caption)
        }
    }

    private var accessibilityDetail: String {
        var parts = [size]
        if recommended { parts.append("Recommended") }
        if let tag { parts.append(tag) }
        if downloading { parts.append("Downloading, \(Int(state.progress * 100)) percent") }
        else if let failure { parts.append("Download failed: \(failure)") }
        else if state.downloaded { parts.append("Downloaded") }
        else { parts.append("Not downloaded") }
        return parts.joined(separator: ", ")
    }
}

/// Where the agent runs, as one of two side-by-side cards.
struct PlaceCard: View {
    let symbol: String
    let title: String
    let detail: Text
    let selected: Bool
    let action: () -> Void

    @Environment(\.isEnabled) private var isEnabled

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Image(systemName: symbol)
                    .font(.system(size: 18))
                    .foregroundStyle(selected ? Theme.accent : .secondary)
                    .frame(width: 26)
                VStack(alignment: .leading, spacing: 1) {
                    Text(title).font(.body.weight(.semibold))
                    detail.font(.callout).foregroundStyle(.secondary).lineLimit(1)
                }
                Spacer(minLength: 4)
                RadioMark(selected: selected)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .frame(maxWidth: .infinity)
            .background(selected ? Theme.accent.opacity(0.09) : ChoiceRow.fill, in: RoundedRectangle(cornerRadius: 10))
            .overlay(
                RoundedRectangle(cornerRadius: 10)
                    .strokeBorder(selected ? AnyShapeStyle(Theme.accent) : AnyShapeStyle(.separator), lineWidth: selected ? 1.5 : 1)
            )
            .contentShape(RoundedRectangle(cornerRadius: 10))
        }
        .buttonStyle(.plain)
        .opacity(isEnabled ? 1 : 0.6)
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(selected ? [.isButton, .isSelected] : .isButton)
    }
}

/// A radio button's mark: a ring, filled with the accent color when selected.
struct RadioMark: View {
    let selected: Bool

    var body: some View {
        ZStack {
            if selected {
                Circle().fill(Theme.accent)
                Circle().fill(.white).frame(width: 6, height: 6)
            } else {
                Circle().strokeBorder(.tertiary, lineWidth: 1.5)
            }
        }
        .frame(width: 16, height: 16)
        .accessibilityHidden(true)
    }
}

struct Badge: View {
    let text: String
    let color: Color

    var body: some View {
        Text(text)
            .font(.caption2.weight(.semibold))
            .foregroundStyle(color)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.14), in: Capsule())
    }
}
