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

    /// Image models this Mac can run and Auto can pick, recommended one first.
    private var imageChoices: [EngineModel] {
        let fitting = app.models.filter { $0.fits_this_mac && $0.autoSelectable && $0.supports(.t2i) }
        let preferred = ["flux2-klein-4b", "z-image-turbo"]
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
    private var needsAgent: Bool { useAgent && !(agentModel?.installed ?? true) }
    private var downloadGB: Double {
        (needsImage ? image?.size_gb ?? 0 : 0) + (needsAgent ? agentModel?.size_gb ?? 0 : 0)
    }
    private var downloading: Bool {
        image?.download.status == "downloading" || (useAgent && agentModel?.isDownloading == true)
    }
    private var imageReady: Bool { image?.downloaded ?? false }

    var body: some View {
        let mac = ThisMac(app)
        let enoughSpace = mac.freeGB.map { $0 > downloadGB + 2 } ?? true
        VStack(alignment: .leading, spacing: 20) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Recommended for this Mac")
                    .font(.system(size: 26, weight: .regular, design: .serif))
                Text("dMLX needs an image model. The agent is optional, but it makes the chat work much better: it reads your messages and images and decides what to do.")
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if !app.isEngineRunning || app.models.isEmpty {
                HStack {
                    ProgressView().controlSize(.small)
                    Text(app.engineStatusText).foregroundStyle(.secondary)
                }
            }

            if let image {
                card(
                    title: "Image model", name: image.name,
                    detail: "\(image.supports(.edit) ? "Makes and edits images" : "Makes images") · \(licenseName(image.license)) · \(gb(image.size_gb))",
                    downloaded: image.downloaded, progress: image.download.status == "downloading" ? image.download.progress : nil,
                    failure: image.download.status == "error" ? image.download.error : nil
                ) {
                    Menu("Change") {
                        ForEach(imageChoices) { m in
                            Button { imageId = m.id } label: {
                                Text("\(m.name) · \(gb(m.size_gb))\(m.downloaded ? " · downloaded" : "")")
                            }
                        }
                    }
                    .menuStyle(.borderlessButton)
                    .fixedSize()
                    .disabled(started && downloading)
                }
            }

            if let agentModel {
                card(
                    title: "Agent", name: agentModel.name,
                    detail: ([licenseName(agentModel.license), String(format: "%.1f GB", agentModel.size_gb)] + (agentModel.note.isEmpty ? [] : [agentModel.note])).joined(separator: " · "),
                    downloaded: agentModel.installed, progress: agentModel.isDownloading ? agentModel.download.progress : nil,
                    failure: agentModel.download.status == "error" ? agentModel.download.error : nil,
                    dimmed: !useAgent
                ) {
                    HStack(spacing: 10) {
                        Menu("Change") {
                            ForEach(agentChoices) { m in
                                Button { agentId = m.id } label: {
                                    Text("\(m.name) · \(String(format: "%.1f GB", m.size_gb))\(m.installed ? " · downloaded" : "")")
                                }
                            }
                        }
                        .menuStyle(.borderlessButton)
                        .fixedSize()
                        .disabled(!useAgent || (started && downloading))
                        if !agentModel.installed {
                            Toggle("Download the agent", isOn: $useAgent)
                                .toggleStyle(.switch)
                                .controlSize(.small)
                                .labelsHidden()
                                .disabled(started && downloading)
                                .help("Without the agent, dMLX routes messages with simple rules.")
                        }
                    }
                }
            }

            footer(mac: mac, enoughSpace: enoughSpace)
        }
        .task { await pollAgent() }
        .onChange(of: app.harnessOnboardingDownload) { requestDownloads() }
        .sheet(item: $licenseModel) { m in
            ModelLicenseView(model: m) {
                ModelLicenseAcceptance.accept(m)
                licenseModel = nil
                startDownloads()
            }
        }
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
                    Button("Download \(gb(downloadGB))") { requestDownloads() }
                        .keyboardShortcut(.defaultAction)
                        .buttonStyle(.borderedProminent)
                        .disabled(!enoughSpace || image == nil || !app.isEngineRunning)
                } else if !imageReady {
                    Button("Continue in Background", action: finish)
                        .keyboardShortcut(.defaultAction)
                } else {
                    Button("Start Creating", action: finish)
                        .keyboardShortcut(.defaultAction)
                        .buttonStyle(.borderedProminent)
                }
            }
        }
    }

    private func card<Accessory: View>(title: String, name: String, detail: String, downloaded: Bool,
                                       progress: Double?, failure: String?, dimmed: Bool = false,
                                       @ViewBuilder accessory: () -> Accessory) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                Text(title.uppercased())
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .tracking(0.6)
                Spacer()
                accessory()
            }
            HStack(alignment: .center, spacing: 12) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(name).font(.headline)
                    Text(detail).font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                if let progress {
                    ProgressView(value: progress).frame(width: 130)
                    Text("\(Int(progress * 100))%").font(.caption).monospacedDigit().foregroundStyle(.secondary)
                } else if downloaded {
                    Label("Downloaded", systemImage: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                        .font(.callout)
                } else if dimmed {
                    Text("Off").font(.callout).foregroundStyle(.secondary)
                }
            }
            if let failure {
                Text("Download failed: \(failure)").font(.caption).foregroundStyle(.red).lineLimit(2)
            }
        }
        .padding(14)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 12))
        .opacity(dimmed ? 0.55 : 1)
    }

    private func spaceLine(mac: ThisMac, enoughSpace: Bool) -> String {
        guard let free = mac.freeGB else { return "Total download: \(gb(downloadGB))." }
        let base = "Total download: \(gb(downloadGB)). This Mac has \(Int(free)) GB free."
        return enoughSpace ? base + " Models are saved in your Hugging Face cache." : base + " Free up some space, or pick smaller models."
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
        guard useAgent, let target = agentModel?.id else { return }
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
    private func licenseName(_ license: String) -> String { license.lowercased() == "apache-2.0" ? "Apache-2.0" : license }
}

struct ModelRow: View {
    @Environment(AppModel.self) private var app
    let model: EngineModel
    var removable = false
    @State private var confirmRemove = false
    @State private var showLicense = false

    var role: String {
        model.supports(.edit) ? "Creates and edits images" : "Creates images"
    }

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(model.name).font(.headline)
                Text("\(role) · \(model.license) · \(sizeText)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            if model.downloaded {
                Label("Downloaded", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                    .font(.callout)
                if removable {
                    Button("Remove…") { confirmRemove = true }
                        .controlSize(.small)
                }
            } else if model.download.status == "downloading" {
                VStack(alignment: .trailing, spacing: 4) {
                    ProgressView(value: model.download.progress).frame(width: 140)
                    Text(String(format: "%.1f of %.1f GB", Double(model.download.done_bytes) / 1e9, Double(model.download.total_bytes) / 1e9))
                        .font(.caption2)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
            } else {
                VStack(alignment: .trailing, spacing: 4) {
                    HStack(spacing: 6) {
                        Button("Download") {
                            if model.requiresLicenseAcceptance && !ModelLicenseAcceptance.hasAccepted(model) {
                                showLicense = true
                            } else {
                                app.download(model)
                            }
                        }
                        if removable && model.isUserAdded {
                            Button("Remove") { app.remove(model) }.controlSize(.small)
                        }
                    }
                    if let error = model.download.error {
                        Text(error).font(.caption2).foregroundStyle(.red).lineLimit(2)
                    }
                }
            }
        }
        .padding(12)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 10))
        .confirmationDialog(removeTitle, isPresented: $confirmRemove, titleVisibility: .visible) {
            Button(model.isUserAdded ? "Remove and Delete Files" : "Delete Files", role: .destructive) { app.remove(model) }
        } message: {
            Text(removeMessage)
        }
        .sheet(isPresented: $showLicense) {
            ModelLicenseView(model: model) {
                ModelLicenseAcceptance.accept(model)
                showLicense = false
                app.download(model)
            }
        }
    }

    var removeTitle: String { model.isUserAdded ? "Remove \(model.name)?" : "Delete the files of \(model.name)?" }

    var removeMessage: String {
        let files = "This deletes \(sizeText) from the Hugging Face cache on this Mac. Other apps that use the same model would need to download it again."
        return model.isUserAdded ? files : files + " \(model.name) stays in the list, and you can download it again anytime."
    }

    var sizeText: String {
        if let gb = model.size_gb, gb > 0 { return String(format: "%.1f GB", gb) }
        if model.download.total_bytes > 0 { return String(format: "%.1f GB", Double(model.download.total_bytes) / 1e9) }
        return "about 5 GB"
    }
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
        VStack(alignment: .leading, spacing: 16) {
            Label("Research-only model", systemImage: "exclamationmark.triangle.fill")
                .font(.title2.weight(.semibold))
                .foregroundStyle(.orange)
            Text("\(model.name) uses the Qwen Research License. The model weights may only be used for non-commercial research or evaluation unless you obtain a separate commercial license from Qwen.")
                .fixedSize(horizontal: false, vertical: true)
            if let value = model.license_url, let url = URL(string: value) {
                Link("Read the Qwen Research License", destination: url)
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
