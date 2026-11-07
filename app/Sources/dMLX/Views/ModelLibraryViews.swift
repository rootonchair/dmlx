import DMLXKit
import SwiftUI

/// Plain words for models listed as one entry per model with versions (Settings → Models, Add Model).
extension ModelHints {
    /// The model dMLX recommends (and ships): listed first, with a badge.
    static let recommendedGroup = "flux-2-klein-4b"

    /// What a model is good at, in a sentence; what it can do (new images, edits) is said separately.
    static func group(_ id: String) -> String {
        switch id {
        case "flux-2-klein-4b": "Fast and good at most things."
        case "flux-2-klein-9b": "Sharper, more faithful images and edits than Klein 4B. Slower and larger."
        case "flux-2-klein-9b-kv": "Klein 9B made faster at edits that use several images."
        case "flux-2-klein-base-4b", "flux-2-klein-base-9b":
            "The untuned original of Klein: much slower, with more varied results. For experts."
        case "flux-3-klein": "Newer Klein with finer detail. About 2 minutes per image."
        case "z-image-turbo": "Very realistic photos, but slower."
        case "z-image-base": "The untuned original of Z-Image: much slower, more varied. For experts."
        case "qwen-image-2-1-base": "Best at writing words in images, like posters, and careful edits. Very slow."
        case "ernie-image-turbo": "Follows long, detailed descriptions well. Slow: about 2.5 minutes per image."
        case "moss-image": "Very slow: minutes per image. Use it when you want its particular look."
        case "boogu-image-turbo": "Photographic, and good at text in English and Chinese. Slow."
        case "krea-2-turbo": "Strong photographic realism. About 2.5 minutes per image."
        case "iris-3b-turbo": "Small and fast, with simpler detail than the larger models."
        default: "An image model saved in mflux format."
        }
    }

    static func abilities(_ tasks: [String]) -> String {
        tasks.contains(GenTask.edit.rawValue) ? "New images and edits" : "New images only"
    }

    /// A version's name: "4-bit", or "Full size".
    static func variantName(_ bits: Int?) -> String {
        bits.map { "\($0)-bit" } ?? "Full size"
    }

    /// What choosing a version means, without the word quantization.
    static func variantHint(_ bits: Int?) -> String {
        switch bits {
        case 3: "Smallest download, lower quality"
        case 4: "Small download, good quality"
        case 5: "Slightly better quality, a bit larger"
        case 6: "Better quality, larger download"
        case 8: "Close to full quality, large download"
        case nil: "Full quality, largest download"
        default: "Smaller download, slightly lower quality"
        }
    }

    /// A license's readable name; for one dMLX does not know, its id in words ("Krea 2 Community License").
    static func license(_ id: String) -> String {
        let short = id.hasSuffix("-license") ? String(id.dropLast("-license".count)) : id
        for candidate in [id, short] where ModelLicense.name(candidate) != candidate { return ModelLicense.name(candidate) }
        if ["other", "unknown", ""].contains(id.lowercased()) { return "License on the model card" }
        let words = id.split(separator: "-").map { $0.count <= 3 && $0.allSatisfy(\.isLetter) && $0 != "non"
            ? $0.uppercased() : $0.prefix(1).uppercased() + $0.dropFirst() }
        let name = words.joined(separator: " ")
        return name.lowercased().hasSuffix("license") ? name : name + " License"
    }

    static func gb(_ value: Double?) -> String? {
        guard let value, value > 0 else { return nil }
        return String(format: "%.1f GB", value)
    }
}

/// The card style of the setup screen's option rows.
struct CardBackground: ViewModifier {
    var selected = false

    func body(content: Content) -> some View {
        content
            .background(selected ? Theme.accent.opacity(0.09) : ChoiceRow.fill, in: RoundedRectangle(cornerRadius: 10))
            .overlay(
                RoundedRectangle(cornerRadius: 10)
                    .strokeBorder(selected ? AnyShapeStyle(Theme.accent) : AnyShapeStyle(.separator), lineWidth: selected ? 1.5 : 1)
            )
    }
}

extension View {
    func card(selected: Bool = false) -> some View { modifier(CardBackground(selected: selected)) }
}

/// A model's name with its badges. No license badge: the license is named on the line below, and a
/// restricted one is accepted in a dialog before the download.
struct ModelGroupTitle: View {
    let group: ModelGroup
    var extra: String?

    var body: some View {
        HStack(spacing: 6) {
            Text(group.name).font(.body.weight(.semibold))
            if group.id == ModelHints.recommendedGroup { Badge(text: "Recommended", color: Theme.accent) }
            if let extra { Badge(text: extra, color: .secondary) }
        }
    }
}

/// What the model can do and its license, and a warning when it may not fit this Mac.
struct ModelGroupFacts: View {
    let group: ModelGroup

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(ModelHints.group(group.id))
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Text("\(ModelHints.abilities(group.tasks)) · \(ModelHints.license(group.license))")
                .font(.caption)
                .foregroundStyle(.secondary)
            if !group.fitsThisMac {
                Label("May not fit this Mac's memory", systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
    }
}

/// Settings → Models: one model, with the versions dMLX knows (downloaded or not) and what to do with each.
struct ModelGroupCard: View {
    let group: ModelGroup
    let otherVersions: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                ModelGroupTitle(group: group)
                Spacer(minLength: 8)
                Button("Other Versions…", action: otherVersions)
                    .buttonStyle(.link)
                    .font(.callout)
                    .help("Smaller or higher-quality versions of \(group.name)")
            }
            ModelGroupFacts(group: group)
            VStack(spacing: 0) {
                ForEach(Array(group.added.enumerated()), id: \.element.id) { index, variant in
                    if index > 0 { Divider() }
                    if let model = variant.model {
                        VariantLine(group: group, model: model)
                            .padding(.vertical, 7)
                    }
                }
            }
            .padding(.horizontal, 10)
            .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 7))
        }
        .padding(12)
        .card()
        .accessibilityElement(children: .contain)
        .accessibilityLabel(group.name)
    }
}

/// One version of a model the engine knows: its size, and Download, progress, or Downloaded with Remove.
struct VariantLine: View {
    @Environment(AppModel.self) private var app
    let group: ModelGroup
    let model: EngineModel
    @State private var confirmRemove = false
    @State private var showLicense = false

    private var title: String { "\(group.name) (\(ModelHints.variantName(model.quantization)))" }

    var body: some View {
        HStack(alignment: .center, spacing: 10) {
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 4) {
                    Text(ModelHints.variantName(model.quantization)).font(.callout.weight(.medium))
                    if model.isExperimental {
                        Text("· Experimental").font(.callout).foregroundStyle(.secondary)
                    }
                }
                Text(ModelHints.variantHint(model.quantization)).font(.caption).foregroundStyle(.secondary)
                if let error = model.download.error, model.download.status == "error" {
                    Text("Download failed: \(error)").font(.caption).foregroundStyle(.red).lineLimit(2)
                }
            }
            Spacer(minLength: 8)
            Text(sizeText).font(.callout).monospacedDigit().foregroundStyle(.secondary)
            actions
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(ModelHints.variantName(model.quantization)), \(sizeText)")
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

    @ViewBuilder
    private var actions: some View {
        if model.downloaded {
            HStack(spacing: 8) {
                Label("Downloaded", systemImage: "checkmark.circle.fill")
                    .labelStyle(DownloadedLabelStyle())
                    .font(.caption)
                Button("Remove…") { confirmRemove = true }
                    .controlSize(.small)
                    .accessibilityLabel("Remove \(title)")
            }
        } else if model.download.status == "downloading" {
            VStack(alignment: .trailing, spacing: 3) {
                ProgressView(value: model.download.progress).frame(width: 110).controlSize(.small)
                Text(String(format: "%.1f of %.1f GB", Double(model.download.done_bytes) / 1e9, Double(model.download.total_bytes) / 1e9))
                    .font(.caption2).monospacedDigit().foregroundStyle(.secondary)
            }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("Downloading, \(Int(model.download.progress * 100)) percent")
        } else {
            HStack(spacing: 6) {
                Button("Download") {
                    if model.requiresLicenseAcceptance && !ModelLicenseAcceptance.hasAccepted(model) {
                        showLicense = true
                    } else {
                        app.download(model)
                    }
                }
                .controlSize(.small)
                .disabled(!app.isEngineRunning)
                .accessibilityLabel("Download \(title)")
                if model.isUserAdded {
                    // Not downloaded, so nothing to delete: only take it off the list.
                    Button { app.remove(model) } label: { Image(systemName: "xmark.circle.fill") }
                        .buttonStyle(.borderless)
                        .foregroundStyle(.secondary)
                        .help("Remove from your models")
                        .accessibilityLabel("Remove \(title) from your models")
                }
            }
        }
    }

    private var removeTitle: String { model.isUserAdded ? "Remove \(title)?" : "Delete the files of \(title)?" }

    private var removeMessage: String {
        let files = "This deletes \(sizeText) from the Hugging Face cache on this Mac. Other apps that use the same model would need to download it again."
        return model.isUserAdded ? files : files + " It stays in the list, and you can download it again anytime."
    }

    private var sizeText: String {
        ModelHints.gb(model.size_gb) ?? ModelHints.gb(model.repo.flatMap { app.catalogSizes[$0] })
            ?? (model.download.total_bytes > 0 ? String(format: "%.1f GB", Double(model.download.total_bytes) / 1e9) : "about 5 GB")
    }
}

/// A green check and the word, in the secondary color (as in the setup screen).
struct DownloadedLabelStyle: LabelStyle {
    func makeBody(configuration: Configuration) -> some View {
        HStack(spacing: 3) {
            configuration.icon.foregroundStyle(.green)
            configuration.title.foregroundStyle(.secondary)
        }
    }
}

/// Settings → Models: which model makes new images and which edits them.
struct TaskModelChoices: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 8) {
                row(.t2i)
                row(.edit)
            }
            Text("Auto picks for this Mac and keeps one model for both when it can, so follow-ups start faster. Also in the message box.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .card()
    }

    private func row(_ task: GenTask) -> some View {
        GridRow {
            Text(task == .edit ? "Edits" : "New images")
                .gridColumnAlignment(.trailing)
            Picker(task == .edit ? "Edits" : "New images", selection: Binding(
                get: { app.chosenModel(for: task)?.id ?? "auto" },
                set: { app.setModelChoice($0, for: [task]) }
            )) {
                Text(autoTitle(task)).tag("auto")
                Divider()
                ForEach(app.choosableModels(for: task)) { m in Text(app.menuName(m)).tag(m.id) }
            }
            .labelsHidden()
            .fixedSize()
        }
    }

    private func autoTitle(_ task: GenTask) -> String {
        guard app.chosenModel(for: task) == nil, let m = app.defaultModel(for: task) else { return "Auto" }
        return "Auto (\(app.menuName(m)))"
    }
}
