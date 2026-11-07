import DMLXKit
import SwiftUI

/// Add a model: one entry per model (like ollama.com/library), each with its versions to pick from, or any
/// Hugging Face repo pasted in the search field. Adding starts the download; progress shows in Settings → Models.
struct AddModelView: View {
    @Environment(AppModel.self) private var app
    @Environment(\.dismiss) private var dismiss

    var initialGroup: String?

    @State private var query = ""
    @State private var selectedGroup: String?
    /// The version picked per model (its repo); the model's default until changed.
    @State private var picked: [String: String] = [:]
    @State private var checking = false
    @State private var adding = false
    @State private var inspection: ModelInspection?
    @State private var error: String?
    @State private var confirmRestrictedLicense = false
    @State private var licenseModel: EngineModel?

    private var groups: [ModelGroup] {
        ModelCatalog.groups(models: app.models, suggestions: app.catalog ?? []).sorted { a, b in
            (a.id == ModelHints.recommendedGroup ? 0 : 1, -a.downloads, a.name)
                < (b.id == ModelHints.recommendedGroup ? 0 : 1, -b.downloads, b.name)
        }
    }

    /// A repo or a huggingface.co link is looked up instead of searched for.
    private var isRepo: Bool { query.contains("/") }

    private var filtered: [ModelGroup] {
        let words = query.lowercased().split(separator: " ").map(String.init)
        guard !words.isEmpty, !isRepo else { return groups }
        return groups.filter { g in
            let text = ([g.name, ModelHints.group(g.id), ModelHints.abilities(g.tasks)] + g.variants.map(\.repo))
                .joined(separator: " ").lowercased()
            return words.allSatisfy { text.contains($0) }
        }
    }

    private var group: ModelGroup? { groups.first { $0.id == selectedGroup } }

    private func variant(of group: ModelGroup) -> ModelVariant? {
        group.variant(picked[group.id]) ?? group.defaultVariant
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Add a Model").font(.title2.weight(.semibold))
                Text("Pick a model, then a version. Smaller versions download faster; larger ones keep a little more detail.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            HStack {
                TextField("Search models, or paste a Hugging Face link", text: $query)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { if isRepo { check(query) } }
                    .accessibilityLabel("Search models, or paste a Hugging Face repo or link")
                if isRepo {
                    Button("Check") { check(query) }
                        .disabled(checking)
                }
            }

            if checking {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Reading the repo from Hugging Face…").font(.callout).foregroundStyle(.secondary)
                }
            } else if let error {
                Label(error, systemImage: "exclamationmark.triangle")
                    .font(.callout)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            } else if let inspection {
                InspectionCard(inspection: inspection)
            }

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 6) {
                        ForEach(filtered) { g in
                            GroupChoice(
                                group: g, selected: g.id == selectedGroup && inspection == nil,
                                variant: Binding(get: { variant(of: g)?.repo ?? "" }, set: { picked[g.id] = $0 }),
                                sizes: app.catalogSizes
                            ) { select(g) }
                            .id(g.id)
                        }
                    }
                    .padding(.vertical, 2)
                }
                .scrollBounceBehavior(.basedOnSize)
                .overlay {
                    if app.catalog == nil && app.catalogError == nil {
                        ProgressView()
                    } else if filtered.isEmpty {
                        Text(isRepo ? "Press Check to look up this repo." : "No model matches \u{201C}\(query)\u{201D}.")
                            .font(.callout).foregroundStyle(.secondary)
                    }
                }
                .onChange(of: app.catalog?.count) {
                    if let initialGroup, selectedGroup == initialGroup { proxy.scrollTo(initialGroup, anchor: .top) }
                }
            }
            if let catalogError = app.catalogError {
                Text(catalogError).font(.caption).foregroundStyle(.secondary)
            }

            HStack {
                Text(footnote).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                Spacer()
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button(primaryTitle) { primary() }
                    .keyboardShortcut(.defaultAction)
                    .buttonStyle(.borderedProminent)
                    .disabled(!primaryEnabled)
            }
        }
        .padding(20)
        .frame(width: 600, height: 640)
        .task {
            selectedGroup = initialGroup ?? ModelHints.recommendedGroup
            await app.loadCatalog()
            if let group { await app.loadSizes(group.variants.map(\.repo)) }
        }
        .onChange(of: selectedGroup) {
            guard let group else { return }
            _Concurrency.Task { await app.loadSizes(group.variants.filter { $0.model == nil }.map(\.repo)) }
        }
        .confirmationDialog(
            "Use a restricted-license model?", isPresented: $confirmRestrictedLicense,
            titleVisibility: .visible
        ) {
            Button("I Understand — Add and Download") { addConfirmed() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This model may prohibit commercial use. By continuing, you confirm that you have read its model card and will follow its license.")
        }
        .sheet(item: $licenseModel) { m in
            ModelLicenseView(model: m) {
                ModelLicenseAcceptance.accept(m)
                licenseModel = nil
                app.download(m)
                dismiss()
            }
        }
    }

    private func select(_ g: ModelGroup) {
        selectedGroup = g.id
        inspection = nil
        error = nil
    }

    // MARK: the main button

    /// What the main button acts on: a looked-up repo, or the picked version of the selected model.
    private enum Target {
        case inspected(ModelInspection)
        case catalog(ModelGroup, ModelVariant)
    }

    private var target: Target? {
        if let inspection { return .inspected(inspection) }
        guard let group, let variant = variant(of: group) else { return nil }
        return .catalog(group, variant)
    }

    private func size(_ variant: ModelVariant) -> Double? {
        variant.model?.size_gb ?? app.catalogSizes[variant.repo]
    }

    private var primaryTitle: String {
        switch target {
        case .inspected(let i):
            return i.already_added != nil ? "Already Added" : String(format: "Add and Download %.1f GB", i.size_gb)
        case .catalog(_, let v):
            if v.isDownloaded { return "Downloaded" }
            if v.isDownloading { return "Downloading…" }
            let gb = ModelHints.gb(size(v)).map { " \($0)" } ?? ""
            return v.isAdded ? "Download\(gb)" : "Add and Download\(gb)"
        case nil:
            return "Add and Download"
        }
    }

    private var primaryEnabled: Bool {
        guard app.isEngineRunning, !adding, !checking else { return false }
        switch target {
        case .inspected(let i): return i.already_added == nil
        case .catalog(_, let v): return !v.isDownloaded && !v.isDownloading
        case nil: return false
        }
    }

    private var footnote: String {
        guard case .catalog(_, let v) = target, let model = v.model else { return "" }
        return model.downloaded ? "This version is on this Mac." : model.isUserAdded ? "Added, not downloaded yet." : "Comes with dMLX."
    }

    private func primary() {
        switch target {
        case .inspected(let i):
            if !i.permissive_license { confirmRestrictedLicense = true } else { addConfirmed() }
        case .catalog(_, let v):
            if let model = v.model {
                if model.requiresLicenseAcceptance && !ModelLicenseAcceptance.hasAccepted(model) {
                    licenseModel = model
                } else {
                    app.download(model)
                    dismiss()
                }
            } else if v.suggestion?.permissive_license == false {
                confirmRestrictedLicense = true
            } else {
                addConfirmed()
            }
        case nil:
            break
        }
    }

    private func check(_ text: String) {
        let text = text.trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty else { return }
        checking = true
        error = nil
        inspection = nil
        _Concurrency.Task {
            defer { checking = false }
            do {
                inspection = try await app.inspectModel(text)
            } catch {
                self.error = error.localizedDescription
            }
        }
    }

    private func addConfirmed() {
        let repo: String
        switch target {
        case .inspected(let i): repo = i.repo
        case .catalog(_, let v): repo = v.repo
        case nil: return
        }
        adding = true
        error = nil
        _Concurrency.Task {
            defer { adding = false }
            do {
                try await app.addModel(repo)
                dismiss()
            } catch {
                self.error = error.localizedDescription
            }
        }
    }
}

/// One model in Add Model, in the setup screen's option style; selected, it shows its versions to pick from.
private struct GroupChoice: View {
    let group: ModelGroup
    let selected: Bool
    @Binding var variant: String
    let sizes: [String: Double]
    let action: () -> Void

    private var state: String? {
        if group.isDownloaded { return "On this Mac" }
        if group.isAdded { return "Added" }
        return nil
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Button(action: action) {
                HStack(alignment: .top, spacing: 12) {
                    RadioMark(selected: selected).padding(.top, 1)
                    VStack(alignment: .leading, spacing: 3) {
                        ModelGroupTitle(group: group, extra: state)
                        ModelGroupFacts(group: group)
                    }
                    Spacer(minLength: 8)
                    if !selected {
                        Text(group.variants.count == 1 ? "1 version" : "\(group.variants.count) versions")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(group.name)
            .accessibilityValue([ModelHints.abilities(group.tasks), ModelHints.license(group.license), state ?? ""]
                .filter { !$0.isEmpty }.joined(separator: ", "))
            .accessibilityHint(ModelHints.group(group.id))
            .accessibilityAddTraits(selected ? [.isButton, .isSelected] : .isButton)

            if selected {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 10) {
                        Text("Version").font(.callout)
                        Picker("Version", selection: $variant) {
                            ForEach(group.variants) { v in Text(label(v)).tag(v.repo) }
                        }
                        .labelsHidden()
                        .fixedSize()
                    }
                    Text(ModelHints.variantHint(group.variant(variant)?.quantization))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding(.leading, 62)
                }
                .padding(.leading, 28)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .card(selected: selected)
    }

    /// "4-bit · 4.3 GB (recommended)", with "on this Mac" or "added" for versions dMLX already has.
    private func label(_ v: ModelVariant) -> String {
        var text = ModelHints.variantName(v.quantization)
        if let gb = ModelHints.gb(v.model?.size_gb ?? sizes[v.repo]) { text += " · \(gb)" }
        if v.quantization == ModelGroup.recommendedBits { text += " (recommended)" }
        if v.isDownloaded { text += " — on this Mac" } else if v.isAdded { text += " — added" }
        return text
    }
}

struct InspectionCard: View {
    let inspection: ModelInspection

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(inspection.name).font(.headline)
            Text("\(inspection.supports(.edit) ? "Creates and edits images" : "Creates images") · \(String(format: "%.1f", inspection.size_gb)) GB download")
                .font(.callout)
                .foregroundStyle(.secondary)
            if inspection.permissive_license {
                Text("License: \(inspection.license)").font(.callout).foregroundStyle(.secondary)
            } else {
                Label("License: \(inspection.license). It may not allow commercial use; read the model card before using the images for work.",
                      systemImage: "exclamationmark.triangle")
                    .font(.callout)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let added = inspection.already_added {
                Text("Already in your models (\(added)).").font(.callout).foregroundStyle(.secondary)
            }
            if let url = URL(string: inspection.url) {
                Link("Open the model card", destination: url).font(.callout)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 10))
    }
}
