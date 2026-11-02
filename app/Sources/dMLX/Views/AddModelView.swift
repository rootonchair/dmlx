import DMLXKit
import SwiftUI

/// Add a model from Hugging Face: pick a suggestion or paste a repo, check it, then add and download.
struct AddModelView: View {
    @Environment(AppModel.self) private var app
    @Environment(\.dismiss) private var dismiss

    @State private var query = ""
    @State private var suggestions: [CatalogSuggestion] = []
    @State private var suggestionsError: String?
    @State private var checking = false
    @State private var adding = false
    @State private var inspection: ModelInspection?
    @State private var error: String?
    @State private var confirmRestrictedLicense = false

    private var filtered: [CatalogSuggestion] {
        let words = query.lowercased().split(separator: " ").map(String.init)
        guard !words.isEmpty, !query.contains("/") else { return suggestions }
        return suggestions.filter { s in
            let text = "\(s.label ?? "") \(s.repo)".lowercased()
            return words.allSatisfy { text.contains($0) }
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Add a model").font(.title2.weight(.semibold))
            Text("dMLX runs FLUX.2 Klein, FLUX.3 Klein, Z-Image, Qwen Image 2.1 Base, ERNIE Image Turbo and MOSS Image models saved in mflux format. Search the community versions below, or paste any Hugging Face repo or link.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            HStack {
                TextField("Search, or paste owner/repo or a huggingface.co link", text: $query)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { check(query) }
                Button("Check") { check(query) }
                    .disabled(query.trimmingCharacters(in: .whitespaces).isEmpty || checking)
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

            Text("Community versions (mflux-community)").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            List(filtered) { s in
                Button {
                    query = s.repo
                    check(s.repo)
                } label: {
                    HStack {
                        VStack(alignment: .leading, spacing: 1) {
                            Text(s.label ?? s.repo)
                            Text(s.repo).font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        if s.added {
                            Text("Added").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
            .listStyle(.bordered)
            .overlay {
                if let suggestionsError {
                    Text(suggestionsError).font(.callout).foregroundStyle(.secondary).padding()
                } else if suggestions.isEmpty {
                    ProgressView()
                } else if filtered.isEmpty {
                    Text("No community version matches. Press Check to look up \"\(query)\".").font(.callout).foregroundStyle(.secondary)
                }
            }

            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Button(addTitle) {
                    if inspection?.permissive_license == false {
                        confirmRestrictedLicense = true
                    } else {
                        addConfirmed()
                    }
                }
                    .keyboardShortcut(.defaultAction)
                    .disabled(inspection == nil || inspection?.already_added != nil || adding)
            }
        }
        .padding(20)
        .frame(width: 560, height: 600)
        .task { await loadSuggestions() }
        .confirmationDialog(
            "Use a restricted-license model?", isPresented: $confirmRestrictedLicense,
            titleVisibility: .visible
        ) {
            Button("I Understand — Add and Download") { addConfirmed() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This model may prohibit commercial use. By continuing, you confirm that you have read its model card and will follow its license.")
        }
    }

    private var addTitle: String {
        if inspection?.already_added != nil { return "Already Added" }
        if let gb = inspection?.size_gb { return String(format: "Add and Download %.1f GB", gb) }
        return "Add and Download"
    }

    private func loadSuggestions() async {
        do {
            suggestions = try await app.catalogSuggestions()
        } catch {
            suggestionsError = "Couldn't load community versions. You can still paste a repo above."
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
        guard let inspection else { return }
        adding = true
        _Concurrency.Task {
            defer { adding = false }
            do {
                try await app.addModel(inspection.repo)
                dismiss()
            } catch {
                self.error = error.localizedDescription
            }
        }
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
