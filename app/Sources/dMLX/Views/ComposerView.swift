import AppKit
import DMLXKit
import SwiftUI
import UniformTypeIdentifiers

struct ComposerView: View {
    @Environment(AppModel.self) private var model
    @FocusState private var focused: Bool
    @State private var dropTargeted = false

    private let suggestions = ["Make it foggy", "Add warm evening light", "Make it black and white", "Variations"]

    var canSend: Bool {
        model.isEngineRunning && !model.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        @Bindable var model = model
        VStack(alignment: .leading, spacing: 10) {
            if model.latestImageTurn != nil && model.draft.isEmpty && model.targetTurnId == nil && model.attachments.isEmpty {
                HStack(spacing: 8) {
                    ForEach(suggestions, id: \.self) { s in
                        Button(s) {
                            if s == "Variations", let t = model.latestImageTurn {
                                model.variation(t.id)
                            } else {
                                model.send(s)
                            }
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }
                }
                .disabled(!model.isEngineRunning)
            }

            VStack(alignment: .leading, spacing: 8) {
                if model.targetTurn != nil || !model.attachments.isEmpty {
                    HStack(spacing: 6) {
                        if let t = model.targetTurn {
                            ContextChip(title: "Editing image \(t.number) · v\(t.activeVersion + 1)") { model.targetTurnId = nil }
                        }
                        ForEach(model.attachments) { a in
                            HStack(spacing: 6) {
                                Thumbnail(resultId: a.id, size: 22)
                                ContextChip(title: a.label) { model.removeAttachment(a.id) }
                            }
                        }
                    }
                }

                TextField(
                    model.latestImageTurn == nil ? "Describe an image…" : "Describe a change, or something new…",
                    text: $model.draft, axis: .vertical
                )
                .textFieldStyle(.plain)
                .font(.body)
                .lineLimit(1...6)
                .focused($focused)
                .onSubmit { if canSend { model.send() } }
                .accessibilityLabel("Describe an image or a change")

                HStack(spacing: 8) {
                    Button {
                        pickFiles()
                    } label: {
                        Image(systemName: "paperclip")
                    }
                    .buttonStyle(.borderless)
                    .help("Attach up to 3 reference images")
                    .accessibilityLabel("Attach reference images")

                    Menu {
                        ForEach(Aspect.allCases) { a in
                            Button(a.label) { model.aspect = a }
                        }
                    } label: {
                        Text(model.aspect.label).font(.caption)
                    }
                    .menuStyle(.borderlessButton)
                    .fixedSize()
                    .help("Size for new images (edits keep the source size)")

                    if model.choosableModels.count > 1 {
                        ModelMenu()
                    }

                    Text(model.routePreview)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)

                    Spacer()

                    Button {
                        model.send()
                    } label: {
                        Image(systemName: "arrow.up")
                            .font(.body.weight(.semibold))
                            .frame(width: 28, height: 28)
                            .background(Circle().fill(canSend ? Theme.accent : Color.secondary.opacity(0.25)))
                            .foregroundStyle(canSend ? Color.black.opacity(0.85) : Color.secondary)
                    }
                    .buttonStyle(.plain)
                    .disabled(!canSend)
                    .keyboardShortcut(.return, modifiers: .command)
                    .accessibilityLabel("Send")
                }
            }
            .padding(12)
            .background(RoundedRectangle(cornerRadius: 16).fill(.background.secondary))
            .overlay(RoundedRectangle(cornerRadius: 16).strokeBorder(dropTargeted ? Theme.accent : Color.secondary.opacity(0.25)))
            .onDrop(of: [.fileURL], isTargeted: $dropTargeted) { providers in
                loadDropped(providers)
                return true
            }
        }
        .onAppear { focused = true }
        .onChange(of: model.targetTurnId) { focused = true }
    }

    private func pickFiles() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.allowedContentTypes = [.image]
        if panel.runModal() == .OK {
            model.attach(files: panel.urls)
        }
    }

    private func loadDropped(_ providers: [NSItemProvider]) {
        for provider in providers {
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                guard let url else { return }
                _Concurrency.Task { @MainActor in model.attach(files: [url]) }
            }
        }
    }
}

struct ContextChip: View {
    let title: String
    let onRemove: () -> Void

    var body: some View {
        Button(action: onRemove) {
            HStack(spacing: 4) {
                Text(title)
                Image(systemName: "xmark").font(.caption2.weight(.bold))
            }
            .font(.caption)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Theme.accent.opacity(0.18), in: RoundedRectangle(cornerRadius: 7))
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(title). Remove")
    }
}


/// Model choice in the message box: Auto, or one downloaded model.
struct ModelMenu: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        Menu {
            Picker("Model", selection: Binding(get: { model.modelChoice }, set: { model.modelChoice = $0 })) {
                Text("Auto — picks for this Mac, one model for new images and edits").tag("auto")
                ForEach(model.choosableModels) { m in
                    Text("\(m.name) — \(AppModel.modelSummary(m))").tag(m.id)
                }
            }
            .pickerStyle(.inline)
            .labelsHidden()
            Divider()
            SettingsLink { Text("Manage models…") }
        } label: {
            Text(model.chosenModel?.name ?? "Auto").font(.caption)
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .help("Model for new images and edits")
        .accessibilityLabel("Model: \(model.chosenModel?.name ?? "Auto")")
    }
}
