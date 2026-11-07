import AppKit
import DMLXKit
import SwiftUI
import TipKit
import UniformTypeIdentifiers

struct ComposerView: View {
    @Environment(AppModel.self) private var model
    @FocusState private var focused: Bool
    @State private var dropTargeted = false

    private let suggestions = ["Make it foggy", "Add warm evening light", "Make it black and white", "Variations"]

    var canSend: Bool {
        model.isEngineRunning && model.planningMessage == nil && !model.isGeneratingHere
            && !model.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        @Bindable var model = model
        VStack(alignment: .leading, spacing: 10) {
            if model.latestImageTurn?.current?.isActive == false && model.draft.isEmpty && model.attachments.isEmpty {
                FlowLayout(spacing: 8) {
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
                if let suggestion = model.promptSuggestion {
                    VStack(alignment: .leading, spacing: 5) {
                        HStack {
                            Label(model.promptSuggestionApplied ? "Enhanced prompt selected" : "Prompt suggestion", systemImage: "sparkles")
                                .font(.caption.weight(.semibold))
                            Spacer()
                            if !model.promptSuggestionApplied {
                                Button("Use") { model.applyPromptSuggestion() }
                                    .buttonStyle(.borderedProminent)
                                    .controlSize(.small)
                            }
                            Button("Dismiss") { model.dismissPromptSuggestion() }
                                .buttonStyle(.borderless)
                                .controlSize(.small)
                        }
                        TextField("Enhanced prompt", text: Binding(
                            get: { model.promptSuggestion?.positive ?? "" },
                            set: { model.promptSuggestion?.positive = $0 }
                        ), axis: .vertical)
                        .textFieldStyle(.roundedBorder)
                        .font(.caption)
                        if suggestion.negative_supported {
                            TextField("Negative prompt (optional)", text: Binding(
                                get: { model.promptSuggestion?.negative ?? "" },
                                set: { model.promptSuggestion?.negative = $0.isEmpty ? nil : $0 }
                            ), axis: .vertical)
                            .textFieldStyle(.roundedBorder)
                            .font(.caption)
                        }
                        if !suggestion.negative_supported {
                            Text("The selected image model does not use a separate negative prompt.")
                                .font(.caption2).foregroundStyle(.secondary)
                        }
                    }
                    .padding(8)
                    .background(Theme.accent.opacity(0.10), in: RoundedRectangle(cornerRadius: 9))
                }

                if model.needsRouteChoice {
                    FlowLayout(spacing: 8) {
                        // The agent's question is in the chat; here are the shortcuts to answer it.
                        Text(model.pendingAsk == nil ? "Choose the workflow" : "Reply in the chat, or:")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        ForEach(model.pendingRouteChoices, id: \.self) { operation in
                            Button(AppModel.operationLabel(operation)) { model.choosePendingRoute(operation) }
                                .buttonStyle(.bordered)
                                .controlSize(.small)
                        }
                        Button("Cancel") { model.cancelPendingRoute() }
                            .buttonStyle(.borderless)
                            .controlSize(.small)
                    }
                }

                if !model.attachments.isEmpty {
                    FlowLayout(spacing: 6) {
                        ForEach(Array(model.attachments.enumerated()), id: \.element.id) { index, a in
                            HStack(spacing: 6) {
                                Thumbnail(resultId: a.id, size: 22)
                                ContextChip(title: "\(index + 1) · \(a.label)") { model.removeAttachment(a.id) }
                            }
                            .draggable(a.id)
                            .dropDestination(for: String.self) { ids, _ in
                                guard let dragged = ids.first else { return false }
                                model.moveAttachment(dragged, before: a.id)
                                return true
                            }
                        }
                    }
                }

                if model.isWaitingForImage {
                    HStack(spacing: 6) {
                        ProgressView().controlSize(.small)
                        Text("Waiting for the image in “\(model.generatingElsewhere?.title ?? "another chat")” to finish, then planning this.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }

                TextField(
                    model.isGeneratingHere ? "Write your next message; send it when the image is done…"
                        : model.pendingAsk != nil ? "Reply, or ask for something else…"
                        : model.latestImageTurn == nil ? "Describe an image…" : "Describe a change, or something new…",
                    text: $model.draft, axis: .vertical
                )
                .textFieldStyle(.plain)
                .font(.body)
                .lineLimit(1...6)
                .focused($focused)
                .onSubmit { if canSend { model.send() } }
                .accessibilityLabel("Describe an image or a change")
                .guideTip(ComposerTip.self, in: Guide.shared.chat, edge: .bottom)

                HStack(spacing: 8) {
                    ScrollView(.horizontal) {
                        HStack(spacing: 8) {
                            Button {
                                pickFiles()
                            } label: {
                                Image(systemName: "paperclip")
                            }
                            .buttonStyle(.borderless)
                            .help("Attach up to 3 reference images")
                            .accessibilityLabel("Attach reference images")
                            .guideTip(AttachTip.self, in: Guide.shared.chat, edge: .bottom)

                            if model.promptEnhancementMode != .off {
                                Button { model.enhanceDraft() } label: {
                                    if model.isEnhancingPrompt {
                                        ProgressView().controlSize(.small)
                                    } else {
                                        Image(systemName: "sparkles")
                                    }
                                }
                                .buttonStyle(.borderless)
                                .disabled(model.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || model.isEnhancingPrompt)
                                .help("Preview an enhanced prompt before sending")
                                .accessibilityLabel("Enhance prompt")
                            }

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
                            .guideTip(SizeTip.self, in: Guide.shared.chat, edge: .bottom)

                            if model.choosableModels.count > 1 {
                                ModelMenu()
                            }

                            Group {
                            if model.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                                Text(model.routePreview)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            } else {
                                Menu {
                                    Button("Create a new image") { model.overrideRoute("image.generate") }
                                    if model.latestImageTurn != nil || !model.attachments.isEmpty {
                                        Button("Edit current or attached images") { model.overrideRoute("image.edit") }
                                        if model.defaultModel(for: "image.upscale") != nil {
                                            Button("Upscale current image 2×") { model.overrideRoute("image.upscale") }
                                        }
                                        if model.defaultModel(for: "image.restore") != nil {
                                            Button("Restore current image") { model.overrideRoute("image.restore") }
                                        }
                                        if model.defaultModel(for: "image.enhance") != nil {
                                            Button("Enhance current image") { model.overrideRoute("image.enhance") }
                                        }
                                    }
                                } label: {
                                    Text(model.routePreview)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)
                                }
                                .menuStyle(.borderlessButton)
                                .fixedSize()
                                .help("Choose the workflow")
                            }
                            }
                            .guideTip(ModeTip.self, in: Guide.shared.chat, edge: .bottom)
                        }
                        .fixedSize(horizontal: true, vertical: false)
                    }
                    .scrollIndicators(.hidden)

                    Spacer(minLength: 0)

                    if model.isGeneratingHere {
                        // The next message may refer to the image being made, so it waits for it; Stop ends it.
                        Button {
                            model.stopGeneratingHere()
                        } label: {
                            Image(systemName: "stop.fill")
                                .font(.caption.weight(.bold))
                                .frame(width: 28, height: 28)
                                .background(Circle().fill(Color.secondary.opacity(0.25)))
                        }
                        .buttonStyle(.plain)
                        .help("Stop the image being made")
                        .accessibilityLabel("Stop")
                    } else {
                        Button {
                            model.send()
                        } label: {
                            Image(systemName: "arrow.up")
                                .font(.body.weight(.semibold))
                                .frame(width: 28, height: 28)
                                .background(Circle().fill(canSend ? Theme.accent : Color.secondary.opacity(0.25)))
                                .foregroundStyle(canSend ? Color.white : Color.secondary)
                        }
                        .buttonStyle(.plain)
                        .disabled(!canSend)
                        .keyboardShortcut(.return, modifiers: .command)
                        .accessibilityLabel("Send")
                    }
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
        .onAppear { model.scheduleRoutePreview() }
        .onChange(of: model.draft) { model.scheduleRoutePreview() }
        .onChange(of: model.attachments) {
            focused = true
            model.scheduleRoutePreview()
        }
        .onChange(of: model.modelChoice) { model.scheduleRoutePreview() }
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
                    .lineLimit(1)
                    .truncationMode(.middle)
                Image(systemName: "xmark").font(.caption2.weight(.bold))
            }
            .font(.caption)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Theme.accent.opacity(0.18), in: RoundedRectangle(cornerRadius: 7))
        }
        .buttonStyle(.plain)
        .frame(maxWidth: 240)
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
