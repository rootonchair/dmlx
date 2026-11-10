import AppKit
import DMLXKit
import SwiftUI
import TipKit
import UniformTypeIdentifiers

struct ComposerView: View {
    @Environment(AppModel.self) private var model
    @FocusState private var focused: Bool
    @State private var dropTargeted = false
    /// The message field's width. The field is rebuilt when it changes: its text view otherwise keeps wrapping
    /// at the old width after the window is resized (lines cut off at the edge, or the last line hidden).
    @State private var fieldWidth: CGFloat = 0

    private let suggestions = ["Make it foggy", "Add warm evening light", "Make it black and white", "Variations"]

    var canSend: Bool {
        model.isEngineRunning && model.planningMessage == nil && !model.isGeneratingHere
            && model.sendAfterUpload == nil
            && !model.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        @Bindable var model = model
        VStack(alignment: .leading, spacing: 10) {
            // Not while a message is being planned or waits on the agent's question: a chip would send another.
            if model.latestImageTurn?.current?.isActive == false && model.draft.isEmpty && model.attachments.isEmpty
                && model.planningMessage == nil && !model.needsRouteChoice && model.pendingAsk == nil {
                FlowLayout(spacing: 8, alignment: .leading) {
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
                    FlowLayout(spacing: 8, alignment: .leading) {
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
                            .keyboardShortcut(.cancelAction)
                    }
                }

                if let layout = model.pendingLayout, !layout.isEmpty {
                    HStack(spacing: 6) {
                        Image(systemName: "rectangle.dashed").foregroundStyle(.secondary)
                        Text("Layout · \(layout.count) box\(layout.count == 1 ? "" : "es")\(model.pendingLayoutSource.map { " · \($0)" } ?? "")")
                            .font(.caption)
                        Button("Change") { model.openBlankLayout() }
                            .buttonStyle(.borderless).controlSize(.small)
                        Button { model.clearPendingLayout() } label: { Image(systemName: "xmark").font(.caption2.weight(.bold)) }
                            .buttonStyle(.borderless)
                            .accessibilityLabel("Remove layout")
                    }
                    .padding(.horizontal, 8).padding(.vertical, 4)
                    .background(Theme.accent.opacity(0.18), in: RoundedRectangle(cornerRadius: 7))
                    .help("These boxes go with your next message; the agent writes the rest")
                }

                if !model.attachments.isEmpty || !model.uploading.isEmpty {
                    FlowLayout(spacing: 6, alignment: .leading) {
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
                        // Still uploading: shown so a send can wait for it (task 19).
                        ForEach(model.uploading, id: \.token) { item in
                            HStack(spacing: 6) {
                                ProgressView().controlSize(.mini)
                                Text("Attaching \(item.label)…").font(.caption).foregroundStyle(.secondary).lineLimit(1)
                            }
                            .padding(.horizontal, 8).padding(.vertical, 4)
                            .background(.quaternary.opacity(0.6), in: RoundedRectangle(cornerRadius: 7))
                            .accessibilityElement(children: .combine)
                        }
                    }
                }

                if model.sendAfterUpload != nil {
                    HStack(spacing: 6) {
                        ProgressView().controlSize(.small)
                        Text("Sends when the image is attached.").font(.caption).foregroundStyle(.secondary)
                        // A way out if the engine hangs: the uploads stop and the message stays (task 19).
                        Button("Cancel") { model.cancelHeldSend() }
                            .buttonStyle(.borderless)
                            .controlSize(.small)
                            .help("Stop attaching and keep the message in the box")
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
                .id(fieldWidth)
                .frame(maxWidth: .infinity, alignment: .leading)
                .onGeometryChange(for: CGFloat.self) { $0.size.width.rounded() } action: { width in
                    let wasFocused = focused
                    fieldWidth = width
                    if wasFocused { _Concurrency.Task { @MainActor in focused = true } }
                }
                .guideTip(ComposerTip.self, in: Guide.shared.chat, edge: .bottom)
                // The text view takes file drops itself and pastes their paths; this catches them first and
                // attaches the files, while text drops and clicks still reach the field (task 19).
                .overlay(FileDropCatcher(targeted: $dropTargeted) { model.attach(files: $0) })

                HStack(alignment: .bottom, spacing: 8) {
                    // Wraps when narrow (the route preview moves to its own line), so every control stays in view.
                    FlowLayout(spacing: 8, alignment: .leading) {
                        Button {
                            pickFiles()
                        } label: {
                            Image(systemName: "paperclip")
                        }
                        .buttonStyle(.borderless)
                        .help(model.attachLimit.count == 1 ? "Attach 1 image" : "Attach up to \(model.attachLimit.count) images")
                        .accessibilityLabel("Attach reference images")
                        .guideTip(AttachTip.self, in: Guide.shared.chat, edge: .bottom)

                        if model.layoutAvailable {
                            Button { model.openBlankLayout() } label: {
                                Label("Layout", systemImage: "rectangle.dashed").font(.caption)
                            }
                            .buttonStyle(.borderless)
                            .help("Draw where things go before the image is made")
                            .accessibilityLabel("Draw a layout")
                        }

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

                        if model.hasModelChoice {
                            // With a model chosen and nothing typed, the route preview is empty: the tip about
                            // choosing new image or edit points here instead of at nothing.
                            ModelMenu()
                                .guideTip(ModeTip.self, in: model.routePreview.isEmpty ? Guide.shared.chat : nil, edge: .bottom)
                        }

                        Group {
                        if model.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || model.hasNoImageModel {
                            Text(model.routePreview)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        } else {
                            Menu {
                                Button(AppModel.operationLabel("image.generate")) { model.overrideRoute("image.generate") }
                                if model.latestImageTurn != nil || !model.attachments.isEmpty {
                                    Button(model.attachments.isEmpty ? AppModel.operationLabel("image.edit") : "Edit attached") { model.overrideRoute("image.edit") }
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
                            .fixedSize(horizontal: false, vertical: true)
                            .help("\(model.routePreview). Click to choose the workflow")
                        }
                        }
                        .guideTip(ModeTip.self, in: model.routePreview.isEmpty ? nil : Guide.shared.chat, edge: .bottom)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)

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
        // The window's minimum size is measured with the composer at its narrowest. Unbounded, its chips and
        // text then wrap one word per line and the minimum height outgrows short windows, which clips the
        // whole window (the reply box falls off the bottom). The smallest window's chat column is wider than this.
        // In a window as tall as the screen it also crashed the app on send (AppKit "more Update Constraints in
        // Window passes than there are views", from the chat's LazyVStack prefetch): the window could not grow
        // to the new minimum. Keep this and FlowLayout's real row widths (task 17, Bugs).
        .frame(minWidth: 300)
        .onAppear { focused = true }
        .onAppear { model.scheduleRoutePreview() }
        .onChange(of: model.draft) { model.scheduleRoutePreview() }
        .onChange(of: model.attachments) {
            focused = true
            model.scheduleRoutePreview()
        }
        .onChange(of: "\(model.generateChoice)|\(model.editChoice)") { model.scheduleRoutePreview() }
        .onChange(of: model.pendingLayout) { model.scheduleRoutePreview() }  // a layout can change the model
        .onChange(of: model.aspect) { model.scheduleRoutePreview() }  // the memory check counts the size
    }

    private func pickFiles() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.allowedContentTypes = [.image]
        if panel.runModal() == .OK {
            model.attach(files: panel.urls)
        }
    }

    /// All dropped files go to `attach` at once, so it can say how many did not fit (task 19).
    private func loadDropped(_ providers: [NSItemProvider]) {
        let group = DispatchGroup()
        let lock = NSLock()
        var urls: [(Int, URL)] = []
        for (index, provider) in providers.enumerated() {
            group.enter()
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                if let url { lock.withLock { urls.append((index, url)) } }
                group.leave()
            }
        }
        group.notify(queue: .main) {
            let files = lock.withLock { urls.sorted { $0.0 < $1.0 }.map(\.1) }
            MainActor.assumeIsolated { model.attach(files: files) }
        }
    }
}

/// A transparent view over the message field that takes only file drops (task 19). Clicks pass through
/// (hitTest is nil), and AppKit picks the drop target by registered drag types, so a plain text drop still
/// goes to the field below.
struct FileDropCatcher: NSViewRepresentable {
    @Binding var targeted: Bool
    let onDrop: ([URL]) -> Void

    func makeNSView(context: Context) -> CatcherView {
        let view = CatcherView()
        view.registerForDraggedTypes([.fileURL])
        return view
    }

    func updateNSView(_ view: CatcherView, context: Context) {
        view.onDrop = onDrop
        view.onTargeted = { targeted = $0 }
    }

    final class CatcherView: NSView {
        var onDrop: ([URL]) -> Void = { _ in }
        var onTargeted: (Bool) -> Void = { _ in }

        override func hitTest(_ point: NSPoint) -> NSView? { nil }

        private func files(_ info: NSDraggingInfo) -> [URL] {
            info.draggingPasteboard.readObjects(
                forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL] ?? []
        }

        override func draggingEntered(_ sender: NSDraggingInfo) -> NSDragOperation {
            guard !files(sender).isEmpty else { return [] }
            onTargeted(true)
            return .copy
        }

        override func draggingUpdated(_ sender: NSDraggingInfo) -> NSDragOperation {
            files(sender).isEmpty ? [] : .copy
        }

        override func draggingExited(_ sender: NSDraggingInfo?) { onTargeted(false) }

        override func performDragOperation(_ sender: NSDraggingInfo) -> Bool {
            onTargeted(false)
            let urls = files(sender)
            guard !urls.isEmpty else { return false }
            onDrop(urls)
            return true
        }
    }
}

struct ContextChip: View {
    let title: String
    let onRemove: () -> Void

    var body: some View {
        Button(action: onRemove) {
            HStack(spacing: 4) {
                Text(Self.short(title))
                    .lineLimit(1)
                Image(systemName: "xmark").font(.caption2.weight(.bold))
            }
            .font(.caption)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Theme.accent.opacity(0.18), in: RoundedRectangle(cornerRadius: 7))
        }
        .buttonStyle(.plain)
        // Its own width: a flexible frame here stretched every chip to 240 pt, far from its thumbnail.
        .fixedSize()
        .accessibilityLabel("\(title). Remove")
    }

    /// A long file name shortened in the middle ("IMG_2024…1234.png").
    static func short(_ title: String, max: Int = 32) -> String {
        guard title.count > max else { return title }
        let half = (max - 1) / 2
        return title.prefix(half) + "…" + title.suffix(max - 1 - half)
    }
}


/// Model choice in the message box, per task: New Images and Edits each open their list (Auto, or a
/// downloaded model that can do the task). The label names the choice in a few words.
/// The composer's model picker, after Claude's: a button naming the model, and a panel with one row per
/// model (its name, one line on what it is good at, a check mark on the chosen one). New images and edits
/// each have their own choice, switched at the top of the panel.
struct ModelMenu: View {
    @Environment(AppModel.self) private var model
    @Environment(\.openSettings) private var openSettings
    @State private var task: GenTask = .t2i

    var body: some View {
        @Bindable var model = model
        Button {
            model.showModelPicker.toggle()
        } label: {
            HStack(spacing: 3) {
                Text(label).lineLimit(1)
                Image(systemName: "chevron.down").font(.system(size: 8, weight: .semibold))
            }
            .font(.caption)
            .foregroundStyle(Theme.accent)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .fixedSize()
        .help("Models for new images and for edits")
        .accessibilityIdentifier("modelMenu")
        .accessibilityLabel("Models")
        .accessibilityValue("New images: \(name(for: .t2i)). Edits: \(name(for: .edit))")
        .popover(isPresented: $model.showModelPicker, arrowEdge: .top) { panel }
    }

    private var panel: some View {
        VStack(alignment: .leading, spacing: 0) {
            Picker("", selection: $task) {
                Text("New Images").tag(GenTask.t2i)
                Text("Edits").tag(GenTask.edit)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 10)
            .padding(.top, 10)
            .padding(.bottom, 6)

            VStack(spacing: 2) {
                ModelPickerRow(title: "Auto", detail: autoDetail, selected: model.chosenModel(for: task) == nil) {
                    choose("auto")
                }
                ForEach(model.choosableModels(for: task)) { m in
                    ModelPickerRow(title: model.menuName(m), detail: ModelHints.picker(m),
                                   selected: model.chosenModel(for: task)?.id == m.id) {
                        choose(m.id)
                    }
                }
            }
            .padding(.horizontal, 6)

            Divider().padding(.vertical, 6)

            ModelPickerRow(title: "Manage Models…", detail: nil, selected: false) {
                model.showModelPicker = false
                model.settingsTab = "models"
                openSettings()
            }
            .padding(.horizontal, 6)
            .padding(.bottom, 6)
        }
        .frame(width: 300)
    }

    private func choose(_ id: String) {
        model.setModelChoice(id, for: [task])
        model.showModelPicker = false
    }

    /// What Auto does: with the agent it picks per message; without it, the model it runs now.
    private var autoDetail: String {
        if model.promptAssistantInstalled { return "Picks the best model for each message" }
        // Only on Auto does the default name what Auto runs (otherwise it is the chosen model).
        let usual = model.chosenModel(for: task) == nil ? model.defaultModel(for: task).map { model.menuName($0) } : nil
        return usual.map { "Uses \($0)" } ?? "Picks a model for you"
    }

    static func title(_ task: GenTask) -> String { task == .edit ? "Edits" : "New Images" }

    private func name(for task: GenTask) -> String {
        model.chosenModel(for: task).map { model.menuName($0) } ?? "Auto"
    }

    /// "Auto"; one name when one model does both; otherwise both, shortened.
    private var label: String {
        let generate = model.chosenModel(for: .t2i), edit = model.chosenModel(for: .edit)
        switch (generate, edit) {
        case (nil, nil): return "Auto"
        case let (g?, e?) where g.id == e.id: return model.menuName(g)
        default: return "New: \(name(for: .t2i)) · Edit: \(name(for: .edit))"
        }
    }
}

/// One row of the model picker: name, a line on what it is good at, a check mark when chosen; highlighted
/// under the pointer like a menu item.
struct ModelPickerRow: View {
    let title: String
    let detail: String?
    let selected: Bool
    let action: () -> Void
    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            HStack(alignment: .center, spacing: 10) {
                VStack(alignment: .leading, spacing: 1) {
                    Text(title).font(.system(size: 13, weight: detail == nil ? .regular : .medium))
                    if let detail {
                        Text(detail).font(.system(size: 11)).foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                Spacer(minLength: 0)
                if selected {
                    Image(systemName: "checkmark").font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(Theme.accent)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, detail == nil ? 5 : 6)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(RoundedRectangle(cornerRadius: 6).fill(hovering ? Color.primary.opacity(0.08) : .clear))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}
