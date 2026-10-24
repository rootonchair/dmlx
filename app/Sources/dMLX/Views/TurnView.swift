import DMLXKit
import SwiftUI

struct TurnView: View {
    let turn: Turn

    var body: some View {
        switch turn.kind {
        case .user: UserBubble(turn: turn)
        case .notice: NoticeRow(text: turn.text)
        case .image: ImageTurnView(turn: turn)
        }
    }
}

struct UserBubble: View {
    @Environment(AppModel.self) private var model
    let turn: Turn
    @State private var editing = false
    @State private var draft = ""
    @State private var hovering = false
    @FocusState private var focused: Bool

    var answer: Turn? { model.imageTurn(answering: turn.id) }
    var prompt: String { turn.text }
    var canEdit: Bool { model.isEngineRunning && answer?.current.map { !$0.isActive } == true }
    var canRun: Bool {
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        return canEdit && !text.isEmpty && text != prompt
    }

    var body: some View {
        HStack {
            Spacer(minLength: 80)
            VStack(alignment: .trailing, spacing: 8) {
                if !turn.attachments.isEmpty {
                    HStack(spacing: 6) {
                        ForEach(turn.attachments, id: \.self) { id in
                            Thumbnail(resultId: id, size: 56)
                        }
                    }
                }
                if editing {
                    editor
                } else {
                    Text(prompt)
                        .textSelection(.enabled)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 9)
                        .background(.quaternary, in: RoundedRectangle(cornerRadius: 16))
                        .contextMenu {
                            if canEdit { Button("Edit Prompt…") { startEditing() } }
                        }
                    if canEdit {
                        // Under the bubble, inside the row's hover area, so the pointer can reach it.
                        Button { startEditing() } label: { Label("Edit", systemImage: "pencil") }
                            .buttonStyle(.borderless)
                            .controlSize(.small)
                            .foregroundStyle(.secondary)
                            .opacity(hovering ? 1 : 0)
                            .animation(.easeOut(duration: 0.12), value: hovering)
                            .help("Edit this prompt and replace the image with a new run")
                            .accessibilityLabel("Edit prompt")
                    }
                }
            }
        }
        // The whole row, spacer included, counts as hovering the message.
        .contentShape(Rectangle())
        .onHover { hovering = $0 }
    }

    private var editor: some View {
        VStack(alignment: .trailing, spacing: 8) {
            TextField("Prompt", text: $draft, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...8)
                .focused($focused)
                .onSubmit { if canRun { run() } }
                .onExitCommand { editing = false }
                .padding(.horizontal, 14)
                .padding(.vertical, 9)
                .background(.quaternary, in: RoundedRectangle(cornerRadius: 16))
                .overlay(RoundedRectangle(cornerRadius: 16).strokeBorder(Theme.accent.opacity(0.7)))
                .accessibilityLabel("Edit prompt")
            HStack(spacing: 8) {
                Button("Cancel") { editing = false }
                Button("Run") { run() }
                    .buttonStyle(.borderedProminent)
                    .disabled(!canRun)
            }
            .controlSize(.small)
        }
        .frame(maxWidth: 560)
        .onAppear { focused = true }
    }

    private func startEditing() {
        draft = prompt
        editing = true
    }

    private func run() {
        model.editPrompt(turn.id, to: draft)
        editing = false
    }
}

struct NoticeRow: View {
    let text: String

    var body: some View {
        Label(text, systemImage: "info.circle")
            .font(.callout)
            .foregroundStyle(.secondary)
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 10))
    }
}

struct Thumbnail: View {
    @Environment(AppModel.self) private var model
    let resultId: String
    var size: CGFloat

    var body: some View {
        Group {
            if let image = model.images.image(resultId) {
                Image(nsImage: image).resizable().scaledToFill()
            } else {
                Rectangle().fill(.quaternary)
            }
        }
        .frame(width: size, height: size)
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .accessibilityLabel("Attached image")
    }
}

struct ImageTurnView: View {
    @Environment(AppModel.self) private var model
    let turn: Turn

    var version: ImageVersion? { turn.current }
    /// A version that stopped before producing an image: no empty frame, only the error and its actions.
    var hasNoImage: Bool { version.map { !$0.isActive && $0.resultId == nil } ?? false }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                Text("Image \(turn.number)").font(.callout.weight(.semibold))
                Text(turn.route).font(.callout).foregroundStyle(.secondary)
            }
            if !hasNoImage {
                ResultImage(turn: turn)
                    .overlay(RoundedRectangle(cornerRadius: Theme.corner).strokeBorder(Color.secondary.opacity(0.2)))
            }

            if let v = version, v.status == .error || v.status == .cancelled {
                ErrorBox(turn: turn, version: v)
            }

            HStack(spacing: 6) {
                if turn.versions.count > 1 {
                    HStack(spacing: 4) {
                        ForEach(Array(turn.versions.enumerated()), id: \.element.id) { i, _ in
                            Button("v\(i + 1)") { model.pickVersion(turn.id, i) }
                                .buttonStyle(VersionButtonStyle(selected: i == turn.activeVersion))
                                .accessibilityAddTraits(i == turn.activeVersion ? .isSelected : [])
                        }
                    }
                    .padding(.trailing, 6)
                }
                if version?.isActive == true {
                    Button("Cancel") { model.cancel(turn.id) }
                } else if !hasNoImage {
                    Button("Variations") { model.variation(turn.id) }
                        .help("Same prompt, new seed")
                    Button("Attach") { model.attachResult(turn.id) }
                        .help("Attach this image to the next message")
                    Button("Crop / Draw") { model.openEditor(turn.id) }
                        .help("Prepare a non-destructive crop or paint overlay, then attach it")
                    Button {
                        model.save(turn.id)
                    } label: {
                        Image(systemName: "square.and.arrow.down")
                    }
                    .help("Save to Pictures/dMLX")
                    .accessibilityLabel("Save to Pictures")
                }
            }
            .buttonStyle(.bordered)
            .disabled(version == nil)

            if let v = version {
                Text(metaLine(v)).font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    func metaLine(_ v: ImageVersion) -> String {
        var parts = [model.modelName(v.model)]
        if v.task == .edit { parts.append("edit") }
        parts.append(v.size.replacingOccurrences(of: "x", with: "×"))
        if let seed = v.seed { parts.append("seed \(seed)") }
        if v.status == .done {
            if let s = v.seconds { parts.append(String(format: "%.1f s", s)) }
            parts.append(v.loads.isEmpty ? "model was already warm" : "loaded \(v.loads.map { AppModel.componentName($0) }.joined(separator: ", "))")
            if let label = AppModel.strategyLabel(v.strategy) { parts.append(label) }
        }
        return parts.joined(separator: " · ")
    }
}

struct VersionButtonStyle: ButtonStyle {
    var selected: Bool

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.callout.weight(selected ? .semibold : .regular))
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(
                RoundedRectangle(cornerRadius: 7)
                    .fill(selected ? Theme.accent : Color.clear))
            .overlay(RoundedRectangle(cornerRadius: 7).strokeBorder(selected ? Color.clear : Color.secondary.opacity(0.35)))
            .foregroundStyle(selected ? Color.black.opacity(0.85) : Color.primary)
            .opacity(configuration.isPressed ? 0.7 : 1)
    }
}

struct ResultImage: View {
    @Environment(AppModel.self) private var model
    let turn: Turn

    var body: some View {
        let v = turn.current
        ZStack(alignment: .bottomLeading) {
            if let rid = v?.resultId, v?.status == .done || v?.isActive == false, let image = model.images.image(rid) {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .onDrag { NSItemProvider(contentsOf: ImageCache.fileURL(rid)) ?? NSItemProvider() }
                    .accessibilityLabel(Text(v?.prompt ?? "Generated image"))
            } else {
                placeholder(v)
            }
            if let v, v.isActive {
                ProgressPill(version: v)
                    .padding(12)
            }
        }
        .frame(maxWidth: 520, alignment: .leading)
        .clipShape(RoundedRectangle(cornerRadius: Theme.corner))
    }

    @ViewBuilder
    func placeholder(_ v: ImageVersion?) -> some View {
        let aspect = aspectRatio(v?.size)
        ZStack {
            // While editing, show the source image blurred as a stand-in preview.
            if let src = v?.sourceIds.first, let image = model.images.image(src) {
                Image(nsImage: image).resizable().aspectRatio(contentMode: .fill).blur(radius: 18).opacity(0.6)
            } else {
                Rectangle().fill(.quaternary.opacity(0.6))
            }
            if v?.isActive == true {
                ProgressView().controlSize(.large)
            } else {
                Image(systemName: "photo").font(.largeTitle).foregroundStyle(.tertiary)
            }
        }
        .aspectRatio(aspect, contentMode: .fit)
        .frame(maxWidth: 520)
        .clipped()
    }

    func aspectRatio(_ size: String?) -> CGFloat {
        guard let size, size != "auto" else { return 1 }
        let parts = size.split(separator: "x").compactMap { Double($0) }
        guard parts.count == 2, parts[1] > 0 else { return 1 }
        return parts[0] / parts[1]
    }
}

struct ProgressPill: View {
    let version: ImageVersion

    var body: some View {
        HStack(spacing: 8) {
            if version.totalSteps > 0 && version.step > 0 {
                ProgressView(value: Double(version.step), total: Double(max(version.totalSteps, 1)))
                    .frame(width: 80)
            }
            Text(version.phase)
                .font(.caption)
                .monospacedDigit()
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(.regularMaterial, in: Capsule())
        .accessibilityElement(children: .combine)
    }
}

struct ErrorBox: View {
    @Environment(AppModel.self) private var model
    let turn: Turn
    let version: ImageVersion

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(version.errorMessage ?? "Something went wrong.", systemImage: version.status == .cancelled ? "stop.circle" : "exclamationmark.triangle")
                .font(.callout)
                .fixedSize(horizontal: false, vertical: true)
            HStack {
                Button("Retry") { model.retry(turn.id, force: false) }
                if version.errorCode == "insufficient_memory" {
                    Button("Try anyway") { model.retry(turn.id, force: true) }
                        .help("Runs in low-memory mode and stops automatically if your Mac gets slow")
                }
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
        .padding(10)
        .frame(maxWidth: 520, alignment: .leading)
        .background(Color.orange.opacity(0.12), in: RoundedRectangle(cornerRadius: 10))
    }
}
