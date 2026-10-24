import DMLXKit
import SwiftUI

struct ChatView: View {
    @Environment(AppModel.self) private var model

    private let starters = [
        "A ceramic mug on a linen tablecloth, soft window light",
        "A fox logo, flat vector, two colors",
        "A small wooden rowboat on a still mountain lake at golden hour, 35mm film photo",
    ]

    var body: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 22) {
                        if model.turns.isEmpty {
                            EmptyStateView(starters: starters)
                        }
                        ForEach(model.turns) { turn in
                            TurnView(turn: turn).id(turn.id)
                        }
                        Color.clear.frame(height: 1).id("bottom")
                    }
                    .frame(maxWidth: 780)
                    .padding(.horizontal, 24)
                    .padding(.top, 24)
                    .frame(maxWidth: .infinity)
                }
                .onChange(of: model.turns.count) {
                    withAnimation(.easeOut(duration: 0.25)) { proxy.scrollTo("bottom", anchor: .bottom) }
                }
                .onChange(of: model.currentSessionId) {
                    proxy.scrollTo("bottom", anchor: .bottom)
                }
            }
            // The transcript must absorb window-height changes. Without an
            // explicit flexible height, an active 520-point image placeholder
            // can take its ideal height and compress the composer to zero.
            .frame(maxHeight: .infinity)
            ComposerView()
                .frame(maxWidth: 780)
                .padding(.horizontal, 24)
                .padding(.bottom, 16)
                .padding(.top, 8)
                .frame(maxWidth: .infinity)
                .background(.background)
                .fixedSize(horizontal: false, vertical: true)
                .layoutPriority(1)
        }
        .navigationTitle(model.sessions.first(where: { $0.id == model.currentSessionId })?.title ?? "New creation")
        .sheet(isPresented: Binding(
            get: { model.editorDocument != nil && model.editorImage != nil },
            set: { if !$0 { model.closeEditor() } }
        )) {
            if let document = model.editorDocument, let image = model.editorImage {
                EditorView(
                    document: document, source: image, isExporting: model.isExportingEditor,
                    onCancel: { model.closeEditor() }, onApply: { model.applyEditor($0) })
            }
        }
    }
}

struct EmptyStateView: View {
    @Environment(AppModel.self) private var model
    let starters: [String]

    var body: some View {
        VStack(spacing: 18) {
            Text("What should we make?")
                .font(.system(size: 36, weight: .regular, design: .serif))
            Text("Describe an image. Follow up in plain words to change it. Every result is kept as a version.")
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 460)
            FlowLayout(spacing: 10) {
                ForEach(starters, id: \.self) { s in
                    Button(s) { model.send(s) }
                        .buttonStyle(.bordered)
                        .controlSize(.large)
                        .disabled(!model.isEngineRunning)
                }
            }
            .frame(maxWidth: 620)
        }
        .frame(maxWidth: .infinity)
        .padding(.top, 90)
    }
}

/// Simple wrapping layout for chips.
struct FlowLayout: Layout {
    var spacing: CGFloat = 8

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? 600
        var x: CGFloat = 0, y: CGFloat = 0, rowHeight: CGFloat = 0, maxX: CGFloat = 0
        for view in subviews {
            let size = view.sizeThatFits(ProposedViewSize(width: width, height: nil))
            if x > 0 && x + size.width > width {
                x = 0
                y += rowHeight + spacing
                rowHeight = 0
            }
            x += size.width + spacing
            maxX = max(maxX, x - spacing)
            rowHeight = max(rowHeight, size.height)
        }
        return CGSize(width: min(maxX, width), height: y + rowHeight)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let rows = rowsFor(width: bounds.width, subviews: subviews)
        var y = bounds.minY
        for row in rows {
            let rowWidth = row.reduce(0) { $0 + $1.1.width } + spacing * CGFloat(max(row.count - 1, 0))
            var x = bounds.minX + (bounds.width - rowWidth) / 2
            let height = row.map(\.1.height).max() ?? 0
            for (view, size) in row {
                view.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
                x += size.width + spacing
            }
            y += height + spacing
        }
    }

    private func rowsFor(width: CGFloat, subviews: Subviews) -> [[(LayoutSubview, CGSize)]] {
        var rows: [[(LayoutSubview, CGSize)]] = [[]]
        var x: CGFloat = 0
        for view in subviews {
            let size = view.sizeThatFits(ProposedViewSize(width: width, height: nil))
            if x > 0 && x + size.width > width {
                rows.append([])
                x = 0
            }
            rows[rows.count - 1].append((view, size))
            x += size.width + spacing
        }
        return rows
    }
}
