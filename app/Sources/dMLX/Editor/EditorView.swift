import AppKit
import DMLXKit
import SwiftUI
import TipKit

private enum EditorTool: String, CaseIterable, Identifiable {
    case crop = "Crop"
    case paint = "Paint"
    case erase = "Erase"
    case layout = "Layout"
    var id: String { rawValue }
}

struct EditorView: View {
    @State private var document: EditorDocument
    @State private var tool: EditorTool = .crop
    @State private var brushColor = Color.red
    @State private var brushWidth = 0.02
    @State private var history: [EditorDocument] = []
    @State private var future: [EditorDocument] = []
    @State private var dragStart: NormalizedPoint?
    @State private var gestureSnapshot: EditorDocument?
    @State private var comparing = false
    // Layout (task 17): only for models that place things from boxes.
    @State private var boxes: [LayoutBox]
    @State private var selectedBox: String?
    @State private var boxHistory: [[LayoutBox]] = []
    @State private var boxFuture: [[LayoutBox]] = []
    @FocusState private var describing: Bool

    let source: NSImage
    let isExporting: Bool
    let onCancel: () -> Void
    let onApply: (EditorDocument) -> Void
    /// The result's layout when its model takes one; nil hides the Layout tool.
    let initialLayout: [LayoutBox]?
    /// A blank canvas from the composer's ▭ Layout: no image yet, so only Layout works.
    let blankLayout: Bool
    let placesByWords: Bool
    let onLayout: ([LayoutBox]) -> Void

    init(
        document: EditorDocument, source: NSImage, isExporting: Bool, startTool: String? = nil,
        layout: [LayoutBox]? = nil, blankLayout: Bool = false, placesByWords: Bool = false,
        onCancel: @escaping () -> Void, onApply: @escaping (EditorDocument) -> Void,
        onLayout: @escaping ([LayoutBox]) -> Void = { _ in }
    ) {
        _document = State(initialValue: document)
        let start = startTool.flatMap(EditorTool.init(rawValue:))
        _tool = State(initialValue: blankLayout ? .layout : (start == .layout && layout == nil ? .crop : start ?? .crop))
        _boxes = State(initialValue: layout ?? [])
        self.source = source
        self.isExporting = isExporting
        self.onCancel = onCancel
        self.onApply = onApply
        self.initialLayout = layout
        self.blankLayout = blankLayout
        self.placesByWords = placesByWords
        self.onLayout = onLayout
    }

    private var tools: [EditorTool] {
        if blankLayout { return [.layout] }
        return EditorTool.allCases.filter { $0 != .layout || initialLayout != nil }
    }

    var body: some View {
        VStack(spacing: 0) {
            // The same in every tool, and narrow enough for the sheet's minimum width: Paint's and Erase's brush
            // controls live in the footer.
            HStack(spacing: 12) {
                Picker("Tool", selection: $tool) {
                    ForEach(tools) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                // One tool (the blank canvas) at a fixed 65 pt squeezed its "Tool" label into a tall, empty bar.
                .frame(width: tools.count > 1 ? CGFloat(65 * tools.count) : nil)
                .fixedSize()
                .help(blankLayout ? "No image yet: draw where things go, then make the image" : "")
                Spacer(minLength: 12)
                Button { undo() } label: { Image(systemName: "arrow.uturn.backward") }
                    .disabled(tool == .layout ? boxHistory.isEmpty : history.isEmpty)
                Button { redo() } label: { Image(systemName: "arrow.uturn.forward") }
                    .disabled(tool == .layout ? boxFuture.isEmpty : future.isEmpty)
                if !blankLayout {
                    Button("Original") { comparing.toggle() }
                        .buttonStyle(.borderedProminent)
                        .tint(comparing ? .secondary : .accentColor)
                }
            }
            .padding()

            // First time in the editor: what the tools do.
            TipView(Guide.shared.editor)
                .padding(.horizontal)
                .padding(.bottom, 10)

            Group {
                if tool == .layout {
                    ZStack {
                        if !blankLayout {
                            GeometryReader { geometry in
                                let rect = EditorGeometry.fittedRect(
                                    imageWidth: document.baseWidth, imageHeight: document.baseHeight, in: geometry.size)
                                Image(nsImage: source).resizable()
                                    .frame(width: rect.width, height: rect.height)
                                    .position(x: rect.midX, y: rect.midY)
                            }
                        } else {
                            GeometryReader { geometry in
                                let rect = EditorGeometry.fittedRect(
                                    imageWidth: document.baseWidth, imageHeight: document.baseHeight, in: geometry.size)
                                Rectangle().fill(Color.white.opacity(0.08))
                                    .frame(width: rect.width, height: rect.height)
                                    .position(x: rect.midX, y: rect.midY)
                            }
                        }
                        if !comparing {
                            LayoutCanvas(
                                boxes: $boxes, selected: $selectedBox, imageWidth: document.baseWidth,
                                imageHeight: document.baseHeight, placesByWords: placesByWords,
                                onBeginChange: rememberBoxes, onDescribe: { describing = true })
                        }
                    }
                } else {
                    EditorCanvas(
                        source: source, document: comparing ? originalView : document,
                        showEdits: !comparing, tool: tool,
                        onChanged: updateGesture, onEnded: finishGesture)
                }
            }
                .frame(minWidth: 320, minHeight: 220, maxHeight: .infinity)
                .background(Color.black.opacity(0.88))

            // The footer: the tool's controls, then the buttons. One row when it fits; otherwise the buttons
            // get their own row, so they are never cut off or behind a scroll.
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 8) {
                    toolControls
                    Spacer(minLength: 12)
                    footerButtons
                }
                VStack(alignment: .leading, spacing: 10) {
                    HStack(spacing: 8) { toolControls }
                    HStack(spacing: 8) {
                        Spacer(minLength: 0)
                        footerButtons
                    }
                }
            }
            .buttonStyle(.bordered)
            .padding()
        }
        .frame(minWidth: 520, idealWidth: 900, minHeight: 420, idealHeight: 680)
        .interactiveDismissDisabled(isExporting)
        .onReceive(NotificationCenter.default.publisher(for: .dmlxHarnessSelectBox)) { note in
            let label = note.object as? String
            selectedBox = (label == "*" ? boxes.first : boxes.first { $0.label == label })?.id
        }
    }

    /// The footer's left side: Crop's presets, the brush (color, size) and a hint for Paint / Erase, or the Layout bar.
    @ViewBuilder private var toolControls: some View {
        if tool == .layout {
            LayoutBar(boxes: $boxes, selected: $selectedBox, placesByWords: placesByWords,
                      blank: blankLayout, describing: $describing, onBeginChange: rememberBoxes)
        } else if tool == .crop {
            Text("Crop:").foregroundStyle(.secondary).fixedSize()
            Button("Full") { setCrop(.full) }.fixedSize()
            Button("Square") { setAspect(1) }.fixedSize()
            Button("3:2") { setAspect(3.0 / 2.0) }.fixedSize()
            Button("2:3") { setAspect(2.0 / 3.0) }.fixedSize()
            Text("or drag a crop rectangle").font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        } else {
            ColorPicker("Color", selection: $brushColor, supportsOpacity: true)
                .labelsHidden()
                .fixedSize()
            Slider(value: $brushWidth, in: 0.003...0.1)
                .frame(width: 110)
                .help("Brush size")
            Text(tool == .erase ? "Drag to erase paint. The original image is never erased." : "Drag over the image to paint.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// The footer's right side, never squeezed: Reset, Cancel and the primary action.
    @ViewBuilder private var footerButtons: some View {
        HStack(spacing: 8) {
            if tool == .layout {
                Button("Reset") { rememberBoxes(); boxes = initialLayout ?? []; selectedBox = nil }
                Button("Cancel", action: onCancel)
                Button(blankLayout ? "Use layout" : "Redraw with layout") {
                    onLayout(LayoutBox.sendable(boxes))
                }
                .buttonStyle(.borderedProminent)
                .disabled(LayoutBox.sendable(boxes).isEmpty && blankLayout)
                .help(blankLayout ? "Attach the boxes to your next message"
                      : "A new version of this image with your boxes (the same seed when no box moved)")
            } else {
                Button("Reset") { mutate { $0.crop = .full; $0.strokes = [] } }
                Button("Cancel", action: onCancel)
                Button {
                    var saved = document
                    saved.updated = .now
                    onApply(saved)
                } label: {
                    if isExporting { ProgressView().controlSize(.small) } else { Text("Attach edited image") }
                }
                .buttonStyle(.borderedProminent)
                .disabled(isExporting)
            }
        }
        .fixedSize()
    }

    private var originalView: EditorDocument {
        EditorDocument(
            baseResultId: document.baseResultId, baseWidth: document.baseWidth,
            baseHeight: document.baseHeight)
    }

    private func mutate(_ change: (inout EditorDocument) -> Void) {
        history.append(document)
        future = []
        change(&document)
    }

    private func setCrop(_ crop: NormalizedRect) { mutate { $0.crop = crop } }

    private func setAspect(_ target: Double) {
        let imageAspect = Double(document.baseWidth) / Double(document.baseHeight)
        if target <= imageAspect {
            let width = target / imageAspect
            setCrop(NormalizedRect(x: (1 - width) / 2, y: 0, width: width, height: 1))
        } else {
            let height = imageAspect / target
            setCrop(NormalizedRect(x: 0, y: (1 - height) / 2, width: 1, height: height))
        }
    }

    private func updateGesture(_ point: NormalizedPoint) {
        if gestureSnapshot == nil {
            gestureSnapshot = document
            dragStart = point
            if tool != .crop {
                let ns = NSColor(brushColor).usingColorSpace(.deviceRGB) ?? .red
                document.strokes.append(PaintStroke(
                    points: [point], width: brushWidth,
                    color: PaintColor(red: ns.redComponent, green: ns.greenComponent, blue: ns.blueComponent, alpha: ns.alphaComponent),
                    erases: tool == .erase))
            }
        } else if tool != .crop {
            document.strokes[document.strokes.count - 1].points.append(point)
        }
        if tool == .crop, let start = dragStart {
            document.crop = NormalizedRect(
                x: min(start.x, point.x), y: min(start.y, point.y),
                width: abs(point.x - start.x), height: abs(point.y - start.y))
        }
    }

    private func finishGesture(_ point: NormalizedPoint) {
        updateGesture(point)
        if let snapshot = gestureSnapshot, snapshot != document {
            history.append(snapshot)
            future = []
        }
        gestureSnapshot = nil
        dragStart = nil
    }

    private func rememberBoxes() {
        boxHistory.append(boxes)
        boxFuture = []
    }

    private func undo() {
        if tool == .layout {
            guard let previous = boxHistory.popLast() else { return }
            boxFuture.append(boxes)
            boxes = previous
            return
        }
        guard let previous = history.popLast() else { return }
        future.append(document)
        document = previous
    }

    private func redo() {
        if tool == .layout {
            guard let next = boxFuture.popLast() else { return }
            boxHistory.append(boxes)
            boxes = next
            return
        }
        guard let next = future.popLast() else { return }
        history.append(document)
        document = next
    }
}

private struct EditorCanvas: View {
    let source: NSImage
    let document: EditorDocument
    let showEdits: Bool
    let tool: EditorTool
    let onChanged: (NormalizedPoint) -> Void
    let onEnded: (NormalizedPoint) -> Void

    var body: some View {
        GeometryReader { geometry in
            let rect = EditorGeometry.fittedRect(
                imageWidth: document.baseWidth, imageHeight: document.baseHeight, in: geometry.size)
            ZStack(alignment: .topLeading) {
                Image(nsImage: source)
                    .resizable()
                    .frame(width: rect.width, height: rect.height)
                    .position(x: rect.midX, y: rect.midY)
                if showEdits {
                    Canvas { context, _ in drawOverlay(context, imageRect: rect) }
                }
            }
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 0)
                .onChanged { value in if let point = EditorGeometry.normalize(value.location, in: rect) { onChanged(point) } }
                .onEnded { value in if let point = EditorGeometry.normalize(value.location, in: rect) { onEnded(point) } })
            .help(tool == .crop ? "Drag to select a crop" : "Drag on the image")
        }
    }

    private func drawOverlay(_ context: GraphicsContext, imageRect: CGRect) {
        // Paint goes on its own layer so an erase stroke clears paint, as the exported image does
        // (EditorRenderer); the photo below is never erased.
        context.drawLayer { layer in
            for stroke in document.strokes where !stroke.points.isEmpty {
                var path = Path()
                let points = stroke.points.map {
                    CGPoint(x: imageRect.minX + $0.x * imageRect.width, y: imageRect.minY + $0.y * imageRect.height)
                }
                path.move(to: points[0])
                for point in points.dropFirst() { path.addLine(to: point) }
                var pen = layer
                pen.blendMode = stroke.erases ? .clear : .normal
                let color = Color(red: stroke.color.red, green: stroke.color.green, blue: stroke.color.blue,
                                  opacity: stroke.color.alpha)
                pen.stroke(
                    path, with: .color(stroke.erases ? .black : color),
                    style: StrokeStyle(lineWidth: max(1, stroke.width * min(imageRect.width, imageRect.height)),
                                       lineCap: .round, lineJoin: .round))
            }
        }
        let crop = CGRect(
            x: imageRect.minX + document.crop.x * imageRect.width,
            y: imageRect.minY + document.crop.y * imageRect.height,
            width: document.crop.width * imageRect.width,
            height: document.crop.height * imageRect.height)
        var dim = Path()
        dim.addRect(imageRect)
        dim.addRect(crop)
        context.fill(dim, with: .color(.black.opacity(0.48)), style: FillStyle(eoFill: true))
        context.stroke(Path(crop), with: .color(.white), style: StrokeStyle(lineWidth: 2, dash: [7, 5]))
    }
}
