import AppKit
import DMLXKit
import SwiftUI

private enum EditorTool: String, CaseIterable, Identifiable {
    case crop = "Crop"
    case paint = "Paint"
    case erase = "Erase"
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

    let source: NSImage
    let isExporting: Bool
    let onCancel: () -> Void
    let onApply: (EditorDocument) -> Void

    init(
        document: EditorDocument, source: NSImage, isExporting: Bool,
        onCancel: @escaping () -> Void, onApply: @escaping (EditorDocument) -> Void
    ) {
        _document = State(initialValue: document)
        self.source = source
        self.isExporting = isExporting
        self.onCancel = onCancel
        self.onApply = onApply
    }

    var body: some View {
        VStack(spacing: 0) {
            ScrollView(.horizontal) {
                HStack(spacing: 12) {
                    Picker("Tool", selection: $tool) {
                        ForEach(EditorTool.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 260)

                    if tool != .crop {
                        ColorPicker("Color", selection: $brushColor, supportsOpacity: true)
                            .labelsHidden()
                        Slider(value: $brushWidth, in: 0.003...0.1)
                            .frame(width: 110)
                            .help("Brush size")
                    }
                    Spacer()
                    Button { undo() } label: { Image(systemName: "arrow.uturn.backward") }
                        .disabled(history.isEmpty)
                    Button { redo() } label: { Image(systemName: "arrow.uturn.forward") }
                        .disabled(future.isEmpty)
                    Button("Original") { comparing.toggle() }
                        .buttonStyle(.borderedProminent)
                        .tint(comparing ? .secondary : .accentColor)
                }
                .fixedSize(horizontal: true, vertical: false)
            }
            .scrollIndicators(.hidden)
            .padding()

            EditorCanvas(
                source: source, document: comparing ? originalView : document,
                showEdits: !comparing, tool: tool,
                onChanged: updateGesture, onEnded: finishGesture)
                .frame(minWidth: 320, minHeight: 220, maxHeight: .infinity)
                .background(Color.black.opacity(0.88))

            ScrollView(.horizontal) {
                HStack(spacing: 8) {
                    if tool == .crop {
                        Text("Crop:").foregroundStyle(.secondary)
                        Button("Full") { setCrop(.full) }
                        Button("Square") { setAspect(1) }
                        Button("3:2") { setAspect(3.0 / 2.0) }
                        Button("2:3") { setAspect(2.0 / 3.0) }
                        Text("or drag a crop rectangle").font(.caption).foregroundStyle(.secondary)
                    } else {
                        Text(tool == .erase ? "Drag to erase paint. The original image is never erased." : "Drag over the image to paint.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
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
                .fixedSize(horizontal: true, vertical: false)
            }
            .scrollIndicators(.hidden)
            .buttonStyle(.bordered)
            .padding()
        }
        .frame(minWidth: 520, idealWidth: 900, minHeight: 420, idealHeight: 680)
        .interactiveDismissDisabled(isExporting)
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

    private func undo() {
        guard let previous = history.popLast() else { return }
        future.append(document)
        document = previous
    }

    private func redo() {
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
        for stroke in document.strokes where !stroke.points.isEmpty {
            var path = Path()
            let points = stroke.points.map {
                CGPoint(x: imageRect.minX + $0.x * imageRect.width, y: imageRect.minY + $0.y * imageRect.height)
            }
            path.move(to: points[0])
            for point in points.dropFirst() { path.addLine(to: point) }
            let color = Color(
                red: stroke.color.red, green: stroke.color.green,
                blue: stroke.color.blue, opacity: stroke.erases ? 0.35 : stroke.color.alpha)
            context.stroke(
                path, with: .color(stroke.erases ? .white.opacity(0.55) : color),
                style: StrokeStyle(lineWidth: max(1, stroke.width * min(imageRect.width, imageRect.height)), lineCap: .round, lineJoin: .round))
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
