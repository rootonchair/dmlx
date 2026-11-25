import DMLXKit
import SwiftUI

/// Crop / Draw → Layout (task 17): boxes over the image for models that place things from a layout.
/// Drag a box to move it, pull its corner to resize, drag on empty space to add one. Boxes the user
/// touched turn blue: they are theirs and stay put on later turns. Nothing here shows JSON.
struct LayoutCanvas: View {
    @Binding var boxes: [LayoutBox]
    @Binding var selected: String?
    let imageWidth: Int
    let imageHeight: Int
    /// FIBO places things by words, not exact boxes: each box shows the word it will get.
    let placesByWords: Bool
    let onBeginChange: () -> Void
    let onDescribe: () -> Void

    private enum Drag { case move(String, NormalizedRect), resize(String, NormalizedRect), draw(String) }
    @State private var drag: Drag?
    @State private var start: NormalizedPoint?

    var body: some View {
        GeometryReader { geometry in
            let rect = EditorGeometry.fittedRect(imageWidth: imageWidth, imageHeight: imageHeight, in: geometry.size)
            ZStack(alignment: .topLeading) {
                ForEach(boxes) { b in
                    BoxView(box: b, selected: selected == b.id, placesByWords: placesByWords)
                        .frame(width: max(b.rect.width * rect.width, 1), height: max(b.rect.height * rect.height, 1))
                        .position(x: rect.minX + (b.rect.x + b.rect.width / 2) * rect.width,
                                  y: rect.minY + (b.rect.y + b.rect.height / 2) * rect.height)
                }
            }
            .frame(width: geometry.size.width, height: geometry.size.height)
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 2)
                .onChanged { value in changed(value, in: rect) }
                .onEnded { _ in ended() })
            .simultaneousGesture(SpatialTapGesture(count: 2).onEnded { value in
                if let id = hit(value.location, in: rect) { selected = id; onDescribe() }
            })
            .simultaneousGesture(SpatialTapGesture(count: 1).onEnded { value in
                selected = hit(value.location, in: rect)
            })
            .help("Drag a box to move it, pull its corner to resize, or drag on empty space to add one")
        }
    }

    private func point(_ location: CGPoint, in rect: CGRect) -> NormalizedPoint {
        NormalizedPoint(x: (location.x - rect.minX) / rect.width, y: (location.y - rect.minY) / rect.height)
    }

    /// The top-most box under a point (the last drawn is on top).
    private func hit(_ location: CGPoint, in rect: CGRect) -> String? {
        let p = point(location, in: rect)
        return boxes.last { b in
            let r = b.rect
            return p.x >= r.x && p.x <= r.x + r.width && p.y >= r.y && p.y <= r.y + r.height
        }?.id
    }

    private func nearCorner(_ location: CGPoint, of b: LayoutBox, in rect: CGRect) -> Bool {
        let r = b.rect
        let corner = CGPoint(x: rect.minX + (r.x + r.width) * rect.width, y: rect.minY + (r.y + r.height) * rect.height)
        return hypot(location.x - corner.x, location.y - corner.y) < 12
    }

    private func changed(_ value: DragGesture.Value, in rect: CGRect) {
        let p = point(value.location, in: rect)
        if drag == nil {
            onBeginChange()
            let origin = point(value.startLocation, in: rect)
            start = origin
            if let b = boxes.last(where: { nearCorner(value.startLocation, of: $0, in: rect) }) {
                drag = .resize(b.id, b.rect)
            } else if let id = hit(value.startLocation, in: rect), let b = boxes.first(where: { $0.id == id }) {
                drag = .move(id, b.rect)
            } else {
                let b = LayoutBox(rect: NormalizedRect(x: origin.x, y: origin.y, width: 0.001, height: 0.001))
                boxes.append(b)
                drag = .draw(b.id)
            }
            if case let .move(id, _) = drag { selected = id }
            if case let .resize(id, _) = drag { selected = id }
            if case let .draw(id) = drag { selected = id }
        }
        guard let start, let drag else { return }
        switch drag {
        case let .move(id, original):
            let dx = p.x - start.x, dy = p.y - start.y
            let x = min(max(original.x + dx, 0), 1 - original.width)
            let y = min(max(original.y + dy, 0), 1 - original.height)
            update(id) { $0.place(NormalizedRect(x: x, y: y, width: original.width, height: original.height)) }
        case let .resize(id, original):
            update(id) { $0.place(NormalizedRect(x: original.x, y: original.y,
                                                 width: max(p.x - original.x, 0.03), height: max(p.y - original.y, 0.03))) }
        case let .draw(id):
            update(id) { $0.place(NormalizedRect(x: min(start.x, p.x), y: min(start.y, p.y),
                                                 width: abs(p.x - start.x), height: abs(p.y - start.y))) }
        }
    }

    private func ended() {
        // A click-sized box is a mis-drag, not a box.
        if case let .draw(id) = drag, let b = boxes.first(where: { $0.id == id }), b.rect.width < 0.03 || b.rect.height < 0.03 {
            boxes.removeAll { $0.id == id }
            selected = nil
        }
        drag = nil
        start = nil
    }

    private func update(_ id: String, _ change: (inout LayoutBox) -> Void) {
        guard let i = boxes.firstIndex(where: { $0.id == id }) else { return }
        change(&boxes[i])
    }
}

/// One box: dashed for text, solid for a picture, blue (the accent) when the user placed it.
struct BoxView: View {
    let box: LayoutBox
    var selected = false
    var placesByWords = false
    var readOnly = false

    var body: some View {
        let color: Color = box.byUser ? Theme.accent : .white
        ZStack(alignment: .topLeading) {
            Rectangle()
                .fill(color.opacity(selected ? 0.22 : 0.1))
            Rectangle()
                .strokeBorder(color.opacity(0.95), style: StrokeStyle(lineWidth: selected ? 2.5 : 1.5,
                                                                      dash: box.kind == .text ? [6, 4] : []))
            Text(title)
                .font(.caption2.weight(.semibold))
                .lineLimit(1)
                .padding(.horizontal, 4).padding(.vertical, 1)
                .background(.black.opacity(0.65), in: RoundedRectangle(cornerRadius: 3))
                .foregroundStyle(.white)
                .padding(3)
            if !readOnly {
                RoundedRectangle(cornerRadius: 2).fill(.white)
                    .frame(width: 9, height: 9)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomTrailing)
                    .padding(2)
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(box.kind == .text ? "Text" : "Picture") box: \(title)")
    }

    private var title: String {
        let name = box.label.isEmpty ? (box.kind == .text ? "Text" : "Unnamed") : box.label
        return placesByWords ? "\(name) · \(LayoutWords.location(box))" : name
    }
}

/// The bottom bar of the Layout tool, where Crop shows its presets: the selected box's kind, name or
/// exact words, and description, in plain words.
struct LayoutBar: View {
    @Binding var boxes: [LayoutBox]
    @Binding var selected: String?
    let placesByWords: Bool
    let blank: Bool
    var describing: FocusState<Bool>.Binding
    let onBeginChange: () -> Void

    var body: some View {
        if let i = boxes.firstIndex(where: { $0.id == selected }) {
            // One row in a wide sheet; in a narrow one the description moves to a second row and the
            // fields shrink, so nothing is cut off.
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 8) {
                    boxHeader(i)
                    nameField(i).frame(width: 170)
                    descField(i).frame(width: 280)
                    letAgentButton(i)
                    deleteButton(i)
                }
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 8) {
                        boxHeader(i)
                        nameField(i).frame(minWidth: 100, maxWidth: .infinity)
                        deleteButton(i)
                    }
                    HStack(spacing: 8) {
                        descField(i).frame(minWidth: 120, maxWidth: .infinity)
                        letAgentButton(i)
                    }
                }
            }
        } else {
            // Wraps, so the long hint never pushes Redraw with layout out of the sheet.
            Text(hint).font(.caption).foregroundStyle(.secondary)
                .frame(maxWidth: 420, alignment: .leading)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder private func boxHeader(_ i: Int) -> some View {
        Text("Box:").foregroundStyle(.secondary).fixedSize()
        Picker("Kind", selection: kindBinding(i)) {
            Text("Picture").tag(LayoutBox.Kind.obj)
            Text("Text").tag(LayoutBox.Kind.text)
        }
        .pickerStyle(.segmented)
        .labelsHidden()
        .frame(width: 140)
    }

    private func nameField(_ i: Int) -> some View {
        TextField(boxes[i].kind == .text ? "Exact words to draw" : "Name, e.g. saxophonist", text: labelBinding(i))
            .textFieldStyle(.roundedBorder)
    }

    private func descField(_ i: Int) -> some View {
        TextField(boxes[i].kind == .text ? "Lettering style (optional): the agent picks one"
                  : "What it looks like (optional): the agent writes it", text: descBinding(i))
            .textFieldStyle(.roundedBorder)
            .focused(describing)
    }

    @ViewBuilder private func letAgentButton(_ i: Int) -> some View {
        if boxes[i].descByUser {
            Button("Let the agent write it") {
                onBeginChange()
                boxes[i].desc = ""
                boxes[i].descByUser = false
            }
            .help("Clear your description; the agent writes one")
            .fixedSize()
        }
    }

    private func deleteButton(_ i: Int) -> some View {
        Button("Delete") {
            onBeginChange()
            boxes.remove(at: i)
            selected = nil
        }
        .fixedSize()
    }

    private var hint: String {
        if blank { return "Drag on the canvas to draw a box for each thing, then name it. The agent writes whatever you leave empty." }
        let words = placesByWords ? " This model gets each box as a place word." : ""
        return "Click a box to name or describe it. Drag to move it, or drag on empty space to add one. Blue boxes are yours.\(words)"
    }

    private func kindBinding(_ i: Int) -> Binding<LayoutBox.Kind> {
        Binding(get: { boxes[i].kind }, set: { onBeginChange(); boxes[i].kind = $0; boxes[i].byUser = true })
    }

    private func labelBinding(_ i: Int) -> Binding<String> {
        Binding(get: { boxes[i].label }, set: { value in
            // Renaming a box clears only a description the agent wrote: it described the old name.
            if !boxes[i].descByUser && value != boxes[i].label { boxes[i].desc = "" }
            boxes[i].label = value
            boxes[i].byUser = true
        })
    }

    private func descBinding(_ i: Int) -> Binding<String> {
        Binding(get: { boxes[i].desc }, set: { value in
            boxes[i].desc = value
            boxes[i].descByUser = !value.trimmingCharacters(in: .whitespaces).isEmpty
            boxes[i].byUser = true
        })
    }
}

/// The boxes over a result in the chat (Show boxes / Hide boxes).
struct LayoutOverlay: View {
    let boxes: [LayoutBox]
    let imageWidth: Int
    let imageHeight: Int
    var placesByWords = false

    var body: some View {
        GeometryReader { geometry in
            let rect = EditorGeometry.fittedRect(imageWidth: imageWidth, imageHeight: imageHeight, in: geometry.size)
            ForEach(boxes) { b in
                BoxView(box: b, placesByWords: placesByWords, readOnly: true)
                    .frame(width: max(b.rect.width * rect.width, 1), height: max(b.rect.height * rect.height, 1))
                    .position(x: rect.minX + (b.rect.x + b.rect.width / 2) * rect.width,
                              y: rect.minY + (b.rect.y + b.rect.height / 2) * rect.height)
            }
        }
        .allowsHitTesting(false)
    }
}
