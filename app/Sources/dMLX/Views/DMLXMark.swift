import SwiftUI

/// The dMLX logo ("Still Water"): a half sun over its reflection in still water.
/// It uses the same 64-unit grid as docs/brand/dmlx-mark.svg. At 32 pt and smaller it draws
/// docs/brand/dmlx-mark-small.svg (two thicker lines) so it stays clear.
struct DMLXMark: View {
    var size: CGFloat = 28
    /// The color of the sun. The water uses the primary text color.
    var sun: Color = Theme.logo

    var body: some View {
        Canvas { ctx, canvas in
            let u = canvas.width / 64
            func pill(_ x: CGFloat, _ y: CGFloat, _ w: CGFloat, _ h: CGFloat) -> Path {
                Path(roundedRect: CGRect(x: x * u, y: y * u, width: w * u, height: h * u), cornerRadius: h * u / 2)
            }
            let small = size <= 32
            let r: CGFloat = small ? 26 : 24
            var half = Path()
            half.addArc(center: CGPoint(x: 32 * u, y: 30 * u), radius: r * u,
                        startAngle: .degrees(180), endAngle: .degrees(0), clockwise: false)
            half.closeSubpath()
            ctx.fill(half, with: .color(sun))

            var water = Path()
            if small {
                water.addPath(pill(6, 36, 52, 9))
                water.addPath(pill(18, 50, 28, 9))
            } else {
                water.addPath(pill(8, 34, 48, 6))
                water.addPath(pill(13, 43, 17, 6))
                water.addPath(pill(34, 43, 17, 6))
                water.addPath(pill(22, 52, 20, 6))
            }
            ctx.fill(water, with: .foreground)
        }
        .foregroundStyle(.primary)
        .frame(width: size, height: size)
        .accessibilityHidden(true)
    }
}

#Preview {
    HStack(spacing: 24) {
        DMLXMark(size: 16)
        DMLXMark(size: 32)
        DMLXMark(size: 64)
        DMLXMark(size: 128)
    }
    .padding()
}
