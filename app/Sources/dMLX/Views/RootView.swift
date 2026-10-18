import DMLXKit
import SwiftUI

struct RootView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var model = model
        NavigationSplitView {
            SidebarView()
                .navigationSplitViewColumnWidth(min: 200, ideal: 240, max: 320)
        } detail: {
            ChatView()
        }
        .sheet(isPresented: $model.showOnboarding) {
            OnboardingView()
                .environment(model)
                .frame(width: 560)
        }
        .overlay(alignment: .bottom) {
            if let toast = model.toast {
                Text(toast)
                    .font(.callout)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .background(.regularMaterial, in: Capsule())
                    .padding(.bottom, 110)
                    .transition(.opacity)
                    .accessibilityAddTraits(.updatesFrequently)
            }
        }
        .animation(.easeOut(duration: 0.2), value: model.toast)
    }
}

struct SidebarView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button {
                model.newSession()
            } label: {
                Label("New creation", systemImage: "plus")
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.vertical, 6)
            }
            .buttonStyle(.bordered)
            .controlSize(.large)
            .padding(12)

            List(selection: Binding(
                get: { model.currentSessionId },
                set: { if let id = $0 { model.select(id) } })
            ) {
                Section("Recent") {
                    ForEach(model.sessions) { session in
                        Text(session.title)
                            .lineLimit(1)
                            .tag(session.id)
                            .contextMenu {
                                Button("Delete", role: .destructive) { model.delete(session: session.id) }
                            }
                    }
                }
            }
            .listStyle(.sidebar)

            StatusCard()
                .padding(12)
        }
    }
}

struct StatusCard: View {
    @Environment(AppModel.self) private var model

    var dotColor: Color {
        switch model.engineState {
        case .running: return model.memory?.memory.system.pressure == "critical" ? .red : .green
        case .starting: return .orange
        case .stopped, .failed: return .red
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Circle().fill(dotColor).frame(width: 8, height: 8).accessibilityHidden(true)
                Text(model.isEngineRunning ? "Ready" : (model.engineState == .starting ? "Starting" : "Not running"))
                    .font(.callout.weight(.medium))
            }
            Text(model.engineStatusText)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            if case .failed = model.engineState {
                Button("Restart engine") { model.engine.restart() }.controlSize(.small)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
        .accessibilityElement(children: .combine)
    }
}
