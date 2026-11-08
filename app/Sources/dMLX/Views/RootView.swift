import DMLXKit
import SwiftUI
import TipKit

struct RootView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var model = model
        NavigationSplitView {
            SidebarView()
                // The sidebar column otherwise reports a minimum taller than short
                // windows, and the split view is then clipped top and bottom (losing
                // the composer and the status card).
                .frame(minHeight: 0, maxHeight: .infinity)
                .navigationSplitViewColumnWidth(min: 200, ideal: 240, max: 320)
        } detail: {
            ChatView()
        }
        .sheet(isPresented: $model.showOnboarding) {
            OnboardingView()
                .environment(model)
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
        // The chat tips start once the setup is closed and an image model is ready.
        .onAppear { updateGuide() }
        .onChange(of: model.showOnboarding) { updateGuide() }
        // The default image model, not the model list: it depends on the capabilities, which load after it.
        .onChange(of: model.defaultModel(for: .t2i)?.id) { updateGuide() }
    }

    private func updateGuide() {
        Guide.shared.setReady(!model.showOnboarding && model.defaultModel(for: .t2i) != nil)
    }
}

struct SidebarView: View {
    @Environment(AppModel.self) private var model
    @State private var sessionToDelete: Session?

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

            // Inline, not a popover: at the top of the window a popover has no room above the button.
            if let tip = Guide.shared.chat.currentTip as? ChatsTip {
                TipView(tip, arrowEdge: .top) { _ in Guide.shared.next(tip) }
                    .padding(.horizontal, 12)
                    .padding(.bottom, 8)
            }

            List(selection: Binding(
                get: { model.currentSessionId },
                set: { if let id = $0 { model.select(id) } })
            ) {
                Section("Recent") {
                    ForEach(model.sessions) { session in
                        SidebarChatRow(
                            session: session,
                            onDelete: { sessionToDelete = session }
                        )
                            .tag(session.id)
                    }
                }
            }
            .listStyle(.sidebar)

            StatusCard()
                .padding(12)
        }
        .confirmationDialog(
            "Delete “\(sessionToDelete?.title ?? "this chat")”?",
            isPresented: Binding(
                get: { sessionToDelete != nil },
                set: { if !$0 { sessionToDelete = nil } }
            ),
            titleVisibility: .visible
        ) {
            if let sessionToDelete {
                Button("Delete Chat", role: .destructive) {
                    model.delete(session: sessionToDelete.id)
                    self.sessionToDelete = nil
                }
            }
            Button("Cancel", role: .cancel) { sessionToDelete = nil }
        } message: {
            Text("This removes the conversation and its editor drafts from this Mac. Generated images are kept because other chats may reference them.")
        }
    }
}

private struct SidebarChatRow: View {
    let session: Session
    let onDelete: () -> Void

    var body: some View {
        HStack(spacing: 6) {
            Text(session.title)
                .lineLimit(1)
                .frame(maxWidth: .infinity, alignment: .leading)
                .help(session.title)

            Menu {
                Button("Delete Chat…", systemImage: "trash", role: .destructive) {
                    onDelete()
                }
            } label: {
                Image(systemName: "ellipsis")
                    .font(.body.weight(.semibold))
                    .frame(width: 24, height: 24)
                    .contentShape(Rectangle())
            }
            .menuIndicator(.hidden)
            .menuStyle(.borderlessButton)
            .foregroundStyle(.secondary)
            .help("Chat options")
            .accessibilityLabel("Options for \(session.title)")
        }
        .contentShape(Rectangle())
        .contextMenu {
            Button("Delete Chat…", role: .destructive) {
                onDelete()
            }
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
