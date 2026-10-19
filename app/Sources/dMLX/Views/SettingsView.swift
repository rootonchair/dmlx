import AppKit
import DMLXKit
import SwiftUI

struct SettingsView: View {
    var body: some View {
        TabView {
            ModelsSettings().tabItem { Label("Models", systemImage: "square.stack.3d.up") }
            MemorySettings().tabItem { Label("Memory", systemImage: "memorychip") }
            EngineSettings().tabItem { Label("Engine", systemImage: "gearshape.2") }
        }
        .padding(16)
    }
}

struct ModelsSettings: View {
    @Environment(AppModel.self) private var model
    @State private var adding = false

    var body: some View {
        @Bindable var model = model
        ScrollView {
            VStack(spacing: 10) {
                if model.choosableModels.count > 1 {
                    Picker("Model", selection: $model.modelChoice) {
                        Text("Auto").tag("auto")
                        ForEach(model.choosableModels) { m in Text(m.name).tag(m.id) }
                    }
                    .help("Also in the message box. Auto uses one model for new images and edits, which keeps follow-ups fastest.")
                    .padding(.bottom, 6)
                }
                ForEach(model.models) { m in
                    ModelRow(model: m, removable: true)
                }
                HStack {
                    Text("Add FLUX.2 Klein or Z-Image models from Hugging Face.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Add Model…") { adding = true }
                        .disabled(!model.isEngineRunning)
                }
                .padding(.top, 4)
                if model.models.isEmpty {
                    Text("Waiting for the engine…").foregroundStyle(.secondary)
                }
            }
        }
        .task { await model.refreshModels() }
        .sheet(isPresented: $adding) {
            AddModelView().environment(model)
        }
    }
}

struct MemorySettings: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        Form {
            if let m = model.memory {
                LabeledContent("Memory tier", value: "\(m.ram_tier_gb) GB")
                LabeledContent("Available now", value: String(format: "%.1f GB", m.memory.system.available_gb))
                LabeledContent("dMLX may use", value: String(format: "%.1f GB", max(m.memory.headroom_gb, 0)))
                LabeledContent("Kept free for macOS", value: String(format: "%.1f GB", m.memory.reserve_gb))
                LabeledContent("Memory pressure", value: m.memory.system.pressure.capitalized)
                LabeledContent("Swap in use", value: String(format: "%.1f GB", m.memory.system.swap_used_gb))
                if m.memory.degraded {
                    Text("Recently under memory pressure: using low-memory mode for a few minutes.")
                        .font(.caption).foregroundStyle(.orange)
                }
                Section("In memory") {
                    if m.memory.resident.isEmpty {
                        Text("Nothing loaded").foregroundStyle(.secondary)
                    }
                    ForEach(m.memory.resident, id: \.self) { r in
                        LabeledContent("\(model.modelName(r.model)) · \(AppModel.componentName(r.component))",
                                       value: String(format: "%.2f GB", r.gb))
                    }
                }
            } else {
                Text("Waiting for the engine…").foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .task {
            while !_Concurrency.Task.isCancelled {
                await model.refreshMemory()
                try? await _Concurrency.Task.sleep(for: .seconds(1))
            }
        }
    }
}

struct EngineSettings: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(model.engineStatusText).font(.callout)
                Spacer()
                Button("Restart engine") { model.engine.restart() }
                Button("Show log file") { NSWorkspace.shared.activateFileViewerSelecting([model.engine.logFile]) }
            }
            ScrollView {
                Text(model.engine.logLines.suffix(300).joined(separator: "\n"))
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 8))
        }
    }
}
