import DMLXKit
import SwiftUI

struct OnboardingView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss

    var ramText: String {
        let gb = model.memory?.ram_tier_gb ?? Int(ProcessInfo.processInfo.physicalMemory / 1_073_741_824)
        return "\(gb) GB"
    }

    var ready: Bool { model.defaultModel(for: .t2i) != nil }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("Welcome to dMLX")
                .font(.system(size: 28, weight: .regular, design: .serif))
            Text("Everything runs on this Mac (\(ramText) memory). Download the models once; after that dMLX works offline. dMLX watches memory so your Mac stays responsive while it works.")
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            if !model.isEngineRunning {
                HStack {
                    ProgressView().controlSize(.small)
                    Text(model.engineStatusText).foregroundStyle(.secondary)
                }
            }

            VStack(spacing: 10) {
                ForEach(model.models.filter(\.fits_this_mac)) { m in
                    ModelRow(model: m)
                }
            }

            HStack {
                Text("Models are Apache-2.0 licensed and stored in your Hugging Face cache.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Button(ready ? "Start creating" : "Later") { dismiss() }
                    .keyboardShortcut(.defaultAction)
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding(24)
        .task {
            while !_Concurrency.Task.isCancelled {
                await model.refreshModels()
                try? await _Concurrency.Task.sleep(for: .seconds(1))
            }
        }
    }
}

struct ModelRow: View {
    @Environment(AppModel.self) private var app
    let model: EngineModel

    var role: String {
        model.supports(.edit) ? "Creates and edits images" : "Creates images"
    }

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(model.name).font(.headline)
                Text("\(role) · \(model.license) · \(sizeText)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            if model.downloaded {
                Label("Downloaded", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                    .font(.callout)
            } else if model.download.status == "downloading" {
                VStack(alignment: .trailing, spacing: 4) {
                    ProgressView(value: model.download.progress).frame(width: 140)
                    Text(String(format: "%.1f of %.1f GB", Double(model.download.done_bytes) / 1e9, Double(model.download.total_bytes) / 1e9))
                        .font(.caption2)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
            } else {
                VStack(alignment: .trailing, spacing: 4) {
                    Button("Download") { app.download(model) }
                    if let error = model.download.error {
                        Text(error).font(.caption2).foregroundStyle(.red).lineLimit(2)
                    }
                }
            }
        }
        .padding(12)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 10))
    }

    var sizeText: String {
        if let gb = model.size_gb, gb > 0 { return String(format: "%.1f GB", gb) }
        if model.download.total_bytes > 0 { return String(format: "%.1f GB", Double(model.download.total_bytes) / 1e9) }
        return "about 5 GB"
    }
}
