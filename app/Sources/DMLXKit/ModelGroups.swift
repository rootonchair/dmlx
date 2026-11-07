import Foundation

/// One build of a model (4-bit, 8-bit, full size): the same model, differing only in download size and quality.
public struct ModelVariant: Identifiable, Hashable, Sendable {
    public var repo: String
    /// Bits of the build; nil for full size.
    public var quantization: Int?
    /// The engine knows it (built in, or added from the app): it can be downloaded, chosen and removed.
    public var model: EngineModel?
    /// The community catalog lists it: it can be added.
    public var suggestion: CatalogSuggestion?

    public var id: String { repo }
    public var isAdded: Bool { model != nil }
    public var isDownloaded: Bool { model?.downloaded ?? false }
    public var isDownloading: Bool { model?.download.status == "downloading" }
    /// Smallest first; full size last.
    var order: Int { quantization ?? 99 }
}

/// One model (family and size, like "FLUX.2 Klein 4B") with its builds as versions, the way Ollama lists
/// a model with its tags. Base (undistilled) and KV checkpoints are models of their own (the engine groups them).
public struct ModelGroup: Identifiable, Hashable, Sendable {
    /// What dMLX ships and recommends: small, and close to the full model's quality.
    public static let recommendedBits = 4

    public var id: String
    public var name: String
    public var tasks: [String]
    public var license: String
    /// Smallest build first.
    public var variants: [ModelVariant]
    /// Downloads of all its builds on Hugging Face (catalog order).
    public var downloads: Int

    public func supports(_ task: GenTask) -> Bool { tasks.contains(task.rawValue) }
    public var added: [ModelVariant] { variants.filter(\.isAdded) }
    public var isAdded: Bool { variants.contains(where: \.isAdded) }
    public var isDownloaded: Bool { variants.contains(where: \.isDownloaded) }
    public var fitsThisMac: Bool {
        variants.compactMap { $0.model?.fits_this_mac ?? $0.suggestion?.fits_this_mac }.first ?? true
    }

    /// The version to offer first: one on this Mac, else one the engine knows, else the 4-bit build, else the
    /// smallest larger one, else the largest.
    public var defaultVariant: ModelVariant? {
        variants.first(where: \.isDownloaded)
            ?? variants.first(where: \.isAdded)
            ?? variants.first { $0.quantization == Self.recommendedBits }
            ?? variants.first { ($0.quantization ?? 99) > Self.recommendedBits }
            ?? variants.last
    }

    public func variant(_ repo: String?) -> ModelVariant? {
        variants.first { $0.repo == repo }
    }
}

public enum ModelCatalog {
    /// The engine's models and the catalog's builds, one group per model. A catalog build with the same bits
    /// as one the engine already has (the built-in Qwen Image 2.1 comes from another repo) is left out.
    /// Groups keep the catalog's order (most downloaded first); models it does not list come after.
    public static func groups(models: [EngineModel], suggestions: [CatalogSuggestion]) -> [ModelGroup] {
        var groups: [String: ModelGroup] = [:]
        var order: [String] = []

        func add(_ variant: ModelVariant, group id: String, name: String, tasks: [String], license: String, downloads: Int) {
            if groups[id] == nil {
                groups[id] = ModelGroup(id: id, name: name, tasks: tasks, license: license, variants: [], downloads: 0)
                order.append(id)
            }
            groups[id]!.variants.append(variant)
            groups[id]!.downloads += downloads
            for task in tasks where !groups[id]!.tasks.contains(task) { groups[id]!.tasks.append(task) }
        }

        let byRepo = Dictionary(models.compactMap { m in m.repo.map { ($0.lowercased(), m) } }, uniquingKeysWith: { a, _ in a })
        var placed = Set<String>()
        for s in suggestions {
            let id = s.group ?? s.repo
            if let m = byRepo[s.repo.lowercased()] {
                placed.insert(m.id)
                let variant = ModelVariant(repo: m.repo ?? s.repo, quantization: m.quantization ?? s.quantization, model: m, suggestion: s)
                add(variant, group: m.group ?? id, name: m.group_name ?? s.group_name ?? m.name, tasks: m.tasks,
                    license: m.license, downloads: s.downloads)
            } else {
                add(ModelVariant(repo: s.repo, quantization: s.quantization, suggestion: s), group: id,
                    name: s.group_name ?? s.label ?? s.repo, tasks: s.tasks ?? [], license: s.license ?? "unknown",
                    downloads: s.downloads)
            }
        }
        for m in models where !placed.contains(m.id) {
            let id = m.group ?? m.id
            let variant = ModelVariant(repo: m.repo ?? m.id, quantization: m.quantization, model: m)
            if var group = groups[id] {
                // The engine's build replaces a catalog build of the same bits from another repo.
                group.variants.removeAll { $0.model == nil && $0.quantization == m.quantization }
                groups[id] = group
            }
            add(variant, group: id, name: m.group_name ?? m.name, tasks: m.tasks, license: m.license, downloads: 0)
        }
        return order.compactMap { id in
            guard var group = groups[id] else { return nil }
            // A model's own license wins over the catalog's card (built-in models name theirs exactly).
            if let licensed = group.variants.first(where: { $0.model != nil })?.model { group.license = licensed.license }
            group.variants.sort { ($0.order, $0.repo) < ($1.order, $1.repo) }
            return group
        }
    }
}
