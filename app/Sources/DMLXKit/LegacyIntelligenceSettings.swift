import Foundation
import Security

/// What the removed router and prompt-enhancement settings left behind (before 2026-11-08 the app had a
/// "Semantic workflow routing" and a "Prompt enhancement" section; the agent replaced both). Clearing it is
/// cheap and idempotent, so it runs at every start.
public enum LegacyIntelligenceSettings {
    static let defaultsKeys = [
        "routerProvider.kind", "routerProvider.baseURL", "routerProvider.model", "routerProvider.timeout",
        "routerProvider.localModelPath", "routerProvider.localRevision",
        "promptProvider.baseURL", "promptProvider.model", "promptProvider.timeout", "promptEnhancementMode",
    ]
    static let keychainServices = ["app.dmlx.workflow-router", "app.dmlx.prompt-provider"]

    /// The router's local Qwen checkpoint (2.1 GB), in the default data folder whatever DMLX_HOME says,
    /// where the old settings looked for it.
    public static var routerCheckpoint: URL {
        Store.defaultDataDirectory().appendingPathComponent("checkpoints/qwen3-4b-instruct-2507-4bit")
    }

    public static func clear(defaults: UserDefaults = .standard, deleteCheckpoint: Bool = true) {
        for key in defaultsKeys { defaults.removeObject(forKey: key) }
        for service in keychainServices {
            let identity: [String: Any] = [
                kSecClass as String: kSecClassGenericPassword,
                kSecAttrService as String: service,
                kSecAttrAccount as String: "api-key",
            ]
            SecItemDelete(identity as CFDictionary)
        }
        guard deleteCheckpoint else { return }
        try? FileManager.default.removeItem(at: routerCheckpoint)
        let checkpoints = routerCheckpoint.deletingLastPathComponent()
        if (try? FileManager.default.contentsOfDirectory(atPath: checkpoints.path))?.isEmpty == true {
            try? FileManager.default.removeItem(at: checkpoints)
        }
    }
}
