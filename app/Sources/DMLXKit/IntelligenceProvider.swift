import Foundation
import Security

/// Non-secret routing configuration. The API key is stored separately in the
/// user's Keychain and is never written to defaults, history, or workflow data.
public enum RouterProviderKind: String, CaseIterable, Codable, Sendable {
    case disabled
    case openAICompatible = "openai-compatible"
    case localLLM = "local-llm"
}

public struct RouterProviderSettings: Equatable, Sendable {
    public static let recommendedLocalRevision = "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"

    public var kind: RouterProviderKind
    public var baseURL: String
    public var model: String
    public var timeout: Double
    public var localModelPath: String
    public var localRevision: String

    public init(
        kind: RouterProviderKind = .openAICompatible,
        baseURL: String = "", model: String = "", timeout: Double = 15,
        localModelPath: String = "", localRevision: String = ""
    ) {
        self.kind = kind
        self.baseURL = baseURL
        self.model = model
        self.timeout = timeout
        self.localModelPath = localModelPath
        self.localRevision = localRevision
    }

    public static func load() -> Self {
        let defaults = UserDefaults.standard
        let storedKind = defaults.string(forKey: "routerProvider.kind").flatMap(RouterProviderKind.init(rawValue:))
        let baseURL = defaults.string(forKey: "routerProvider.baseURL") ?? ""
        let model = defaults.string(forKey: "routerProvider.model") ?? ""
        let checkpoint = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("dMLX/checkpoints/qwen3-4b-instruct-2507-4bit").path
        var isDirectory: ObjCBool = false
        let hasRecommendedLocalModel = FileManager.default.fileExists(
            atPath: checkpoint, isDirectory: &isDirectory) && isDirectory.boolValue
        let storedLocalPath = defaults.string(forKey: "routerProvider.localModelPath")
        let localPath = storedLocalPath ?? (hasRecommendedLocalModel ? checkpoint : "")
        let localRevision = defaults.string(forKey: "routerProvider.localRevision")
            ?? (hasRecommendedLocalModel ? recommendedLocalRevision : "")
        return Self(
            kind: storedKind ?? (hasRecommendedLocalModel
                ? .localLLM : (baseURL.isEmpty && model.isEmpty ? .disabled : .openAICompatible)),
            baseURL: baseURL, model: model,
            timeout: defaults.object(forKey: "routerProvider.timeout") as? Double ?? 15,
            localModelPath: localPath,
            localRevision: localRevision)
    }

    public func validate() throws {
        if kind == .disabled { return }
        if kind == .localLLM {
            var isDirectory: ObjCBool = false
            guard !localRevision.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                  localModelPath.hasPrefix("/"),
                  FileManager.default.fileExists(atPath: localModelPath, isDirectory: &isDirectory),
                  isDirectory.boolValue else {
                throw RouterSettingsError.invalid("Choose an existing local checkpoint folder and enter its pinned revision.")
            }
            return
        }
        guard !baseURL.isEmpty, !model.isEmpty,
              let url = URL(string: baseURL), let scheme = url.scheme?.lowercased(), let host = url.host else {
            throw RouterSettingsError.invalid("Enter both an absolute base URL and a model ID.")
        }
        let loopback = ["localhost", "127.0.0.1", "::1"].contains(host.lowercased())
        guard scheme == "https" || (scheme == "http" && loopback) else {
            throw RouterSettingsError.invalid("Remote providers must use HTTPS. HTTP is allowed only on this Mac.")
        }
        guard timeout >= 0.1, timeout <= 120 else {
            throw RouterSettingsError.invalid("Timeout must be between 0.1 and 120 seconds.")
        }
    }

    public func save(apiKey: String) throws {
        try validate()
        let defaults = UserDefaults.standard
        defaults.set(kind.rawValue, forKey: "routerProvider.kind")
        defaults.set(baseURL, forKey: "routerProvider.baseURL")
        defaults.set(model, forKey: "routerProvider.model")
        defaults.set(timeout, forKey: "routerProvider.timeout")
        defaults.set(localModelPath, forKey: "routerProvider.localModelPath")
        defaults.set(localRevision, forKey: "routerProvider.localRevision")
        try RouterSecretStore.save(kind == .openAICompatible ? apiKey : "")
    }
}

public enum RouterSettingsError: LocalizedError {
    case invalid(String)
    case keychain(OSStatus)

    public var errorDescription: String? {
        switch self {
        case .invalid(let message): return message
        case .keychain(let status):
            return SecCopyErrorMessageString(status, nil) as String? ?? "Keychain error \(status)"
        }
    }
}

public enum RouterSecretStore {
    private static let service = "app.dmlx.workflow-router"
    private static let account = "api-key"

    public static func load() -> String {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data else { return "" }
        return String(decoding: data, as: UTF8.self)
    }

    public static func save(_ value: String) throws {
        let identity: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        SecItemDelete(identity as CFDictionary)
        guard !value.isEmpty else { return }
        var item = identity
        item[kSecValueData as String] = Data(value.utf8)
        item[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        let status = SecItemAdd(item as CFDictionary, nil)
        guard status == errSecSuccess else { throw RouterSettingsError.keychain(status) }
    }
}

public struct PromptProviderSettings: Equatable, Sendable {
    public var baseURL: String
    public var model: String
    public var timeout: Double

    public init(baseURL: String = "", model: String = "", timeout: Double = 30) {
        self.baseURL = baseURL
        self.model = model
        self.timeout = timeout
    }

    public static func load() -> Self {
        let defaults = UserDefaults.standard
        return Self(
            baseURL: defaults.string(forKey: "promptProvider.baseURL") ?? "",
            model: defaults.string(forKey: "promptProvider.model") ?? "",
            timeout: defaults.object(forKey: "promptProvider.timeout") as? Double ?? 30)
    }

    public func validate() throws {
        if baseURL.isEmpty && model.isEmpty { return }
        guard !baseURL.isEmpty, !model.isEmpty,
              let url = URL(string: baseURL), let scheme = url.scheme?.lowercased(), let host = url.host else {
            throw RouterSettingsError.invalid("Enter both an absolute base URL and a model ID.")
        }
        let loopback = ["localhost", "127.0.0.1", "::1"].contains(host.lowercased())
        guard scheme == "https" || (scheme == "http" && loopback) else {
            throw RouterSettingsError.invalid("Remote providers must use HTTPS. HTTP is allowed only on this Mac.")
        }
        guard timeout >= 0.1, timeout <= 120 else {
            throw RouterSettingsError.invalid("Timeout must be between 0.1 and 120 seconds.")
        }
    }

    public func save(apiKey: String) throws {
        try validate()
        let defaults = UserDefaults.standard
        defaults.set(baseURL, forKey: "promptProvider.baseURL")
        defaults.set(model, forKey: "promptProvider.model")
        defaults.set(timeout, forKey: "promptProvider.timeout")
        try PromptSecretStore.save(apiKey)
    }
}

public enum PromptSecretStore {
    private static let service = "app.dmlx.prompt-provider"
    private static let account = "api-key"

    public static func load() -> String {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data else { return "" }
        return String(decoding: data, as: UTF8.self)
    }

    public static func save(_ value: String) throws {
        let identity: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        SecItemDelete(identity as CFDictionary)
        guard !value.isEmpty else { return }
        var item = identity
        item[kSecValueData as String] = Data(value.utf8)
        item[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        let status = SecItemAdd(item as CFDictionary, nil)
        guard status == errSecSuccess else { throw RouterSettingsError.keychain(status) }
    }
}
