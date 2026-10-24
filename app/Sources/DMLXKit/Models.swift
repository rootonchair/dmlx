import Foundation

/// A chat session ("creation"). Turns are stored separately and loaded per session.
public struct Session: Codable, Identifiable, Hashable, Sendable {
    public var id: String
    public var title: String
    public var created: Date
    public var updated: Date

    public init(id: String = UUID().uuidString, title: String = "New creation", created: Date = .now, updated: Date = .now) {
        self.id = id
        self.title = title
        self.created = created
        self.updated = updated
    }
}

public enum GenTask: String, Codable, Sendable {
    case t2i
    case edit
}

public enum VersionStatus: String, Codable, Sendable {
    case queued, rendering, done, error, cancelled
}

/// One generated image (or one attempt at it). Versions of the same turn share a prompt.
public struct ImageVersion: Codable, Identifiable, Hashable, Sendable {
    public var id: String
    public var status: VersionStatus
    public var jobId: String?
    public var resultId: String?
    public var prompt: String
    public var model: String
    public var task: GenTask
    public var operation: String?
    public var size: String
    public var seed: Int?
    public var sourceIds: [String]
    public var step: Int
    public var totalSteps: Int
    public var phase: String
    public var strategy: String?
    public var loads: [String]
    public var seconds: Double?
    public var errorCode: String?
    public var errorMessage: String?
    public var label: String?
    public var workflowRoute: WorkflowRoute?
    public var mediaInputs: [MediaInput]?
    public var workflowPrompt: PromptSpec?

    public init(
        id: String = UUID().uuidString, status: VersionStatus = .queued, jobId: String? = nil, resultId: String? = nil,
        prompt: String, model: String, task: GenTask, operation: String? = nil, size: String, seed: Int? = nil, sourceIds: [String] = [],
        step: Int = 0, totalSteps: Int = 0, phase: String = "queued", strategy: String? = nil, loads: [String] = [],
        seconds: Double? = nil, errorCode: String? = nil, errorMessage: String? = nil, label: String? = nil,
        workflowRoute: WorkflowRoute? = nil, mediaInputs: [MediaInput]? = nil, workflowPrompt: PromptSpec? = nil
    ) {
        self.id = id
        self.status = status
        self.jobId = jobId
        self.resultId = resultId
        self.prompt = prompt
        self.model = model
        self.task = task
        self.operation = operation
        self.size = size
        self.seed = seed
        self.sourceIds = sourceIds
        self.step = step
        self.totalSteps = totalSteps
        self.phase = phase
        self.strategy = strategy
        self.loads = loads
        self.seconds = seconds
        self.errorCode = errorCode
        self.errorMessage = errorMessage
        self.label = label
        self.workflowRoute = workflowRoute
        self.mediaInputs = mediaInputs
        self.workflowPrompt = workflowPrompt
    }

    public var isActive: Bool { status == .queued || status == .rendering }
}

public enum TurnKind: String, Codable, Sendable {
    case user, image, notice
}

/// A row in the thread. Image turns carry their versions; `activeVersion` is the one shown.
public struct Turn: Codable, Identifiable, Hashable, Sendable {
    public var id: String
    public var sessionId: String
    public var index: Int
    public var kind: TurnKind
    public var text: String
    public var number: Int  // "Image 3"
    public var route: String
    public var versions: [ImageVersion]
    public var activeVersion: Int
    public var attachments: [String]  // result ids attached to a user turn
    public var attachmentInputs: [MediaInput]?
    public var created: Date

    public init(
        id: String = UUID().uuidString, sessionId: String, index: Int, kind: TurnKind, text: String = "",
        number: Int = 0, route: String = "", versions: [ImageVersion] = [], activeVersion: Int = 0,
        attachments: [String] = [], attachmentInputs: [MediaInput]? = nil, created: Date = .now
    ) {
        self.id = id
        self.sessionId = sessionId
        self.index = index
        self.kind = kind
        self.text = text
        self.number = number
        self.route = route
        self.versions = versions
        self.activeVersion = activeVersion
        self.attachments = attachments
        self.attachmentInputs = attachmentInputs
        self.created = created
    }

    public var current: ImageVersion? {
        versions.indices.contains(activeVersion) ? versions[activeVersion] : versions.last
    }
}

/// What the engine reports about a model (subset of /v1/models).
public struct EngineModel: Codable, Identifiable, Hashable, Sendable {
    public struct Download: Codable, Hashable, Sendable {
        public var status: String
        public var total_bytes: Int
        public var done_bytes: Int
        public var progress: Double
        public var error: String?
    }

    public var id: String
    public var name: String
    public var license: String
    public var tasks: [String]
    public var max_references: Int
    public var min_ram_gb: Int
    public var size_gb: Double?
    public var downloaded: Bool
    public var resident: Bool
    public var pinned: Bool
    public var fits_this_mac: Bool
    public var download: Download
    public var repo: String?
    public var user_added: Bool?
    public var auto_select: Bool?
    public var experimental: Bool?
    public var license_url: String?
    public var license_acceptance_id: String?

    public func supports(_ task: GenTask) -> Bool { tasks.contains(task.rawValue) }
    /// Added from the app (removable); built-in models can only have their files deleted.
    public var isUserAdded: Bool { user_added ?? false }
    public var autoSelectable: Bool { auto_select ?? true }
    public var isExperimental: Bool { experimental ?? false }
    public var requiresLicenseAcceptance: Bool { license_acceptance_id != nil }
}

/// What a Hugging Face repo is, as the engine reads it before anything is downloaded.
public struct ModelInspection: Codable, Hashable, Sendable {
    public var repo: String
    public var id: String
    public var name: String
    public var family: String
    public var tasks: [String]
    public var license: String
    public var permissive_license: Bool
    public var quantization: Int?
    public var size_gb: Double
    public var url: String
    public var already_added: String?

    public func supports(_ task: GenTask) -> Bool { tasks.contains(task.rawValue) }
}

public struct CatalogSuggestion: Codable, Hashable, Sendable, Identifiable {
    public var repo: String
    public var label: String?
    public var downloads: Int
    public var added: Bool
    public var id: String { repo }
}
