import Foundation

public enum MediaType: String, Codable, Hashable, Sendable {
    case image, video, mask
}

public enum InputRole: String, Codable, Hashable, Sendable {
    case source, reference, composite, mask, sketch
}

public struct MediaInput: Codable, Identifiable, Hashable, Sendable {
    public var id: String
    public var media_type: MediaType
    public var role: InputRole?
    public var order: Int
    public var metadata: [String: String]

    public init(
        id: String, mediaType: MediaType = .image, role: InputRole? = nil,
        order: Int = 0, metadata: [String: String] = [:]
    ) {
        self.id = id
        self.media_type = mediaType
        self.role = role
        self.order = order
        self.metadata = metadata
    }
}

public enum PromptEnhancementMode: String, Codable, Hashable, Sendable {
    case off, ask, auto
}

public struct PromptSpec: Codable, Hashable, Sendable {
    public var user: String
    public var positive: String
    public var negative: String?
    public var enhancement_mode: PromptEnhancementMode
    public var compiler: String

    public init(
        user: String, positive: String? = nil, negative: String? = nil,
        enhancementMode: PromptEnhancementMode = .off, compiler: String = "passthrough@1"
    ) {
        self.user = user
        self.positive = positive ?? user
        self.negative = negative
        self.enhancement_mode = enhancementMode
        self.compiler = compiler
    }
}

public struct WorkflowParameters: Codable, Hashable, Sendable {
    public var size: String?
    public var width: Int?
    public var height: Int?
    public var steps: Int?
    public var seed: Int?
    public var n: Int?
    public var strategy: String?
    public var force: Bool?
    public var scale: Int?
    public var softness: Double?

    public init(
        size: String? = nil, width: Int? = nil, height: Int? = nil, steps: Int? = nil,
        seed: Int? = nil, n: Int? = nil, strategy: String? = nil, force: Bool? = nil,
        scale: Int? = nil, softness: Double? = nil
    ) {
        self.size = size
        self.width = width
        self.height = height
        self.steps = steps
        self.seed = seed
        self.n = n
        self.strategy = strategy
        self.force = force
        self.scale = scale
        self.softness = softness
    }
}

public struct WorkflowStep: Codable, Identifiable, Hashable, Sendable {
    public var id: String
    public var operation: String
    public var executor: String?
    public var inputs: [MediaInput]
    public var prompt: PromptSpec
    public var parameters: WorkflowParameters

    public init(
        id: String = "step_1", operation: String, executor: String? = nil,
        inputs: [MediaInput] = [], prompt: PromptSpec, parameters: WorkflowParameters = .init()
    ) {
        self.id = id
        self.operation = operation
        self.executor = executor
        self.inputs = inputs
        self.prompt = prompt
        self.parameters = parameters
    }
}

public struct WorkflowRoute: Codable, Hashable, Sendable {
    public var intent: String
    public var confidence: Double?
    public var router: String?
    public var probabilities: [String: Double]?
    public var model_version: String?
    public var latency_ms: Double?
    public var memory_mb: Double?

    public init(
        intent: String, confidence: Double? = nil, router: String? = nil,
        probabilities: [String: Double]? = nil, modelVersion: String? = nil,
        latencyMS: Double? = nil, memoryMB: Double? = nil
    ) {
        self.intent = intent
        self.confidence = confidence
        self.router = router
        self.probabilities = probabilities
        self.model_version = modelVersion
        self.latency_ms = latencyMS
        self.memory_mb = memoryMB
    }
}

public struct WorkflowPlan: Codable, Hashable, Sendable {
    public var schema_version: Int
    public var steps: [WorkflowStep]
    public var route: WorkflowRoute?

    public init(steps: [WorkflowStep], route: WorkflowRoute? = nil, schemaVersion: Int = 1) {
        self.schema_version = schemaVersion
        self.steps = steps
        self.route = route
    }
}

public struct InputConstraint: Codable, Hashable, Sendable {
    public var min: Int
    public var max: Int
    public var ordered: Bool
}

public struct WorkflowCapability: Codable, Hashable, Sendable {
    public struct PromptSupport: Codable, Hashable, Sendable {
        public var positive: Bool
        public var negative: Bool
    }

    public var model: String
    public var name: String
    public var downloaded: Bool
    public var fits_this_mac: Bool
    public var operation: String
    public var inputs: [String: InputConstraint]
    public var prompt: PromptSupport
}

public struct CapabilityCatalog: Codable, Sendable {
    public var schema_version: Int
    public var operations: [String: [WorkflowCapability]]
}

public struct WorkflowDecision: Codable, Hashable, Sendable {
    public var intent: String
    public var confidence: Double
    public var probabilities: [String: Double]
    public var provider: String
    public var input_result_ids: [String]?
    public var model_version: String?
    public var latency_ms: Double?
    public var memory_mb: Double?
    /// From the agent: the prompt it wrote for the image model, or the question it asks instead of guessing.
    public var prompt: String?
    public var ask: String?
    /// From the agent, when the user left the model on Auto: the image model it picked for this message.
    public var model: String?
    /// Chat images the agent looked at before deciding.
    public var looked: [String]?

    public init(
        intent: String, confidence: Double, probabilities: [String: Double], provider: String,
        inputResultIds: [String]? = nil, modelVersion: String? = nil,
        latencyMS: Double? = nil, memoryMB: Double? = nil, prompt: String? = nil, ask: String? = nil
    ) {
        self.intent = intent
        self.confidence = confidence
        self.probabilities = probabilities
        self.provider = provider
        self.input_result_ids = inputResultIds
        self.model_version = modelVersion
        self.latency_ms = latencyMS
        self.memory_mb = memoryMB
        self.prompt = prompt
        self.ask = ask
    }

    public func requiresConfirmation(threshold: Double = 0.75) -> Bool {
        intent == "ambiguous" || confidence < threshold
    }
}

/// A compact, chronological record supplied to the semantic router. It keeps
/// prompts and image identities without sending image pixels to an LLM.
public struct RoutingHistoryItem: Codable, Hashable, Sendable {
    public var sequence: Int
    public var kind: String
    public var text: String
    public var image_number: Int?
    public var operation: String?
    public var result_id: String?
    public var input_ids: [String]

    public init(
        sequence: Int, kind: String, text: String, imageNumber: Int? = nil,
        operation: String? = nil, resultId: String? = nil, inputIds: [String] = []
    ) {
        self.sequence = sequence
        self.kind = kind
        self.text = text
        self.image_number = imageNumber
        self.operation = operation
        self.result_id = resultId
        self.input_ids = inputIds
    }

    var routingIdentity: String {
        let values = [
            String(sequence), kind, text, image_number.map(String.init) ?? "",
            operation ?? "", result_id ?? "",
        ] + input_ids
        return values.map { "\($0.utf8.count):\($0)" }.joined(separator: "|")
    }

    var jsonObject: [String: Any] {
        var value: [String: Any] = [
            "sequence": sequence, "kind": kind, "text": text, "input_ids": input_ids,
        ]
        if let image_number { value["image_number"] = image_number }
        if let operation { value["operation"] = operation }
        if let result_id { value["result_id"] = result_id }
        return value
    }
}

public struct RoutingContext: Hashable, Sendable {
    public var message: String
    public var currentResultId: String?
    public var attachmentIds: [String]
    public var history: [RoutingHistoryItem]
    public var selectedModel: String?

    public init(
        message: String, currentResultId: String?, attachmentIds: [String],
        history: [RoutingHistoryItem] = [], selectedModel: String? = nil
    ) {
        self.message = message
        self.currentResultId = currentResultId
        self.attachmentIds = attachmentIds
        self.history = history
        self.selectedModel = selectedModel
    }

    /// Collision-safe and order-sensitive without process-randomized hashing.
    public var key: String {
        ([message, currentResultId ?? "", selectedModel ?? ""] + attachmentIds + history.map(\.routingIdentity))
            .map { "\($0.utf8.count):\($0)" }
            .joined(separator: "|")
    }
}

public struct RouteRequestIdentity: Hashable, Sendable {
    public var id: UUID
    public var contextKey: String

    public init(context: RoutingContext, id: UUID = UUID()) {
        self.id = id
        self.contextKey = context.key
    }

    public func accepts(context: RoutingContext, requestID: UUID) -> Bool {
        id == requestID && contextKey == context.key
    }
}

public struct RouterTestResult: Codable, Hashable, Sendable {
    public var ok: Bool
    public var provider: String
    public var intent: String
    public var latency_ms: Double
}

/// The local, image-aware prompt assistant (GET /admin/prompt-assistant).
public struct PromptAssistantStatus: Codable, Hashable, Sendable {
    public var repo: String
    public var installed: Bool
    public var size_gb: Double
    public var resident: Bool
    public var download: EngineModel.Download
    /// The model the agent runs on, and every model it can run on.
    public var selected: String?
    public var models: [Model]?

    public var isDownloading: Bool { download.status == "downloading" }

    public struct Model: Codable, Hashable, Sendable, Identifiable {
        public var id: String
        public var name: String
        public var repo: String
        public var license: String
        public var size_gb: Double
        public var note: String
        public var installed: Bool
        public var download: EngineModel.Download

        public var isDownloading: Bool { download.status == "downloading" }
    }
}

public struct CompiledPromptResponse: Codable, Hashable, Sendable {
    public var user: String
    public var positive: String
    public var negative: String?
    public var compiler: String
    public var negative_supported: Bool
}
