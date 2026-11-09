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
    /// The user's boxes, for models with a prompt format (task 17).
    public var layout: [LayoutBox]?

    public init(
        size: String? = nil, width: Int? = nil, height: Int? = nil, steps: Int? = nil,
        seed: Int? = nil, n: Int? = nil, strategy: String? = nil, force: Bool? = nil,
        scale: Int? = nil, softness: Double? = nil, layout: [LayoutBox]? = nil
    ) {
        self.layout = layout
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

public struct CapabilityCatalog: Codable, Hashable, Sendable {
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
    /// In a hint (no plan) with the agent installed: the models the agent will pick from on send, when
    /// there are several, so the hint can name them rather than one that may not run.
    public var agent_choices: [String]?

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

extension WorkflowDecision {
    /// A one-workflow decision the app made itself (no agent), checked by the engine's route for that
    /// workflow: the engine's answer is kept when it agrees on the workflow, since it may name a model that
    /// fits in memory now in place of the app's Auto model (task 17). Without an answer, or when the engine
    /// disagrees (it counts models a little differently), the app's own decision stands.
    public func checked(by engine: WorkflowDecision?) -> WorkflowDecision {
        guard let engine, engine.intent == intent else { return self }
        return engine
    }
}

/// What the composer's route hint asks the engine, so it names the model the send will run (task 17).
public struct RoutePreviewQuery: Equatable, Sendable {
    /// The workflows to choose from; one when the app already knows it (the engine then only checks the
    /// model fits in memory now), nil for all.
    public var candidates: [String]?
    /// The composer's boxes: with them the engine names a model that can place things, as the send does.
    public var layout: [LayoutBox]?

    /// nil when the hint needs no engine call: a workflow with the user's model (it runs as chosen), or
    /// several workflows with the agent installed (it plans the message on send; the hint says so).
    /// One workflow on Auto asks also with the agent: the route then names the model that fits, or the
    /// agent's choices, and runs no agent (round 3 found the agent-on hint naming the swapped-out model).
    public static func make(
        structural: String?, onAuto: Bool, agentInstalled: Bool, layout: [LayoutBox]?
    ) -> RoutePreviewQuery? {
        let boxes = (layout?.isEmpty ?? true) ? nil : layout
        if let structural {
            return onAuto ? RoutePreviewQuery(candidates: [structural], layout: boxes) : nil
        }
        return agentInstalled ? nil : RoutePreviewQuery(candidates: nil, layout: boxes)
    }
}

/// Which model a workflow runs, the same rule for the composer's hint and the send (task 17).
public enum RouteModel {
    /// The user's model when they chose one; on Auto the decision's model when the app can run it (the
    /// agent's pick, or the engine's swap for one that fits), else the app's own Auto model.
    public static func running(chosen: String?, decision: WorkflowDecision, runnable: Set<String>,
                               auto: String?) -> String? {
        if let chosen { return chosen }
        if let picked = decision.model, runnable.contains(picked) { return picked }
        return auto
    }

    /// The models a hint names: the one that runs, or, when the agent picks on send among several, those.
    /// None on Auto while the engine has not checked the decision yet (`checked` false): the app's own Auto
    /// model may be the one it swaps out, and a hint never names a model that may not run (test round 4).
    public static func named(chosen: String?, decision: WorkflowDecision, runnable: Set<String>,
                             auto: String?, checked: Bool = true) -> [String] {
        if chosen == nil, !checked { return [] }
        if chosen == nil, let choices = decision.agent_choices?.filter(runnable.contains), choices.count > 1 {
            return choices
        }
        return running(chosen: chosen, decision: decision, runnable: runnable, auto: auto).map { [$0] } ?? []
    }

    /// The hint's words for the agent's choices, kept short: two named, then "or another model".
    public static func agentPicks(_ names: [String]) -> String {
        switch names.count {
        case 0: return "the agent picks the model"
        case 1, 2: return "the agent picks " + names.joined(separator: " or ")
        default: return "the agent picks \(names[0]), \(names[1]) or another model"
        }
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
    /// Where the agent runs: "local", or a service ("anthropic", "openai", "ollama", "openai-compatible").
    public var provider: Provider?

    public struct Provider: Codable, Hashable, Sendable {
        public var kind: String
        public var model: String?
        public var base_url: String?
    }

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
