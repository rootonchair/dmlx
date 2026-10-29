import Foundation

public struct JobReport: Codable, Hashable, Sendable {
    public var strategy: String?
    public var loads: [String]?
    public var evicted: [String]?
    public var forced: Bool?
    public var timings: [String: Double]?
    public var peak_mlx_gb: Double?
    public var max_pressure: String?
    public var swap_delta_gb: Double?
    public var total_seconds: Double?
}

/// Payload of one server-sent event. Fields are optional; which ones are set depends on the event.
public struct EventPayload: Codable, Sendable {
    public var job_id: String?
    public var position: Int?
    public var model: String?
    public var width: Int?
    public var height: Int?
    public var steps: Int?
    public var seed: Int?
    public var step: Int?
    public var total: Int?
    public var component: String?
    public var strategy: String?
    public var result_id: String?
    public var result_ids: [String]?
    public var index: Int?
    public var report: JobReport?
    public var code: String?
    public var message: String?
    public var need_gb: Double?
    public var available_gb: Double?
}

public struct EngineEvent: Sendable {
    public var name: String
    public var payload: EventPayload
}

public struct EngineErrorBody: Codable, Sendable {
    public struct Inner: Codable, Sendable {
        public var code: String?
        public var message: String?
    }
    public var error: Inner?
}

public enum EngineClientError: Error, LocalizedError {
    case http(Int, String?, String?)
    case notRunning

    public var errorDescription: String? {
        switch self {
        case .http(_, _, let message): return message ?? "The engine returned an error."
        case .notRunning: return "The engine is not running."
        }
    }

    public var code: String? {
        if case .http(_, let code, _) = self { return code }
        return nil
    }
}

public struct MemoryStatus: Codable, Sendable {
    public struct System: Codable, Sendable {
        public var total_gb: Double
        public var available_gb: Double
        public var swap_used_gb: Double
        public var pressure: String
    }
    public struct Resident: Codable, Sendable, Hashable {
        public var model: String
        public var component: String
        public var gb: Double
        public var pinned: Bool
    }
    public struct Memory: Codable, Sendable {
        public var system: System
        public var headroom_gb: Double
        public var reserve_gb: Double
        public var mlx_active_gb: Double
        public var resident: [Resident]
        public var degraded: Bool
    }
    public var queue: Int
    public var memory: Memory
    public var ram_tier_gb: Int
}

public final class EngineClient: @unchecked Sendable {
    public let baseURL: URL
    public let token: String
    private let session: URLSession

    public init(baseURL: URL, token: String) {
        self.baseURL = baseURL
        self.token = token
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 3600
        config.timeoutIntervalForResource = 24 * 3600
        self.session = URLSession(configuration: config)
    }

    private func request(_ path: String, method: String = "GET", json: [String: Any]? = nil) throws -> URLRequest {
        var req = URLRequest(url: baseURL.appendingPathComponent(path))
        req.httpMethod = method
        req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        if let json {
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: json)
        }
        return req
    }

    private func send<T: Decodable>(_ req: URLRequest, as type: T.Type) async throws -> T {
        let (data, response) = try await session.data(for: req)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(status) else {
            let body = try? JSONDecoder().decode(EngineErrorBody.self, from: data)
            throw EngineClientError.http(status, body?.error?.code, body?.error?.message)
        }
        return try JSONDecoder().decode(T.self, from: data)
    }

    public func health() async -> Bool {
        guard let req = try? request("health") else { return false }
        var r = req
        r.timeoutInterval = 2
        guard let (_, response) = try? await session.data(for: r) else { return false }
        return (response as? HTTPURLResponse)?.statusCode == 200
    }

    public func models() async throws -> [EngineModel] {
        struct List: Decodable { var data: [EngineModel] }
        return try await send(try request("v1/models"), as: List.self).data
    }

    public func memory() async throws -> MemoryStatus {
        try await send(try request("admin/memory"), as: MemoryStatus.self)
    }

    public func capabilities() async throws -> CapabilityCatalog {
        try await send(try request("v1/capabilities"), as: CapabilityCatalog.self)
    }

    public func routeWorkflow(
        message: String, currentResultId: String?, currentPrompt: String?, attachments: [String],
        history: [RoutingHistoryItem] = [], selectedModel: String? = nil, plan: Bool = false, wait: Bool = false
    ) async throws -> WorkflowDecision {
        var body: [String: Any] = [
            "message": message, "attachments": attachments,
            "history": history.map(\.jsonObject),
        ]
        // A sent message (not the typing preview) is planned by the local agent when it is installed.
        if plan { body["plan"] = true }
        // An image is being made in another chat: the engine plans this once it is done (or beside it).
        if plan && wait { body["wait"] = true }
        if let currentResultId { body["current_result_id"] = currentResultId }
        if let currentPrompt { body["current_prompt"] = currentPrompt }
        if let selectedModel { body["selected_model"] = selectedModel }
        return try await send(try request("v1/workflows/route", method: "POST", json: body), as: WorkflowDecision.self)
    }

    public func testRouter() async throws -> RouterTestResult {
        try await send(try request("admin/intelligence/router/test", method: "POST", json: [:]), as: RouterTestResult.self)
    }

    /// For an edit, `imageIds` lets the local prompt assistant see the images being edited.
    public func compilePrompt(prompt: String, operation: String, model: String?, imageIds: [String] = []) async throws -> CompiledPromptResponse {
        var body: [String: Any] = ["prompt": prompt, "operation": operation]
        if let model { body["model"] = model }
        if !imageIds.isEmpty { body["image_ids"] = imageIds }
        return try await send(try request("v1/prompts/compile", method: "POST", json: body), as: CompiledPromptResponse.self)
    }

    public func promptAssistant() async throws -> PromptAssistantStatus {
        try await send(try request("admin/prompt-assistant"), as: PromptAssistantStatus.self)
    }

    /// Downloads an agent model; without one, the selected model.
    public func pullPromptAssistant(model: String? = nil) async throws -> PromptAssistantStatus {
        let body: [String: Any] = model.map { ["model": $0] } ?? [:]
        return try await send(try request("admin/prompt-assistant/pull", method: "POST", json: body), as: PromptAssistantStatus.self)
    }

    /// Makes the agent run on another model from the next message.
    public func selectPromptAssistant(model: String) async throws -> PromptAssistantStatus {
        try await send(try request("admin/prompt-assistant/select", method: "POST", json: ["model": model]),
                       as: PromptAssistantStatus.self)
    }

    public func pull(model: String) async throws {
        struct Ack: Decodable { var status: String }
        _ = try await send(try request("admin/models/\(model)/pull", method: "POST", json: [:]), as: Ack.self)
    }

    public func catalogSuggestions() async throws -> [CatalogSuggestion] {
        struct List: Decodable { var data: [CatalogSuggestion] }
        return try await send(try request("admin/catalog/suggestions"), as: List.self).data
    }

    public func inspect(repo: String) async throws -> ModelInspection {
        try await send(try request("admin/models/inspect", method: "POST", json: ["repo": repo]), as: ModelInspection.self)
    }

    public func addModel(repo: String) async throws -> EngineModel {
        try await send(try request("admin/models", method: "POST", json: ["repo": repo, "download": true]), as: EngineModel.self)
    }

    /// Deletes the downloaded files (the model stays listed and can be downloaded again). Returns GB freed.
    public func deleteFiles(model: String) async throws -> Double {
        struct Ack: Decodable { var freed_gb: Double }
        return try await send(try request("admin/models/\(model)/files", method: "DELETE"), as: Ack.self).freed_gb
    }

    /// Removes a model added from the app, with its files. Returns GB freed.
    public func removeModel(model: String) async throws -> Double {
        struct Ack: Decodable { var freed_gb: Double }
        return try await send(try request("admin/models/\(model)", method: "DELETE"), as: Ack.self).freed_gb
    }

    public func cancel(job: String) async throws {
        struct Ack: Decodable { var status: String }
        _ = try await send(try request("v1/jobs/\(job)", method: "DELETE"), as: Ack.self)
    }

    public func upload(file: URL) async throws -> String {
        struct Ack: Decodable { var id: String }
        let boundary = "dmlx-\(UUID().uuidString)"
        var req = try request("v1/uploads", method: "POST")
        req.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        var body = Data()
        body.append(Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"\(file.lastPathComponent)\"\r\nContent-Type: application/octet-stream\r\n\r\n".utf8))
        body.append(try Data(contentsOf: file))
        body.append(Data("\r\n--\(boundary)--\r\n".utf8))
        req.httpBody = body
        return try await send(req, as: Ack.self).id
    }

    /// Start a generation or edit and stream its events until a terminal event.
    public func run(
        task: GenTask, prompt: String, model: String?, size: String?, seed: Int? = nil,
        sourceIds: [String] = [], mediaInputs: [MediaInput]? = nil,
        force: Bool = false, route: WorkflowRoute? = nil, promptSpec: PromptSpec? = nil,
        operation requestedOperation: String? = nil
    ) -> AsyncThrowingStream<EngineEvent, Error> {
        let fallbackInputs = sourceIds.enumerated().map { index, id in
            MediaInput(id: id, role: index == 0 ? .source : .reference, order: index)
        }
        let inputs = mediaInputs ?? fallbackInputs
        let operation = requestedOperation ?? (task == .edit ? "image.edit" : "image.generate")
        let step = WorkflowStep(
            operation: operation,
            executor: model,
            inputs: inputs,
            prompt: promptSpec ?? PromptSpec(user: prompt),
            parameters: WorkflowParameters(size: size, seed: seed, force: force))
        return runWorkflow(WorkflowPlan(steps: [step], route: route ?? WorkflowRoute(intent: operation)))
    }

    public func runWorkflow(_ plan: WorkflowPlan) -> AsyncThrowingStream<EngineEvent, Error> {
        do {
            let encoded = try JSONEncoder().encode(plan)
            var body = try JSONSerialization.jsonObject(with: encoded) as? [String: Any] ?? [:]
            body["stream"] = true
            return stream(path: "v1/workflows", body: body)
        } catch {
            return AsyncThrowingStream { continuation in
                continuation.finish(throwing: error)
            }
        }
    }

    private func stream(path: String, body: [String: Any]) -> AsyncThrowingStream<EngineEvent, Error> {
        AsyncThrowingStream { continuation in
            let worker = _Concurrency.Task {
                do {
                    let req = try request(path, method: "POST", json: body)
                    let (bytes, response) = try await session.bytes(for: req)
                    let status = (response as? HTTPURLResponse)?.statusCode ?? 0
                    guard (200..<300).contains(status) else {
                        var data = Data()
                        for try await byte in bytes { data.append(byte) }
                        let err = try? JSONDecoder().decode(EngineErrorBody.self, from: data)
                        throw EngineClientError.http(status, err?.error?.code, err?.error?.message)
                    }
                    var parser = SSEParser()
                    for try await line in bytes.lines {
                        if let event = parser.feed(line) {
                            continuation.yield(event)
                            if ["done", "error", "cancelled"].contains(event.name) { break }
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in worker.cancel() }
        }
    }
}

/// Minimal SSE parser for our server: one `event:` line followed by one `data:` line per event.
public struct SSEParser: Sendable {
    private var pendingName = "message"

    public init() {}

    public mutating func feed(_ line: String) -> EngineEvent? {
        if line.hasPrefix("event:") {
            pendingName = line.dropFirst(6).trimmingCharacters(in: .whitespaces)
            return nil
        }
        if line.hasPrefix("data:") {
            let json = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
            let payload = (try? JSONDecoder().decode(EventPayload.self, from: Data(json.utf8))) ?? EventPayload()
            let event = EngineEvent(name: pendingName, payload: payload)
            pendingName = "message"
            return event
        }
        return nil
    }
}
