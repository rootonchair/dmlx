import Foundation

/// Launches the Python engine as a child process, waits until it is ready, restarts it if it dies.
@MainActor
public final class EngineProcess {
    public enum State: Equatable, Sendable {
        case stopped
        case starting
        case running(port: Int)
        case failed(String)
    }

    public private(set) var state: State = .stopped {
        didSet { onStateChange?(state) }
    }
    public var onStateChange: ((State) -> Void)?
    public let token = UUID().uuidString.replacingOccurrences(of: "-", with: "")
    public private(set) var logLines: [String] = []

    private var process: Process?
    private var restarts = 0
    private var stopping = false
    private let logURL: URL

    public init() {
        let logs = Store.dataDirectory().appendingPathComponent("logs")
        try? FileManager.default.createDirectory(at: logs, withIntermediateDirectories: true)
        logURL = logs.appendingPathComponent("engine.log")
    }

    public var client: EngineClient? {
        if case .running(let port) = state {
            return EngineClient(baseURL: URL(string: "http://127.0.0.1:\(port)")!, token: token)
        }
        return nil
    }

    /// How to run the engine: a bundled interpreter in release builds, `uv run` in development.
    struct Launch {
        var executable: URL
        var arguments: [String]
        var environment: [String: String]
    }

    static func resolveLaunch(portFile: URL, token: String) -> Launch? {
        var env = ProcessInfo.processInfo.environment
        env["DMLX_TOKEN"] = token
        env["PYTHONUNBUFFERED"] = "1"
        let router = RouterProviderSettings.load()
        if router.kind == .localLLM, !router.localModelPath.isEmpty, !router.localRevision.isEmpty {
            env["DMLX_ROUTER_PROVIDER"] = "local-llm"
            env["DMLX_LOCAL_LLM_MODEL_PATH"] = router.localModelPath
            env["DMLX_LOCAL_LLM_REVISION"] = router.localRevision
        } else if router.kind == .openAICompatible, !router.baseURL.isEmpty, !router.model.isEmpty {
            env["DMLX_ROUTER_PROVIDER"] = "openai-compatible"
            env["DMLX_ROUTER_BASE_URL"] = router.baseURL
            env["DMLX_ROUTER_MODEL"] = router.model
            env["DMLX_ROUTER_TIMEOUT"] = String(router.timeout)
            let key = RouterSecretStore.load()
            if !key.isEmpty { env["DMLX_ROUTER_API_KEY"] = key }
        } else if router.kind == .disabled {
            env["DMLX_ROUTER_PROVIDER"] = "disabled"
        }
        let prompt = PromptProviderSettings.load()
        if !prompt.baseURL.isEmpty, !prompt.model.isEmpty {
            env["DMLX_PROMPT_BASE_URL"] = prompt.baseURL
            env["DMLX_PROMPT_MODEL"] = prompt.model
            env["DMLX_PROMPT_TIMEOUT"] = String(prompt.timeout)
            let key = PromptSecretStore.load()
            if !key.isEmpty { env["DMLX_PROMPT_API_KEY"] = key }
        }
        let serveArgs = ["serve", "--port", "0", "--port-file", portFile.path, "--parent-pid", String(ProcessInfo.processInfo.processIdentifier)]

        // 1. Bundled engine: Contents/Resources/engine/python/bin/python3
        if let resources = Bundle.main.resourceURL {
            let python = resources.appendingPathComponent("engine/python/bin/python3")
            if FileManager.default.isExecutableFile(atPath: python.path) {
                env["PYTHONNOUSERSITE"] = "1"
                return Launch(executable: python, arguments: ["-I", "-m", "dmlx.cli"] + serveArgs, environment: env)
            }
        }
        // 2. Development: DMLX_ENGINE_DIR or ../engine relative to the repo, run with uv.
        let candidates = [
            ProcessInfo.processInfo.environment["DMLX_ENGINE_DIR"],
            URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("../../../engine").standardized.path,
        ].compactMap { $0 }
        for dir in candidates where FileManager.default.fileExists(atPath: dir + "/pyproject.toml") {
            let uvPaths = [
                env["HOME"].map { $0 + "/.local/bin/uv" },
                "/opt/homebrew/bin/uv",
                "/usr/local/bin/uv",
            ].compactMap { $0 }
            if let uv = uvPaths.first(where: { FileManager.default.isExecutableFile(atPath: $0) }) {
                var runArguments = ["run", "--project", dir]
                if router.kind == .localLLM { runArguments += ["--extra", "local-llm"] }
                return Launch(
                    executable: URL(fileURLWithPath: uv),
                    arguments: runArguments + ["dmlx"] + serveArgs, environment: env)
            }
        }
        return nil
    }

    public func start() {
        guard process == nil else { return }
        stopping = false
        state = .starting
        let portFile = FileManager.default.temporaryDirectory.appendingPathComponent("dmlx-port-\(UUID().uuidString)")
        guard let launch = Self.resolveLaunch(portFile: portFile, token: token) else {
            state = .failed("Could not find the dMLX engine. Reinstall the app.")
            return
        }
        let proc = Process()
        proc.executableURL = launch.executable
        proc.arguments = launch.arguments
        proc.environment = launch.environment
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe
        // Append across restarts so the log from before a crash survives; cap its size.
        if let size = (try? FileManager.default.attributesOfItem(atPath: logURL.path))?[.size] as? Int, size > 20_000_000 {
            try? FileManager.default.removeItem(at: logURL)
        }
        if !FileManager.default.fileExists(atPath: logURL.path) {
            FileManager.default.createFile(atPath: logURL.path, contents: nil)
        }
        let logHandle = try? FileHandle(forWritingTo: logURL)
        logHandle?.seekToEndOfFile()
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            logHandle?.write(data)
            let text = String(decoding: data, as: UTF8.self)
            _Concurrency.Task { @MainActor in self?.appendLog(text, portFile: portFile) }
        }
        proc.terminationHandler = { [weak self] p in
            _Concurrency.Task { @MainActor in self?.handleExit(code: p.terminationStatus) }
        }
        do {
            try proc.run()
            process = proc
        } catch {
            state = .failed("Could not start the engine: \(error.localizedDescription)")
        }
    }

    private func appendLog(_ text: String, portFile: URL) {
        for line in text.split(separator: "\n", omittingEmptySubsequences: true) {
            logLines.append(String(line))
            if line.hasPrefix("DMLX_READY port="), let port = Int(line.split(separator: "=").last ?? "") {
                restarts = 0
                state = .running(port: port)
                try? FileManager.default.removeItem(at: portFile)
            }
        }
        if logLines.count > 2000 { logLines.removeFirst(logLines.count - 2000) }
    }

    private func handleExit(code: Int32) {
        process = nil
        if stopping {
            state = .stopped
            return
        }
        restarts += 1
        if restarts > 5 {
            state = .failed("The engine keeps stopping (exit \(code)). See Settings → Engine log.")
            return
        }
        state = .starting
        let delay = pow(2.0, Double(restarts - 1))
        logLines.append("engine exited with code \(code); restarting in \(Int(delay))s")
        _Concurrency.Task { @MainActor in
            try? await _Concurrency.Task.sleep(for: .seconds(delay))
            self.start()
        }
    }

    public func stop() {
        stopping = true
        process?.terminate()
    }

    /// Stops the engine and waits until it has exited (killed after the timeout), so it writes no more
    /// files: Uninstall deletes the data folder right after.
    public func stopAndWait(timeout: Duration = .seconds(5)) async {
        stopping = true
        guard let proc = process else { return }
        proc.terminate()
        let deadline = ContinuousClock.now + timeout
        while proc.isRunning && ContinuousClock.now < deadline {
            try? await _Concurrency.Task.sleep(for: .milliseconds(100))
        }
        if proc.isRunning {
            kill(proc.processIdentifier, SIGKILL)
            proc.waitUntilExit()
        }
    }

    public func restart() {
        restarts = 0
        if let p = process {
            stopping = false
            p.terminate()  // terminationHandler restarts it
        } else {
            start()
        }
    }

    public var logFile: URL { logURL }
}
