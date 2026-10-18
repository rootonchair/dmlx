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
        let logs = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("dMLX/logs")
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
                return Launch(
                    executable: URL(fileURLWithPath: uv),
                    arguments: ["run", "--project", dir, "dmlx"] + serveArgs, environment: env)
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
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
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
