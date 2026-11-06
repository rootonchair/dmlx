import Foundation
import SQLite3

private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

public enum StoreError: Error, CustomStringConvertible {
    case sqlite(String)
    public var description: String {
        switch self { case .sqlite(let m): return "SQLite: \(m)" }
    }
}

/// Local history: sessions and turns, turns stored as JSON payloads.
public final class Store: @unchecked Sendable {
    private var db: OpaquePointer?
    private let queue = DispatchQueue(label: "dmlx.store")
    private let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.dateEncodingStrategy = .secondsSince1970
        return e
    }()
    private let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .secondsSince1970
        return d
    }()

    public init(path: URL) throws {
        try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard sqlite3_open(path.path, &db) == SQLITE_OK else {
            throw StoreError.sqlite("cannot open \(path.path)")
        }
        try exec("PRAGMA journal_mode=WAL;")
        try exec("""
            CREATE TABLE IF NOT EXISTS sessions (
              id TEXT PRIMARY KEY, title TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS turns (
              id TEXT PRIMARY KEY, session_id TEXT NOT NULL, idx INTEGER NOT NULL, payload TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS turns_session ON turns(session_id, idx);
            """)
    }

    deinit { sqlite3_close(db) }

    public static func defaultURL() -> URL {
        dataDirectory().appendingPathComponent("app.sqlite")
    }

    /// The app's data folder: DMLX_HOME when set (tests, screenshots), the same rule as the engine;
    /// otherwise ~/Library/Application Support/dMLX.
    public static func dataDirectory() -> URL {
        if let home = ProcessInfo.processInfo.environment["DMLX_HOME"], !home.isEmpty {
            return URL(fileURLWithPath: home)
        }
        return FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("dMLX")
    }

    private func exec(_ sql: String) throws {
        var err: UnsafeMutablePointer<CChar>?
        if sqlite3_exec(db, sql, nil, nil, &err) != SQLITE_OK {
            let message = err.map { String(cString: $0) } ?? "unknown"
            sqlite3_free(err)
            throw StoreError.sqlite(message)
        }
    }

    private func prepare(_ sql: String) throws -> OpaquePointer? {
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
            throw StoreError.sqlite(String(cString: sqlite3_errmsg(db)))
        }
        return stmt
    }

    private func bind(_ stmt: OpaquePointer?, _ values: [Any]) {
        for (i, value) in values.enumerated() {
            let idx = Int32(i + 1)
            switch value {
            case let s as String: sqlite3_bind_text(stmt, idx, s, -1, SQLITE_TRANSIENT)
            case let d as Double: sqlite3_bind_double(stmt, idx, d)
            case let n as Int: sqlite3_bind_int64(stmt, idx, Int64(n))
            default: sqlite3_bind_null(stmt, idx)
            }
        }
    }

    private func run(_ sql: String, _ values: [Any] = []) throws {
        try queue.sync {
            let stmt = try prepare(sql)
            defer { sqlite3_finalize(stmt) }
            bind(stmt, values)
            guard sqlite3_step(stmt) == SQLITE_DONE else {
                throw StoreError.sqlite(String(cString: sqlite3_errmsg(db)))
            }
        }
    }

    // MARK: sessions

    public func sessions() throws -> [Session] {
        try queue.sync {
            let stmt = try prepare("SELECT id, title, created, updated FROM sessions ORDER BY updated DESC")
            defer { sqlite3_finalize(stmt) }
            var out: [Session] = []
            while sqlite3_step(stmt) == SQLITE_ROW {
                out.append(Session(
                    id: String(cString: sqlite3_column_text(stmt, 0)),
                    title: String(cString: sqlite3_column_text(stmt, 1)),
                    created: Date(timeIntervalSince1970: sqlite3_column_double(stmt, 2)),
                    updated: Date(timeIntervalSince1970: sqlite3_column_double(stmt, 3))))
            }
            return out
        }
    }

    public func save(_ session: Session) throws {
        try run(
            "INSERT INTO sessions(id, title, created, updated) VALUES(?,?,?,?) "
                + "ON CONFLICT(id) DO UPDATE SET title=excluded.title, updated=excluded.updated",
            [session.id, session.title, session.created.timeIntervalSince1970, session.updated.timeIntervalSince1970])
    }

    public func delete(session id: String) throws {
        try run("DELETE FROM turns WHERE session_id=?", [id])
        try run("DELETE FROM sessions WHERE id=?", [id])
    }

    // MARK: turns

    public func turns(session id: String) throws -> [Turn] {
        try queue.sync {
            let stmt = try prepare("SELECT payload FROM turns WHERE session_id=? ORDER BY idx")
            defer { sqlite3_finalize(stmt) }
            bind(stmt, [id])
            var out: [Turn] = []
            while sqlite3_step(stmt) == SQLITE_ROW {
                let json = String(cString: sqlite3_column_text(stmt, 0))
                if let turn = try? decoder.decode(Turn.self, from: Data(json.utf8)) {
                    out.append(turn)
                }
            }
            return out
        }
    }

    public func save(_ turn: Turn) throws {
        let payload = String(decoding: try encoder.encode(turn), as: UTF8.self)
        try run(
            "INSERT INTO turns(id, session_id, idx, payload) VALUES(?,?,?,?) "
                + "ON CONFLICT(id) DO UPDATE SET payload=excluded.payload, idx=excluded.idx",
            [turn.id, turn.sessionId, turn.index, payload])
    }
}
