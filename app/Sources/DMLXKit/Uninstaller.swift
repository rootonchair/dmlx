import Foundation

/// Everything dMLX keeps outside its app bundle, so "Uninstall dMLX…" can delete all of it: macOS runs no
/// code when an app is dragged to the Trash, so this is the only way the models go with the app.
public enum Uninstaller {
    /// The app's folders outside its bundle. The data folder holds the models (models-cache), the agent
    /// model, chats, images, uploads and logs.
    public static func folders(bundleID: String, library: URL? = nil, dataFolder: URL = Store.dataDirectory()) -> [URL] {
        let library = library ?? FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask)[0]
        return [
            dataFolder,
            library.appendingPathComponent("Caches/\(bundleID)"),
            library.appendingPathComponent("HTTPStorages/\(bundleID)"),
            library.appendingPathComponent("Saved Application State/\(bundleID).savedState"),
            library.appendingPathComponent("WebKit/\(bundleID)"),
        ]
    }

    /// The space the folders take on disk, in bytes.
    public static func size(of folders: [URL]) -> Int64 {
        var total: Int64 = 0
        let keys: Set<URLResourceKey> = [.totalFileAllocatedSizeKey, .isRegularFileKey]
        for folder in folders {
            guard let walker = FileManager.default.enumerator(at: folder, includingPropertiesForKeys: Array(keys)) else {
                continue
            }
            for case let url as URL in walker {
                guard let values = try? url.resourceValues(forKeys: keys), values.isRegularFile == true else { continue }
                total += Int64(values.totalFileAllocatedSize ?? 0)
            }
        }
        return total
    }

    /// Deletes the folders, the settings and the API keys in the Keychain. Folders that are already gone
    /// are not an error; the first other failure is thrown after the rest are deleted.
    public static func removeData(bundleID: String, folders: [URL]) throws {
        try? RouterSecretStore.save("")  // an empty key deletes the Keychain item
        try? PromptSecretStore.save("")
        UserDefaults.standard.removePersistentDomain(forName: bundleID)
        try removeFolders(folders)
    }

    /// Deletes the folders that pass isSafeToDelete; see removeData.
    public static func removeFolders(_ folders: [URL]) throws {
        var failure: Error?
        for folder in folders where isSafeToDelete(folder) && FileManager.default.fileExists(atPath: folder.path) {
            do { try FileManager.default.removeItem(at: folder) } catch { failure = failure ?? error }
        }
        if let failure { throw failure }
    }

    /// Refuses the home folder, its parents and anything near the root, in case DMLX_HOME points somewhere
    /// it should not.
    public static func isSafeToDelete(_ folder: URL) -> Bool {
        let path = folder.standardizedFileURL.resolvingSymlinksInPath().path
        let home = URL(fileURLWithPath: NSHomeDirectory()).standardizedFileURL.resolvingSymlinksInPath().path
        return path.split(separator: "/").count >= 3 && !(home + "/").hasPrefix(path + "/")
    }

    /// A shell script that waits for the app to quit, then deletes what quitting writes again (settings,
    /// saved window state). It runs detached, after the app is gone.
    public static func afterQuitScript(pid: Int32, bundleID: String, folders: [URL]) -> String {
        func quoted(_ s: String) -> String { "'" + s.replacingOccurrences(of: "'", with: "'\\''") + "'" }
        let paths = folders.filter(isSafeToDelete).map { quoted($0.path) }.joined(separator: " ")
        return """
            while kill -0 \(pid) 2>/dev/null; do sleep 0.2; done
            defaults delete \(quoted(bundleID)) 2>/dev/null
            rm -rf \(paths) \(quoted(NSHomeDirectory() + "/Library/Preferences/\(bundleID).plist"))
            """
    }
}
