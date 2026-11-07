import XCTest

@testable import DMLXKit

final class UninstallerTests: XCTestCase {
    func testListsTheDataFolderAndTheAppsLibraryFolders() {
        let library = URL(fileURLWithPath: "/Users/someone/Library")
        let data = URL(fileURLWithPath: "/Users/someone/Library/Application Support/dMLX")
        let paths = Uninstaller.folders(bundleID: "io.github.dmlx.app", library: library, dataFolder: data).map(\.path)
        XCTAssertEqual(paths.first, data.path)  // the models live in its models-cache
        XCTAssertTrue(paths.contains("/Users/someone/Library/Caches/io.github.dmlx.app"))
        XCTAssertTrue(paths.contains("/Users/someone/Library/Saved Application State/io.github.dmlx.app.savedState"))
    }

    func testDeletesTheFoldersAndMeasuresThemFirst() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("uninstall-\(UUID().uuidString)")
        let data = root.appendingPathComponent("dMLX")
        let models = data.appendingPathComponent("models-cache/models--org--model/blobs")
        try FileManager.default.createDirectory(at: models, withIntermediateDirectories: true)
        try Data(count: 100_000).write(to: models.appendingPathComponent("abc"))
        defer { try? FileManager.default.removeItem(at: root) }

        XCTAssertGreaterThanOrEqual(Uninstaller.size(of: [data]), 100_000)
        try Uninstaller.removeFolders([data, root.appendingPathComponent("never-created")])
        XCTAssertFalse(FileManager.default.fileExists(atPath: data.path))
    }

    func testNeverDeletesTheHomeFolderOrAnythingAboveIt() {
        let home = URL(fileURLWithPath: NSHomeDirectory())
        XCTAssertFalse(Uninstaller.isSafeToDelete(home))
        XCTAssertFalse(Uninstaller.isSafeToDelete(home.deletingLastPathComponent()))
        XCTAssertFalse(Uninstaller.isSafeToDelete(URL(fileURLWithPath: "/")))
        XCTAssertFalse(Uninstaller.isSafeToDelete(URL(fileURLWithPath: "/tmp")))
        XCTAssertFalse(Uninstaller.isSafeToDelete(home.appendingPathComponent("Documents")))
        XCTAssertFalse(Uninstaller.isSafeToDelete(home.appendingPathComponent("Library")))
        XCTAssertTrue(Uninstaller.isSafeToDelete(home.appendingPathComponent("Library/Application Support/dMLX")))
    }

    func testReportsAFolderItRefusesInsteadOfSkippingIt() {
        let documents = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Documents")
        guard FileManager.default.fileExists(atPath: documents.path) else { return }
        XCTAssertThrowsError(try Uninstaller.removeFolders([documents]))
        XCTAssertTrue(FileManager.default.fileExists(atPath: documents.path))
    }

    func testTakesOnlyTheDefaultOrAMarkedFolderAsTheDataFolder() throws {
        XCTAssertTrue(Uninstaller.isOwnDataFolder(Store.defaultDataDirectory()))
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("data-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        XCTAssertFalse(Uninstaller.isOwnDataFolder(folder))  // e.g. DMLX_HOME=~/Documents
        FileManager.default.createFile(atPath: folder.appendingPathComponent(Store.dataFolderMarker).path, contents: Data())
        XCTAssertTrue(Uninstaller.isOwnDataFolder(folder))
    }

    func testTheAfterQuitScriptQuotesPathsAndSkipsUnsafeOnes() {
        let script = Uninstaller.afterQuitScript(pid: 42, bundleID: "io.github.dmlx.app", folders: [
            URL(fileURLWithPath: NSHomeDirectory() + "/Library/Saved Application State/it's.savedState"),
            URL(fileURLWithPath: NSHomeDirectory()),
        ])
        XCTAssertTrue(script.contains("kill -0 42"))
        XCTAssertTrue(script.contains("'\(NSHomeDirectory())/Library/Saved Application State/it'\\''s.savedState'"))
        XCTAssertFalse(script.contains("'\(NSHomeDirectory())' "))
    }
}
