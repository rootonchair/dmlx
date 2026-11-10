import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import XCTest

@testable import DMLXKit

/// Task 19: the app fixes from the smoke test.
final class AppRulesTests: XCTestCase {
    // MARK: sizes in decimal GB

    func testDiskSizesAreDecimal() {
        XCTAssertEqual(DiskSize.gb(bytes: 4_620_000_000), 4.62, accuracy: 1e-9)
        XCTAssertEqual(DiskSize.progress(done: 1_000_000_000, total: 4_620_000_000), "1.0 of 4.6 GB")
        XCTAssertEqual(DiskSize.text(gb: 4.62), "4.6 GB")
    }

    func testResumeLabel() {
        XCTAssertEqual(DiskSize.resumeLabel(partialBytes: 1_000_000_000, totalBytes: 4_620_000_000),
                       "Resume download · 1.0 of 4.6 GB")
        XCTAssertEqual(DiskSize.resumeLabel(partialBytes: 250_000_000, totalBytes: nil), "Resume download · 0.2 GB done")
        // Never more than the whole.
        XCTAssertEqual(DiskSize.resumeLabel(partialBytes: 5_000_000_000, totalBytes: 4_620_000_000),
                       "Resume download · 4.6 of 4.6 GB")
    }

    /// Test round 4: trashing a 12 MB partial download toasted "freed 0.0 GB".
    func testTheRemoveToastShowsMegabytesForASmallPartial() {
        XCTAssertEqual(DiskSize.removedMessage(name: "FLUX.2 Klein 4B", freedGB: 0.01), "Removed FLUX.2 Klein 4B · freed 10 MB")
        XCTAssertEqual(DiskSize.removedMessage(name: "FLUX.2 Klein 4B", freedGB: 0.11), "Removed FLUX.2 Klein 4B · freed 0.1 GB")
        XCTAssertEqual(DiskSize.removedMessage(name: "FLUX.2 Klein 4B", freedGB: 4.62), "Removed FLUX.2 Klein 4B · freed 4.6 GB")
        XCTAssertEqual(DiskSize.removedMessage(name: "FLUX.2 Klein 4B", freedGB: 0), "Removed FLUX.2 Klein 4B")
    }

    /// Test round 3: a 40 MB partial read "Resume download · 0.0 of 4.6 GB", as if nothing was kept.
    func testASmallPartialDownloadShowsMegabytes() {
        XCTAssertEqual(DiskSize.resumeLabel(partialBytes: 39_850_416, totalBytes: 4_620_000_000),
                       "Resume download · 40 MB of 4.6 GB")
        XCTAssertEqual(DiskSize.resumeLabel(partialBytes: 39_850_416, totalBytes: nil), "Resume download · 40 MB done")
        XCTAssertEqual(DiskSize.amount(bytes: 39_850_416), "40 MB")
        XCTAssertEqual(DiskSize.amount(bytes: 300_000), "1 MB")  // never "0 MB" for something kept
        XCTAssertEqual(DiskSize.amount(bytes: 99_400_000), "99 MB")
        XCTAssertEqual(DiskSize.amount(bytes: 100_000_000), "0.1 GB")
        XCTAssertEqual(DiskSize.progress(done: 5_000_000, total: 4_620_000_000), "5 MB of 4.6 GB")
    }

    private func model(_ extra: String) throws -> EngineModel {
        let json = """
        {"id":"flux2-klein-4b","name":"FLUX.2 Klein 4B","license":"apache-2.0","tasks":["t2i","edit"],
         "max_references":3,"min_ram_gb":16,"size_gb":4.62,"downloaded":false,"resident":false,"pinned":false,
         "fits_this_mac":true,\(extra)}
        """
        return try JSONDecoder().decode(EngineModel.self, from: Data(json.utf8))
    }

    func testPartialDownloadFromModelList() throws {
        let idle = #""download":{"status":"idle","total_bytes":0,"done_bytes":0,"progress":0}"#
        // An older engine sends no partial_bytes.
        XCTAssertFalse(try model(idle).hasPartialDownload)
        let partial = try model(idle + #","partial_bytes":1000000000"#)
        XCTAssertTrue(partial.hasPartialDownload)
        XCTAssertEqual(partial.sizeBytes, 4_620_000_000)  // the listed size when no download total is known
        // While it downloads, the progress shows instead of Resume; the download's exact total wins.
        let running = try model(#""download":{"status":"downloading","total_bytes":4621000000,"done_bytes":1,"progress":0.1},"partial_bytes":1000000000"#)
        XCTAssertFalse(running.hasPartialDownload)
        XCTAssertTrue(running.isDownloading)
        XCTAssertEqual(running.sizeBytes, 4_621_000_000)
    }

    // MARK: attaching

    func testAttachPlanSaysHowManyWereLeftOut() {
        let klein = AttachLimit(modelName: "FLUX.2 Klein 4B", modelMax: 3)
        let five = AttachPlan(files: 5, current: 0, uploading: 0, limit: klein)
        XCTAssertEqual(five, AttachPlan(files: 5, current: 0, uploading: 0, limit: 3))
        XCTAssertEqual(five.take, 3)
        XCTAssertEqual(five.skipped, 2)
        XCTAssertEqual(five.message(klein), "2 files were not attached: FLUX.2 Klein 4B accepts up to 3 images.")
        XCTAssertNil(AttachPlan(files: 2, current: 1, uploading: 0, limit: klein).message(klein))
        XCTAssertEqual(AttachPlan(files: 2, current: 2, uploading: 0, limit: klein).message(klein),
                       "1 file was not attached: FLUX.2 Klein 4B accepts up to 3 images.")
        XCTAssertEqual(AttachPlan(files: 1, current: 3, uploading: 0, limit: klein).message(klein),
                       "Not attached: FLUX.2 Klein 4B accepts up to 3 images.")
    }

    func testAttachLimitIsTheModelsOwnOrTheApps() {
        // Qwen Image 2.1 takes 10: the app's 3 is the limit, and the toast says so instead of blaming the model.
        let qwen = AttachLimit(modelName: "Qwen Image 2.1", modelMax: 10)
        XCTAssertEqual(qwen.count, 3)
        XCTAssertEqual(AttachPlan(files: 4, current: 0, uploading: 0, limit: qwen).message(qwen),
                       "1 file was not attached: dMLX attaches up to 3 images to a message.")
        // A model that takes fewer than 3 sets the limit.
        let one = AttachLimit(modelName: "Single", modelMax: 1)
        XCTAssertEqual(one.count, 1)
        XCTAssertEqual(AttachPlan(files: 2, current: 0, uploading: 0, limit: one).message(one),
                       "1 file was not attached: Single accepts 1 image.")
        // No edit model known yet (none downloaded): the app's limit.
        XCTAssertEqual(AttachLimit(modelName: nil, modelMax: nil).count, 3)
        XCTAssertEqual(AttachLimit(modelName: "Text only", modelMax: 0).count, 3)
    }

    func testUploadsInFlightHoldTheirPlace() {
        // Files dropped one by one while the first ones still upload can't go over the limit.
        let plan = AttachPlan(files: 2, current: 1, uploading: 2, limit: 3)
        XCTAssertEqual(plan.take, 0)
        XCTAssertEqual(plan.skipped, 2)
    }

    func testFailureMessageNamesTheFileOnce() {
        XCTAssertEqual(UploadImage.failureMessage(file: "notes.txt", error: "notes.txt is not an image dMLX can open."),
                       "Couldn't attach: notes.txt is not an image dMLX can open.")
        XCTAssertEqual(UploadImage.failureMessage(file: "a.png", error: "The engine returned an error."),
                       "Couldn't attach a.png: The engine returned an error.")
    }

    func testHEICIsConvertedToAnUprightPNG() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("dmlx-heic-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let heic = dir.appendingPathComponent("IMG_0001.HEIC")
        // A 64×32 image stored with orientation 6 (rotated 90°), as a portrait iPhone photo is.
        let context = CGContext(data: nil, width: 64, height: 32, bitsPerComponent: 8, bytesPerRow: 0,
                                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.setFillColor(red: 0, green: 0, blue: 1, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: 64, height: 32))
        guard let dest = CGImageDestinationCreateWithURL(heic as CFURL, UTType.heic.identifier as CFString, 1, nil) else {
            throw XCTSkip("This Mac can't write HEIC")
        }
        CGImageDestinationAddImage(dest, context.makeImage()!, [kCGImagePropertyOrientation: 6] as CFDictionary)
        guard CGImageDestinationFinalize(dest) else { throw XCTSkip("This Mac can't write HEIC") }

        XCTAssertTrue(UploadImage.needsConversion(heic))
        XCTAssertTrue(UploadImage.needsConversion(URL(fileURLWithPath: "/x/photo.heif")))
        XCTAssertFalse(UploadImage.needsConversion(URL(fileURLWithPath: "/x/photo.png")))
        XCTAssertFalse(UploadImage.needsConversion(URL(fileURLWithPath: "/x/photo.jpg")))

        let png = try UploadImage.convertToPNG(heic)
        defer { try? FileManager.default.removeItem(at: png.deletingLastPathComponent()) }
        XCTAssertEqual(png.lastPathComponent, "IMG_0001.png")
        let source = try XCTUnwrap(CGImageSourceCreateWithURL(png as CFURL, nil))
        XCTAssertEqual(CGImageSourceGetType(source) as String?, UTType.png.identifier)
        let props = try XCTUnwrap(CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any])
        // Rotated upright: 32 wide, 64 high.
        XCTAssertEqual(props[kCGImagePropertyPixelWidth] as? Int, 32)
        XCTAssertEqual(props[kCGImagePropertyPixelHeight] as? Int, 64)
    }

    func testUnreadableFileIsAClearError() {
        let bad = FileManager.default.temporaryDirectory.appendingPathComponent("dmlx-bad-\(UUID().uuidString).heic")
        try? Data("not an image".utf8).write(to: bad)
        defer { try? FileManager.default.removeItem(at: bad) }
        XCTAssertThrowsError(try UploadImage.convertToPNG(bad)) { error in
            XCTAssertEqual(error.localizedDescription, "\(bad.lastPathComponent) could not be read as an image.")
        }
    }

    /// A Display P3 photo (HEIC or JPEG) is converted to sRGB: the engine reads pixels as sRGB (fix round 1).
    func testWideColourPhotosBecomeSRGB() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("dmlx-p3-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let p3 = CGColorSpace(name: CGColorSpace.displayP3)!
        let context = CGContext(data: nil, width: 16, height: 16, bitsPerComponent: 8, bytesPerRow: 0, space: p3,
                                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        // A colour inside both gamuts, whose numbers differ between P3 and sRGB.
        let colour = CGColor(colorSpace: p3, components: [0.2, 0.6, 0.2, 1])!
        let want = colour.converted(to: CGColorSpace(name: CGColorSpace.sRGB)!, intent: .defaultIntent, options: nil)!
            .components!.prefix(3).map { Int(($0 * 255).rounded()) }
        XCTAssertNotEqual(want, [51, 153, 51])  // otherwise this test could not tell
        context.setFillColor(colour)
        context.fill(CGRect(x: 0, y: 0, width: 16, height: 16))
        let image = context.makeImage()!

        func write(_ name: String, _ type: UTType) throws -> URL {
            let url = dir.appendingPathComponent(name)
            guard let dest = CGImageDestinationCreateWithURL(url as CFURL, type.identifier as CFString, 1, nil) else {
                throw XCTSkip("This Mac can't write \(type.identifier)")
            }
            CGImageDestinationAddImage(dest, image, nil)
            guard CGImageDestinationFinalize(dest) else { throw XCTSkip("This Mac can't write \(type.identifier)") }
            return url
        }
        func profile(_ url: URL) throws -> String? {
            let source = try XCTUnwrap(CGImageSourceCreateWithURL(url as CFURL, nil))
            let props = try XCTUnwrap(CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any])
            return props[kCGImagePropertyProfileName] as? String
        }
        func firstPixel(_ url: URL) throws -> [UInt8] {
            let source = try XCTUnwrap(CGImageSourceCreateWithURL(url as CFURL, nil))
            let cg = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
            let data = try XCTUnwrap(cg.dataProvider?.data as Data?)
            return Array(data.prefix(3))
        }

        for (name, type) in [("p3.heic", UTType.heic), ("p3.jpg", UTType.jpeg)] {
            let file = try write(name, type)
            XCTAssertEqual(try profile(file), "Display P3", name)
            XCTAssertTrue(UploadImage.needsConversion(file), name)
            let png = try UploadImage.convertToPNG(file)
            defer { try? FileManager.default.removeItem(at: png.deletingLastPathComponent()) }
            let out = try XCTUnwrap(try profile(png), name)
            XCTAssertTrue(UploadImage.isSRGB(profileName: out), "\(name): \(out)")
            let rgb = try firstPixel(png).map(Int.init)
            for c in 0..<3 { XCTAssertEqual(rgb[c], want[c], accuracy: 4, "\(name): \(rgb) vs \(want)") }
        }
        // An sRGB PNG is uploaded as it is.
        let srgb = dir.appendingPathComponent("plain.png")
        let plain = CGContext(data: nil, width: 4, height: 4, bitsPerComponent: 8, bytesPerRow: 0,
                              space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        let dest = try XCTUnwrap(CGImageDestinationCreateWithURL(srgb as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(dest, plain.makeImage()!, nil)
        XCTAssertTrue(CGImageDestinationFinalize(dest))
        XCTAssertFalse(UploadImage.needsConversion(srgb))
    }

    // MARK: toasts, the quit question, a held send

    func testOneDropsNotesShareTheToast() {
        let skipped = "2 files were not attached: FLUX.2 Klein 4B accepts up to 3 images."
        let failed = "Couldn't attach: notes.txt is not an image dMLX can open."
        XCTAssertEqual(ToastText.merged(skipped, failed), skipped + " " + failed)
        XCTAssertEqual(ToastText.merged(nil, failed), failed)
        XCTAssertEqual(ToastText.merged(failed, failed), failed)
        // A short toast stays 3 s; a long, merged one stays long enough to read, never more than 10 s.
        XCTAssertEqual(ToastText.seconds(for: "Saved"), 3)
        XCTAssertGreaterThan(ToastText.seconds(for: skipped + " " + failed), 6)
        XCTAssertEqual(ToastText.seconds(for: String(repeating: "x", count: 400)), 10)
    }

    func testImagesAttachedWhileAMessageIsPlannedStay() {
        struct Item: Identifiable, Equatable { var id: String }
        // "a red boat" went out with a.png; b.png was attached while it was planned.
        let kept = SentAttachments.kept([Item(id: "a"), Item(id: "b")], sent: ["a"])
        XCTAssertEqual(kept, [Item(id: "b")])
        XCTAssertEqual(SentAttachments.kept([Item(id: "a")], sent: ["a"]), [])
        XCTAssertEqual(SentAttachments.kept([Item(id: "b")], sent: []), [Item(id: "b")])
    }

    func testTheLoadHintShowsMemoryNotDiskSize() {
        XCTAssertEqual(LoadHint.text(name: "Z-Image Turbo", loads: false, memoryGB: 5.4), "Z-Image Turbo")
        XCTAssertEqual(LoadHint.text(name: "Z-Image Turbo", loads: true, memoryGB: 5.43), "loads Z-Image Turbo · about 5.4 GB")
        XCTAssertEqual(LoadHint.text(name: "Z-Image Turbo", loads: true, memoryGB: nil), "loads Z-Image Turbo")
        // The engine's memory figure is read from the model list; old engines leave it out.
        let json = #"{"id":"z","name":"Z","license":"apache-2.0","tasks":["t2i"],"max_references":0,"min_ram_gb":16,"size_gb":6.2,"memory_gb":5.4,"downloaded":true,"resident":false,"pinned":false,"fits_this_mac":true,"download":{"status":"done","total_bytes":1,"done_bytes":1,"progress":1}}"#
        let m = try! JSONDecoder().decode(EngineModel.self, from: Data(json.utf8))
        XCTAssertEqual(m.memory_gb, 5.4)
        let old = try! JSONDecoder().decode(EngineModel.self, from: Data(json.replacingOccurrences(of: #""memory_gb":5.4,"#, with: "").utf8))
        XCTAssertNil(old.memory_gb)
    }

    func testOneQuitQuestionAtATime() {
        var gate = QuitGate()
        XCTAssertEqual(gate.request(downloading: false), .quit)
        XCTAssertEqual(gate.request(downloading: true), .ask)
        // A second quit (Dock, logout) while the question is open: no second alert.
        XCTAssertEqual(gate.request(downloading: true), .ignore)
        XCTAssertEqual(gate.request(downloading: true), .ignore)
        gate.answered(quit: false)
        // Cancel, then quit again: asked again.
        XCTAssertEqual(gate.request(downloading: true), .ask)
        gate.answered(quit: true)
        // Agreed: AppKit's own terminate check after that quits without asking.
        XCTAssertEqual(gate.request(downloading: true), .quit)
    }

    func testQuitAsksDuringAnAgentDownloadToo() {
        XCTAssertNil(QuitQuestion.detail(imageDownloads: [], agentDownload: nil))
        XCTAssertEqual(QuitQuestion.detail(imageDownloads: [], agentDownload: "Qwen3-VL 8B"),
                       "Qwen3-VL 8B stops now. Download it again and it resumes where it stopped.")
        XCTAssertEqual(QuitQuestion.detail(imageDownloads: ["FLUX.2 Klein 4B"], agentDownload: nil),
                       "FLUX.2 Klein 4B stops now. Download it again and it resumes where it stopped.")
        XCTAssertEqual(QuitQuestion.detail(imageDownloads: ["FLUX.2 Klein 4B"], agentDownload: "Qwen3-VL 8B"),
                       "2 downloads stop now. Download them again and they resume where they stopped.")
    }

    func testHintWaitsForImagesStillAttaching() {
        XCTAssertEqual(UploadWait.hint(attaching: 1), "Attaching an image · decided when it is attached")
        XCTAssertEqual(UploadWait.hint(attaching: 3), "Attaching 3 images · decided when they are attached")
        XCTAssertFalse(UploadWait.hint(attaching: 2).contains("generate"))
        // A hung engine can't hold a send for an hour (the client's general limit).
        XCTAssertLessThanOrEqual(UploadWait.timeout, 300)
    }

    // MARK: the welcome sheet

    func testWelcomeOpensOncePerLaunch() {
        XCTAssertTrue(OnboardingRule.opensByItself(alreadyOfferedThisLaunch: false, hasImageModel: false,
                                                   setupCompleted: true, agentInstalled: false))
        // After Set Up Later, an engine restart does not open it again.
        XCTAssertFalse(OnboardingRule.opensByItself(alreadyOfferedThisLaunch: true, hasImageModel: false,
                                                    setupCompleted: true, agentInstalled: false))
        // Set up with an image model and the agent: not shown.
        XCTAssertFalse(OnboardingRule.opensByItself(alreadyOfferedThisLaunch: false, hasImageModel: true,
                                                    setupCompleted: false, agentInstalled: true))
        // Setup never finished and no agent: shown on the next launch.
        XCTAssertTrue(OnboardingRule.opensByItself(alreadyOfferedThisLaunch: false, hasImageModel: true,
                                                   setupCompleted: false, agentInstalled: false))
    }

    // MARK: the setup sheet on a 14-inch display

    func testSetupSheetEndsAboveTheDock() {
        // 14-inch MacBook: 982 pt screen, Dock visible (visible frame from y = 62); window 572 high, top at
        // y = 200 from the top, so its content (below a 52 pt toolbar) starts at 982 - 200 - 52 = 730.
        let room = SheetFit.maxSheetHeight(contentTop: 730, visibleBottom: 62)
        XCTAssertEqual(room, 656)
        // The choices want 580 pt; title and buttons take 200: the choices get what is left and scroll.
        let choices = SheetFit.choicesHeight(wanted: 580, maxSheet: room - 56, fixed: 200)
        XCTAssertEqual(choices, 400)
        XCTAssertLessThanOrEqual(choices + 200 + 56, room)
        // A tall screen: no scrolling.
        XCTAssertEqual(SheetFit.choicesHeight(wanted: 580, maxSheet: 1200, fixed: 200), 580)
        // Never squeezed below a usable list.
        XCTAssertEqual(SheetFit.choicesHeight(wanted: 580, maxSheet: 250, fixed: 200), 140)
    }
}
