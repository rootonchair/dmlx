import XCTest

@testable import DMLXKit

final class StoreTests: XCTestCase {
    func testLegacyTurnJSONDecodesWithoutWorkflowFields() throws {
        let json = #"""
        {
          "id":"turn-old","sessionId":"session-old","index":0,"kind":"image",
          "text":"boat","number":1,"route":"Generated","activeVersion":0,
          "attachments":[],"created":0,
          "versions":[{
            "id":"version-old","status":"done","resultId":"img_old","prompt":"boat",
            "model":"z-image-turbo","task":"t2i","size":"1024x1024","sourceIds":[],
            "step":0,"totalSteps":0,"phase":"Done","loads":[]
          }]
        }
        """#
        let turn = try JSONDecoder().decode(Turn.self, from: Data(json.utf8))
        XCTAssertEqual(turn.current?.resultId, "img_old")
        XCTAssertNil(turn.attachmentInputs)
        XCTAssertNil(turn.current?.workflowRoute)
        XCTAssertNil(turn.current?.operation)
    }

    func testRoundTrip() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("dmlx-test-\(UUID()).sqlite")
        defer { try? FileManager.default.removeItem(at: url) }
        let store = try Store(path: url)
        var session = Session(title: "Lake boat")
        try store.save(session)
        var turn = Turn(sessionId: session.id, index: 0, kind: .image, number: 1, route: "Generated",
                        versions: [ImageVersion(prompt: "boat", model: "z-image-turbo", task: .t2i, size: "1024x1024")])
        try store.save(turn)
        turn.versions[0].status = .done
        turn.versions[0].resultId = "img_1"
        try store.save(turn)
        session.title = "Renamed"
        try store.save(session)

        let reopened = try Store(path: url)
        XCTAssertEqual(try reopened.sessions().first?.title, "Renamed")
        let turns = try reopened.turns(session: session.id)
        XCTAssertEqual(turns.count, 1)
        XCTAssertEqual(turns[0].current?.resultId, "img_1")
        try reopened.delete(session: session.id)
        XCTAssertTrue(try reopened.sessions().isEmpty)
    }

    func testDeletingOneSessionRemovesOnlyItsTurns() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("dmlx-delete-\(UUID()).sqlite")
        defer { try? FileManager.default.removeItem(at: url) }
        let store = try Store(path: url)
        let first = Session(title: "First")
        let second = Session(title: "Second")
        try store.save(first)
        try store.save(second)
        try store.save(Turn(sessionId: first.id, index: 0, kind: .notice, text: "first"))
        try store.save(Turn(sessionId: second.id, index: 0, kind: .notice, text: "second"))

        try store.delete(session: first.id)

        XCTAssertEqual(try store.sessions().map(\.id), [second.id])
        XCTAssertTrue(try store.turns(session: first.id).isEmpty)
        XCTAssertEqual(try store.turns(session: second.id).map(\.text), ["second"])
    }

    func testTheAgentsReplyIsKeptAsItsOwnKindOfTurn() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("dmlx-reply-\(UUID()).sqlite")
        defer { try? FileManager.default.removeItem(at: url) }
        let store = try Store(path: url)
        let session = Session(title: "Hello")
        try store.save(session)
        try store.save(Turn(sessionId: session.id, index: 0, kind: .user, text: "hello"))
        try store.save(Turn(sessionId: session.id, index: 1, kind: .reply, text: "Hi! Describe an image to make."))

        XCTAssertEqual(try Store(path: url).turns(session: session.id).map(\.kind), [.user, .reply])
    }
}

final class SSEParserTests: XCTestCase {
    func testParsesEvents() {
        var parser = SSEParser()
        XCTAssertNil(parser.feed("event: step"))
        let step = parser.feed(#"data: {"job_id":"job_1","step":3,"total":8}"#)
        XCTAssertEqual(step?.name, "step")
        XCTAssertEqual(step?.payload.step, 3)
        XCTAssertNil(parser.feed(": keep-alive"))
        _ = parser.feed("event: done")
        let done = parser.feed(#"data: {"job_id":"job_1","result_ids":["img_1"],"report":{"strategy":"eager","loads":[]}}"#)
        XCTAssertEqual(done?.payload.result_ids, ["img_1"])
        XCTAssertEqual(done?.payload.report?.strategy, "eager")
    }
}

final class EngineModelTests: XCTestCase {
    func testResearchModelMetadataDecodes() throws {
        let json = #"{"id":"qwen-image-2.1-q4","name":"Qwen Image 2.1 Base Q4","license":"qwen-research","tasks":["t2i","edit"],"max_references":10,"min_ram_gb":24,"downloaded":true,"resident":false,"pinned":false,"fits_this_mac":true,"download":{"status":"done","total_bytes":1,"done_bytes":1,"progress":1},"auto_select":false,"experimental":true,"license_url":"https://example.com/license","license_acceptance_id":"qwen-research-2026-09-20"}"#
        let model = try JSONDecoder().decode(EngineModel.self, from: Data(json.utf8))
        XCTAssertFalse(model.autoSelectable)
        XCTAssertTrue(model.requiresLicenseAcceptance)
        XCTAssertTrue(model.isExperimental)
        XCTAssertEqual(model.max_references, 10)
    }

    func testLegacyModelDefaultsToAutoSelectable() throws {
        let json = #"{"id":"legacy","name":"Legacy","license":"apache-2.0","tasks":["t2i"],"max_references":0,"min_ram_gb":8,"downloaded":true,"resident":false,"pinned":false,"fits_this_mac":true,"download":{"status":"done","total_bytes":1,"done_bytes":1,"progress":1}}"#
        let model = try JSONDecoder().decode(EngineModel.self, from: Data(json.utf8))
        XCTAssertTrue(model.autoSelectable)
        XCTAssertFalse(model.requiresLicenseAcceptance)
    }
}

final class WorkflowTests: XCTestCase {
    func testRoutingContextKeyRejectsStaleTextCurrentImageAndAttachmentOrder() {
        let history = [RoutingHistoryItem(
            sequence: 0, kind: "image", text: "a blue car", imageNumber: 1,
            operation: "image.generate", resultId: "img_1")]
        let original = RoutingContext(
            message: "combine", currentResultId: "img_current", attachmentIds: ["img_1", "img_2"],
            history: history)
        XCTAssertNotEqual(original.key, RoutingContext(
            message: "combine these", currentResultId: "img_current", attachmentIds: ["img_1", "img_2"],
            history: history).key)
        XCTAssertNotEqual(original.key, RoutingContext(
            message: "combine", currentResultId: "img_other", attachmentIds: ["img_1", "img_2"],
            history: history).key)
        XCTAssertNotEqual(original.key, RoutingContext(
            message: "combine", currentResultId: "img_current", attachmentIds: ["img_2", "img_1"],
            history: history).key)
        XCTAssertNotEqual(original.key, RoutingContext(
            message: "combine", currentResultId: "img_current", attachmentIds: ["img_1", "img_2"],
            history: [RoutingHistoryItem(
                sequence: 0, kind: "image", text: "a red car", imageNumber: 1,
                operation: "image.generate", resultId: "img_1")]).key)
        XCTAssertNotEqual(original.key, RoutingContext(
            message: "combine", currentResultId: "img_current", attachmentIds: ["img_1", "img_2"],
            history: history, selectedModel: "qwen-image-2.1-q4").key)
        XCTAssertNotEqual(
            RoutingContext(message: "a\u{1f}b", currentResultId: nil, attachmentIds: ["c"]).key,
            RoutingContext(message: "a", currentResultId: "b", attachmentIds: ["c"]).key)
    }

    func testRoutingHistoryRoundTripPreservesChronologyAndReferences() throws {
        let history = [
            RoutingHistoryItem(
                sequence: 0, kind: "image", text: "a blue car", imageNumber: 1,
                operation: "image.generate", resultId: "img_1"),
            RoutingHistoryItem(
                sequence: 1, kind: "image", text: "make it red", imageNumber: 2,
                operation: "image.edit", resultId: "img_2", inputIds: ["img_1"]),
        ]
        let decoded = try JSONDecoder().decode(
            [RoutingHistoryItem].self, from: JSONEncoder().encode(history))
        XCTAssertEqual(decoded, history)
        XCTAssertEqual(decoded.map(\.result_id), ["img_1", "img_2"])
    }

    func testLowConfidenceAndAmbiguousDecisionsRequireConfirmation() {
        XCTAssertTrue(WorkflowDecision(
            intent: "image.edit", confidence: 0.74, probabilities: [:], provider: "test").requiresConfirmation())
        XCTAssertTrue(WorkflowDecision(
            intent: "ambiguous", confidence: 0.99, probabilities: [:], provider: "test").requiresConfirmation())
        XCTAssertFalse(WorkflowDecision(
            intent: "image.edit", confidence: 0.75, probabilities: [:], provider: "test").requiresConfirmation())
    }

    func testWorkflowDecisionPreservesOrderedHistoricalInputs() throws {
        let decision = WorkflowDecision(
            intent: "image.edit", confidence: 0.9,
            probabilities: ["image.edit": 0.9, "image.generate": 0.1], provider: "test",
            inputResultIds: ["img_1", "img_3"])
        let decoded = try JSONDecoder().decode(
            WorkflowDecision.self, from: JSONEncoder().encode(decision))
        XCTAssertEqual(decoded.input_result_ids, ["img_1", "img_3"])
    }

    func testRouteRequestIdentityRejectsStaleAndReplacedRequests() {
        let context = RoutingContext(message: "edit it", currentResultId: "img_1", attachmentIds: [])
        let first = RouteRequestIdentity(context: context)
        XCTAssertTrue(first.accepts(context: context, requestID: first.id))
        XCTAssertFalse(first.accepts(
            context: RoutingContext(message: "edit it", currentResultId: "img_1", attachmentIds: ["img_2"]),
            requestID: first.id))

        let replacement = RouteRequestIdentity(context: context)
        XCTAssertFalse(replacement.accepts(context: context, requestID: first.id))
        XCTAssertTrue(replacement.accepts(context: context, requestID: replacement.id))
    }

    func testWorkflowRoundTripPreservesInputOrder() throws {
        let inputs = [
            MediaInput(id: "img_1", role: .source, order: 0),
            MediaInput(id: "img_2", role: .reference, order: 1),
            MediaInput(id: "img_3", role: .reference, order: 2),
        ]
        let plan = WorkflowPlan(steps: [WorkflowStep(
            operation: "image.edit", executor: "flux2-klein-4b", inputs: inputs,
            prompt: PromptSpec(user: "combine them"), parameters: WorkflowParameters(size: "1024x1024"))])
        let decoded = try JSONDecoder().decode(WorkflowPlan.self, from: JSONEncoder().encode(plan))
        XCTAssertEqual(decoded, plan)
        XCTAssertEqual(decoded.steps[0].inputs.map(\.id), ["img_1", "img_2", "img_3"])
    }

    func testRouterSettingsRequireTLSAwayFromLoopback() throws {
        XCTAssertNoThrow(try RouterProviderSettings(baseURL: "http://localhost:1234/v1", model: "router").validate())
        XCTAssertThrowsError(try RouterProviderSettings(baseURL: "http://example.com/v1", model: "router").validate())
        XCTAssertNoThrow(try RouterProviderSettings(baseURL: "https://example.com/v1", model: "router").validate())
        XCTAssertThrowsError(try RouterProviderSettings(baseURL: "https://example.com/v1", model: "").validate())
        let local = FileManager.default.temporaryDirectory.appendingPathComponent("local-llm-test-\(UUID())")
        try FileManager.default.createDirectory(at: local, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: local) }
        XCTAssertNoThrow(try RouterProviderSettings(
            kind: .localLLM, localModelPath: local.path, localRevision: "pinned-revision").validate())
        XCTAssertThrowsError(try RouterProviderSettings(
            kind: .localLLM, localModelPath: local.path, localRevision: "").validate())
    }
}

final class EditorTests: XCTestCase {
    func testEditorGeometryIsInvariantAcrossDisplayAndRetinaScales() {
        let small = EditorGeometry.fittedRect(imageWidth: 400, imageHeight: 200, in: CGSize(width: 300, height: 300))
        let large = EditorGeometry.fittedRect(imageWidth: 400, imageHeight: 200, in: CGSize(width: 600, height: 600))
        XCTAssertEqual(small, CGRect(x: 0, y: 75, width: 300, height: 150))
        XCTAssertEqual(large, CGRect(x: 0, y: 150, width: 600, height: 300))
        let logical = EditorGeometry.normalize(CGPoint(x: 75, y: 112.5), in: small)
        let retinaPixels = EditorGeometry.normalize(CGPoint(x: 150, y: 225), in: large)
        XCTAssertEqual(logical, NormalizedPoint(x: 0.25, y: 0.25))
        XCTAssertEqual(retinaPixels, logical)
        XCTAssertNil(EditorGeometry.normalize(CGPoint(x: 10, y: 10), in: small))
    }

    func testCropAndStrokeCoordinatesStayAligned() {
        let crop = NormalizedRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5)
        XCTAssertEqual(
            EditorGeometry.pixelCrop(crop, sourceWidth: 400, sourceHeight: 200),
            CGRect(x: 100, y: 50, width: 200, height: 100))
        let display = CGRect(x: 20, y: 40, width: 800, height: 400)
        let point = EditorGeometry.normalize(CGPoint(x: 420, y: 240), in: display)
        XCTAssertEqual(point, NormalizedPoint(x: 0.5, y: 0.5))
        XCTAssertEqual((point!.x - crop.x) / crop.width, 0.5, accuracy: 0.0001)
        XCTAssertEqual((point!.y - crop.y) / crop.height, 0.5, accuracy: 0.0001)
    }

    func testNormalizedCropIsClamped() {
        let crop = NormalizedRect(x: -0.2, y: 0.8, width: 2, height: 0.5)
        XCTAssertEqual(crop.x, 0)
        XCTAssertEqual(crop.y, 0.8)
        XCTAssertEqual(crop.width, 1)
        XCTAssertEqual(crop.height, 0.2, accuracy: 0.0001)
    }

    func testEditorDocumentPersistsWithoutChangingOriginal() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("dmlx-editor-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let store = EditorDocumentStore(root: root)
        let original = EditorDocument(baseResultId: "img_1", baseWidth: 400, baseHeight: 200)
        var edited = original
        edited.crop = NormalizedRect(x: 0.25, y: 0, width: 0.5, height: 1)
        try store.save(edited, sessionId: "session/one")
        XCTAssertEqual(store.load(sessionId: "session/one", resultId: "img_1"), edited)
        XCTAssertEqual(original.crop, .full)
    }

    func testRendererAppliesCropDimensions() throws {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(
            data: nil, width: 100, height: 80, bitsPerComponent: 8, bytesPerRow: 0,
            space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: 100, height: 80))
        let source = context.makeImage()!
        let document = EditorDocument(
            baseResultId: "img_1", baseWidth: 100, baseHeight: 80,
            crop: NormalizedRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5))
        let data = try EditorRenderer.render(document, source: source)
        let output = NSBitmapImageRep(data: data)
        XCTAssertEqual(output?.pixelsWide, 50)
        XCTAssertEqual(output?.pixelsHigh, 40)
    }

    func testRendererPreservesVerticalOrientation() throws {
        let context = CGContext(
            data: nil, width: 20, height: 30, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: 20, height: 15))
        context.setFillColor(CGColor(red: 0, green: 0, blue: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 15, width: 20, height: 15))
        let source = context.makeImage()!
        let expected = NSBitmapImageRep(cgImage: source)
        let document = EditorDocument(baseResultId: "asymmetric", baseWidth: 20, baseHeight: 30)
        let rendered = NSBitmapImageRep(data: try EditorRenderer.render(document, source: source))!

        for point in [(x: 10, y: 3), (x: 10, y: 26)] {
            let before = expected.colorAt(x: point.x, y: point.y)!
            let after = rendered.colorAt(x: point.x, y: point.y)!
            XCTAssertEqual(after.redComponent, before.redComponent, accuracy: 0.01)
            XCTAssertEqual(after.blueComponent, before.blueComponent, accuracy: 0.01)
        }
    }

    func testRendererPaintAndEraserOnlyChangeOverlay() throws {
        let context = CGContext(
            data: nil, width: 100, height: 100, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.setFillColor(CGColor(gray: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: 100, height: 100))
        let source = context.makeImage()!
        let points = [NormalizedPoint(x: 0.2, y: 0.5), NormalizedPoint(x: 0.8, y: 0.5)]
        let paint = PaintStroke(points: points, width: 0.2, color: PaintColor(red: 0, green: 0, blue: 0))
        var document = EditorDocument(baseResultId: "img", baseWidth: 100, baseHeight: 100, strokes: [paint])
        let painted = NSBitmapImageRep(data: try EditorRenderer.render(document, source: source))!
        XCTAssertLessThan(painted.colorAt(x: 50, y: 50)?.redComponent ?? 1, 0.2)

        document.strokes.append(PaintStroke(points: points, width: 0.2, erases: true))
        let erased = NSBitmapImageRep(data: try EditorRenderer.render(document, source: source))!
        XCTAssertGreaterThan(erased.colorAt(x: 50, y: 50)?.redComponent ?? 0, 0.9)
    }

    func testRendererKeepsTopAlignedStrokeAtTop() throws {
        let context = CGContext(
            data: nil, width: 100, height: 100, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.setFillColor(CGColor(gray: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: 100, height: 100))
        let stroke = PaintStroke(
            points: [NormalizedPoint(x: 0.2, y: 0.2), NormalizedPoint(x: 0.8, y: 0.2)],
            width: 0.1, color: PaintColor(red: 0, green: 0, blue: 0))
        let document = EditorDocument(
            baseResultId: "top-stroke", baseWidth: 100, baseHeight: 100, strokes: [stroke])
        let rendered = NSBitmapImageRep(data: try EditorRenderer.render(
            document, source: context.makeImage()!))!

        XCTAssertLessThan(rendered.colorAt(x: 50, y: 20)?.redComponent ?? 1, 0.2)
        XCTAssertGreaterThan(rendered.colorAt(x: 50, y: 80)?.redComponent ?? 0, 0.9)
    }
}

final class ModelGroupTests: XCTestCase {
    private func engineModel(_ id: String, repo: String, group: String, name: String, bits: Int?, downloaded: Bool,
                             tasks: [String] = ["t2i", "edit"], license: String = "apache-2.0") throws -> EngineModel {
        let json: [String: Any] = [
            "id": id, "name": name, "license": license, "tasks": tasks, "max_references": 3, "min_ram_gb": 16,
            "downloaded": downloaded, "resident": false, "pinned": false, "fits_this_mac": true, "repo": repo,
            "download": ["status": "idle", "total_bytes": 0, "done_bytes": 0, "progress": 0],
            "group": group, "group_name": name, "quantization": bits as Any,
        ]
        return try JSONDecoder().decode(EngineModel.self, from: JSONSerialization.data(withJSONObject: json))
    }

    private func suggestion(_ repo: String, group: String, name: String, bits: Int?, downloads: Int = 1) -> CatalogSuggestion {
        CatalogSuggestion(repo: repo, downloads: downloads, group: group, group_name: name, quantization: bits,
                          tasks: ["t2i", "edit"], license: "other", permissive_license: false, fits_this_mac: true)
    }

    func testBuildsOfOneModelAreOneGroupWithTheEnginesBuildsMarked() throws {
        let klein = try engineModel("flux2-klein-4b", repo: "mflux-community/flux2-klein-4b-mflux-q4",
                                    group: "flux-2-klein-4b", name: "FLUX.2 Klein 4B", bits: 4, downloaded: true)
        let suggestions = [
            suggestion("mflux-community/flux2-klein-4b-mflux-q8", group: "flux-2-klein-4b", name: "FLUX.2 Klein 4B", bits: 8, downloads: 5),
            suggestion("mflux-community/flux2-klein-4b-mflux-q4", group: "flux-2-klein-4b", name: "FLUX.2 Klein 4B", bits: 4, downloads: 9),
            suggestion("mflux-community/flux2-klein-4b-mflux-bf16", group: "flux-2-klein-4b", name: "FLUX.2 Klein 4B", bits: nil),
            suggestion("mflux-community/flux2-klein-9b-mflux-q4", group: "flux-2-klein-9b", name: "FLUX.2 Klein 9B", bits: 4),
        ]
        let groups = ModelCatalog.groups(models: [klein], suggestions: suggestions)
        XCTAssertEqual(groups.map(\.id), ["flux-2-klein-4b", "flux-2-klein-9b"])
        let first = groups[0]
        XCTAssertEqual(first.variants.map(\.quantization), [4, 8, nil])  // smallest first, full size last
        XCTAssertEqual(first.defaultVariant?.model?.id, "flux2-klein-4b")
        XCTAssertEqual(first.license, "apache-2.0")  // the engine's license wins over the card's
        XCTAssertTrue(first.isDownloaded)
        XCTAssertEqual(first.downloads, 15)
        XCTAssertEqual(groups[1].defaultVariant?.quantization, 4)
        XCTAssertFalse(groups[1].isAdded)
    }

    func testAnEngineBuildReplacesACatalogBuildOfTheSameBitsAndUnlistedModelsComeLast() throws {
        let qwen = try engineModel("qwen-image-2.1-q4", repo: "OsaurusAI/Qwen-Image-2.1-mflux-4bit",
                                   group: "qwen-image-2-1-base", name: "Qwen Image 2.1 Base", bits: 4, downloaded: false,
                                   license: "qwen-research")
        let zimage = try engineModel("z-image-turbo", repo: "mflux-community/z-image-turbo-mflux-q4",
                                     group: "z-image-turbo", name: "Z-Image Turbo", bits: 4, downloaded: false, tasks: ["t2i"])
        let suggestions = [
            suggestion("mflux-community/qwen-image-2-1-mflux-q4", group: "qwen-image-2-1-base", name: "Qwen Image 2.1 Base", bits: 4),
            suggestion("mflux-community/qwen-image-2-1-mflux-q8", group: "qwen-image-2-1-base", name: "Qwen Image 2.1 Base", bits: 8),
        ]
        let groups = ModelCatalog.groups(models: [qwen, zimage], suggestions: suggestions)
        XCTAssertEqual(groups.map(\.id), ["qwen-image-2-1-base", "z-image-turbo"])
        XCTAssertEqual(groups[0].variants.map(\.repo), ["OsaurusAI/Qwen-Image-2.1-mflux-4bit", "mflux-community/qwen-image-2-1-mflux-q8"])
        XCTAssertEqual(groups[0].defaultVariant?.model?.id, "qwen-image-2.1-q4")
        XCTAssertFalse(groups[1].supports(.edit))
    }

    func testWithoutA4BitBuildTheDefaultIsTheNextLargerOne() {
        let group = ModelCatalog.groups(models: [], suggestions: [
            suggestion("a/m-q3", group: "m", name: "M", bits: 3),
            suggestion("a/m-bf16", group: "m", name: "M", bits: nil),
            suggestion("a/m-q6", group: "m", name: "M", bits: 6),
        ])[0]
        XCTAssertEqual(group.defaultVariant?.repo, "a/m-q6")
    }
}

final class LayoutTests: XCTestCase {
    func testSameGeometryIgnoresWordsButNotPlaces() {
        let a = LayoutBox(id: "d", label: "FRIDAY", rect: NormalizedRect(x: 0.2, y: 0.4, width: 0.5, height: 0.1))
        var renamed = a
        renamed.label = "SATURDAY"
        XCTAssertTrue(LayoutBox.sameGeometry([a], [renamed]))
        var moved = a
        moved.place(NormalizedRect(x: 0.5, y: 0.88, width: 0.4, height: 0.08))
        XCTAssertFalse(LayoutBox.sameGeometry([a], [moved]))
        XCTAssertFalse(LayoutBox.sameGeometry([a], []))
    }

    func testMovingABoxClearsTheAgentsDescriptionButNotTheUsers() {
        var agents = LayoutBox(label: "FRIDAY 9 PM", desc: "smaller, below main title",
                               rect: NormalizedRect(x: 0.2, y: 0.37, width: 0.5, height: 0.13), byUser: false)
        agents.place(NormalizedRect(x: 0.56, y: 0.88, width: 0.4, height: 0.08))
        XCTAssertEqual(agents.desc, "")
        XCTAssertTrue(agents.byUser)

        var mine = LayoutBox(label: "sax", desc: "gold sax", descByUser: true,
                             rect: NormalizedRect(x: 0.1, y: 0.3, width: 0.3, height: 0.4))
        mine.place(NormalizedRect(x: 0.5, y: 0.3, width: 0.3, height: 0.4))
        XCTAssertEqual(mine.desc, "gold sax")
    }

    func testDecodesTheEnginesLayoutAndKeepsOldVersionsReadable() throws {
        let json = #"[{"id":"date","kind":"text","label":"FRIDAY 9 PM","box":[880,520,960,960],"desc":"","desc_by_user":false,"by_user":true},{"id":"a1","kind":"obj","label":"sax","box":[260,300,820,700]}]"#
        let boxes = try JSONDecoder().decode([LayoutBox].self, from: Data(json.utf8))
        XCTAssertEqual(boxes[0].kind, .text)
        XCTAssertTrue(boxes[0].byUser)
        XCTAssertFalse(boxes[1].byUser)
        XCTAssertEqual(boxes[0].rect.x, 0.52, accuracy: 1e-9)
        XCTAssertEqual(boxes[0].rect.height, 0.08, accuracy: 1e-9)
        // A version saved before task 17 has no layout fields.
        let old = #"{"id":"v","status":"done","prompt":"boat","model":"m","task":"t2i","size":"auto","sourceIds":[],"step":0,"totalSteps":0,"phase":"Done","loads":[]}"#
        let version = try JSONDecoder().decode(ImageVersion.self, from: Data(old.utf8))
        XCTAssertNil(version.layout)
        XCTAssertNil(version.note)
    }

    func testMovingABoxMakesItTheUsersAndRoundTrips() {
        var box = LayoutBox(kind: .obj, label: "fox", rect: NormalizedRect(x: 0.1, y: 0.2, width: 0.3, height: 0.4), byUser: false)
        XCTAssertEqual(box.box, [200, 100, 600, 400])
        box.place(NormalizedRect(x: 0.7, y: 0.7, width: 0.25, height: 0.25))
        XCTAssertTrue(box.byUser)
        XCTAssertEqual(box.box, [700, 700, 950, 950])
        XCTAssertEqual(LayoutWords.location(box), "bottom right")
    }

    func testOnlySendableBoxesGoWithTheRequest() throws {
        let empty = LayoutBox(kind: .text, label: "  ", rect: .full)
        var described = LayoutBox(kind: .obj, label: "sax", desc: "  ", descByUser: true, rect: .full)
        described.label = " sax "
        let sent = LayoutBox.sendable([empty, described])
        XCTAssertEqual(sent.count, 1)
        XCTAssertEqual(sent[0].label, "sax")
        XCTAssertFalse(sent[0].descByUser)  // an empty description is the agent's to write

        let parameters = WorkflowParameters(size: "1024x768", layout: sent)
        let encoded = try JSONSerialization.jsonObject(with: JSONEncoder().encode(parameters)) as? [String: Any]
        let first = (encoded?["layout"] as? [[String: Any]])?.first
        XCTAssertEqual(first?["by_user"] as? Bool, true)
        XCTAssertEqual(first?["desc_by_user"] as? Bool, false)
        XCTAssertEqual(first?["box"] as? [Int], [0, 0, 1000, 1000])
    }
}

final class RouteBodyTests: XCTestCase {
    /// Auto's own model goes as `auto_models`, apart from the user's pick, so the engine may swap it when it
    /// won't fit; the size goes too, for the engine's memory check (task 17).
    func testAutoModelsAndSizeAreSentApartFromTheUsersPick() {
        let body = EngineClient.routeBody(
            message: "make it warmer", currentResultId: "img_1", currentPrompt: nil, attachments: [],
            selectedModels: ["image.generate": "qwen-image"], autoModels: ["image.edit": "flux2-klein-4b"],
            size: "1216x832", plan: true)
        XCTAssertEqual(body["selected_models"] as? [String: String], ["image.generate": "qwen-image"])
        XCTAssertEqual(body["auto_models"] as? [String: String], ["image.edit": "flux2-klein-4b"])
        XCTAssertEqual(body["size"] as? String, "1216x832")
    }

    func testNothingExtraIsSentWithoutThem() {
        let body = EngineClient.routeBody(message: "a fox", currentResultId: nil, currentPrompt: nil, attachments: [])
        XCTAssertNil(body["auto_models"])
        XCTAssertNil(body["size"])
        XCTAssertNil(body["selected_models"])
        XCTAssertNil(body["candidates"])
    }

    /// Without the agent, a send with one possible workflow (an attached photo, "make it warmer") still asks
    /// the engine, for that workflow only, so the engine can swap an Auto model that won't fit (test round 2).
    func testAOneWorkflowSendAsksOnlyForThatWorkflow() {
        let body = EngineClient.routeBody(
            message: "make it warmer", currentResultId: nil, currentPrompt: nil, attachments: ["att_1"],
            autoModels: ["image.edit": "flux2-klein-9b"], size: "1024x1024", candidates: ["image.edit"])
        XCTAssertEqual(body["candidates"] as? [String], ["image.edit"])
        XCTAssertNil(body["plan"])
    }

    func testTheEnginesCheckedModelIsKeptWhenItAgreesOnTheWorkflow() {
        let local = WorkflowDecision(intent: "image.edit", confidence: 1, probabilities: ["image.edit": 1],
                                     provider: "deterministic")
        var engine = local
        engine.model = "flux2-klein-4b"
        XCTAssertEqual(local.checked(by: engine).model, "flux2-klein-4b")
        XCTAssertNil(local.checked(by: nil).model)  // engine unreachable: the app's own decision runs
        var other = engine
        other.intent = "ambiguous"
        XCTAssertEqual(local.checked(by: other), local)
    }

    /// Test round 3: with the agent on, a one-workflow hint returned before asking the engine and named the
    /// Auto model the engine then swapped out ("Klein9B AutoTest" named, Klein 4B ran). It now asks too.
    func testTheHintAsksForOneWorkflowOnAutoAlsoWithTheAgent() {
        let query = RoutePreviewQuery.make(structural: "image.edit", onAuto: true, agentInstalled: true, layout: nil)
        XCTAssertEqual(query?.candidates, ["image.edit"])
        XCTAssertNil(RoutePreviewQuery.make(structural: "image.edit", onAuto: false, agentInstalled: true, layout: nil))
        XCTAssertNil(RoutePreviewQuery.make(structural: nil, onAuto: true, agentInstalled: true, layout: nil))
        XCTAssertEqual(RoutePreviewQuery.make(structural: nil, onAuto: true, agentInstalled: false, layout: nil),
                       RoutePreviewQuery(candidates: nil, layout: nil))
    }

    /// Test round 3: the hint's route sent no layout, so a layout send's hint named a model that can't place
    /// things while the send ran a layout model. The hint now sends the composer's boxes too.
    func testTheHintSendsTheComposersLayout() {
        let boxes = [LayoutBox(id: "d", label: "FRIDAY", rect: NormalizedRect(x: 0.2, y: 0.4, width: 0.5, height: 0.1))]
        XCTAssertEqual(RoutePreviewQuery.make(structural: nil, onAuto: true, agentInstalled: false, layout: boxes)?.layout,
                       boxes)
        XCTAssertEqual(RoutePreviewQuery.make(structural: "image.generate", onAuto: true, agentInstalled: true,
                                              layout: boxes)?.layout, boxes)
        XCTAssertNil(RoutePreviewQuery.make(structural: nil, onAuto: true, agentInstalled: false, layout: [])?.layout)
    }

    /// The hint names what the send runs: the user's model; on Auto the engine's swap, else the app's Auto
    /// model; and when the agent picks on send among several, those.
    func testTheHintNamesTheModelThatRuns() {
        var decision = WorkflowDecision(intent: "image.edit", confidence: 1, probabilities: ["image.edit": 1],
                                        provider: "deterministic")
        let runnable: Set = ["flux2-klein-4b", "klein9b-autotest"]
        XCTAssertEqual(RouteModel.named(chosen: nil, decision: decision, runnable: runnable, auto: "klein9b-autotest"),
                       ["klein9b-autotest"])
        decision.model = "flux2-klein-4b"  // the engine's swap: the 9B won't fit
        XCTAssertEqual(RouteModel.named(chosen: nil, decision: decision, runnable: runnable, auto: "klein9b-autotest"),
                       ["flux2-klein-4b"])
        XCTAssertEqual(RouteModel.running(chosen: nil, decision: decision, runnable: runnable, auto: "klein9b-autotest"),
                       "flux2-klein-4b")
        XCTAssertEqual(RouteModel.named(chosen: "klein9b-autotest", decision: decision, runnable: runnable, auto: nil),
                       ["klein9b-autotest"])
        decision.model = nil
        decision.agent_choices = ["flux2-klein-4b", "klein9b-autotest", "not-downloaded"]
        XCTAssertEqual(RouteModel.named(chosen: nil, decision: decision, runnable: runnable, auto: "klein9b-autotest"),
                       ["flux2-klein-4b", "klein9b-autotest"])
        XCTAssertEqual(RouteModel.named(chosen: "flux2-klein-4b", decision: decision, runnable: runnable, auto: nil),
                       ["flux2-klein-4b"])  // the user's model is not the agent's to change
    }

    /// Test round 4: until the engine checks the app's own decision, the hint names no model on Auto
    /// (it named the Auto model the engine then swapped out); the user's model is named at once.
    func testTheHintNamesNoAutoModelUntilTheEngineChecksIt() {
        let decision = WorkflowDecision(intent: "image.edit", confidence: 1, probabilities: ["image.edit": 1],
                                        provider: "deterministic")
        let runnable: Set = ["flux2-klein-4b", "klein9b-autotest"]
        XCTAssertEqual(RouteModel.named(chosen: nil, decision: decision, runnable: runnable,
                                        auto: "klein9b-autotest", checked: false), [])
        XCTAssertEqual(RouteModel.named(chosen: "klein9b-autotest", decision: decision, runnable: runnable,
                                        auto: nil, checked: false), ["klein9b-autotest"])
    }

    /// Test round 4: the agent-on hint listed five models. Two are named, then "or another model".
    func testTheAgentChoicesHintIsShort() {
        XCTAssertEqual(RouteModel.agentPicks(["FLUX.2 Klein 4B", "FLUX.2 Klein 9B"]),
                       "the agent picks FLUX.2 Klein 4B or FLUX.2 Klein 9B")
        XCTAssertEqual(RouteModel.agentPicks(["FLUX.2 Klein 4B", "FLUX.2 Klein 9B", "Klein9B AutoTest"]),
                       "the agent picks FLUX.2 Klein 4B, FLUX.2 Klein 9B or another model")
    }

    func testAgentChoicesDecodeFromTheEngine() throws {
        let json = #"{"intent":"image.edit","confidence":1,"probabilities":{},"provider":"deterministic","agent_choices":["a","b"]}"#
        let decision = try JSONDecoder().decode(WorkflowDecision.self, from: Data(json.utf8))
        XCTAssertEqual(decision.agent_choices, ["a", "b"])
    }
}
