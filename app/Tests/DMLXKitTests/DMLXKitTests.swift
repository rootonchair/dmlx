import XCTest

@testable import DMLXKit

final class RoutingTests: XCTestCase {
    func testEmptyIsNone() {
        XCTAssertEqual(Router.route("   ", RoutingContext()), .none)
    }

    func testFirstMessageGenerates() {
        XCTAssertEqual(Router.route("a boat on a lake", RoutingContext()), .generate)
    }

    func testFollowUpChangeEditsLatest() {
        let ctx = RoutingContext(latestResultId: "img_1", latestTurnId: "t1")
        XCTAssertEqual(Router.route("make it night", ctx), .edit(sourceIds: ["img_1"], fromTurn: "t1"))
        XCTAssertEqual(Router.route("Add a lantern on the boat", ctx), .edit(sourceIds: ["img_1"], fromTurn: "t1"))
    }

    func testFreshDescriptionGeneratesEvenWithHistory() {
        let ctx = RoutingContext(latestResultId: "img_1", latestTurnId: "t1")
        XCTAssertEqual(Router.route("a fox logo, flat, two colors", ctx), .generate)
        XCTAssertEqual(Router.route("new image of a red car", ctx), .generate)
    }

    func testExplicitTargetWins() {
        let ctx = RoutingContext(targetResultId: "img_2", targetTurnId: "t2", latestResultId: "img_9", latestTurnId: "t9")
        XCTAssertEqual(Router.route("a completely different scene", ctx), .edit(sourceIds: ["img_2"], fromTurn: "t2"))
    }

    func testAttachmentsAreReferences() {
        let ctx = RoutingContext(targetResultId: "img_2", targetTurnId: "t2", attachmentIds: ["up_1", "img_2"])
        XCTAssertEqual(Router.route("put this hat on her", ctx), .edit(sourceIds: ["img_2", "up_1"], fromTurn: "t2"))
        let onlyAttach = RoutingContext(attachmentIds: ["up_1"])
        XCTAssertEqual(Router.route("turn this into a watercolor", onlyAttach), .edit(sourceIds: ["up_1"], fromTurn: nil))
    }

    func testVideoIsRecognised() {
        XCTAssertEqual(Router.route("animate it, 3 seconds", RoutingContext(latestResultId: "img_1")), .video)
    }
}

final class StoreTests: XCTestCase {
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
