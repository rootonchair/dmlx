// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "dMLX",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "dMLX", targets: ["dMLX"]),
        .library(name: "DMLXKit", targets: ["DMLXKit"]),
    ],
    targets: [
        .target(
            name: "DMLXKit",
            linkerSettings: [.linkedLibrary("sqlite3")]
        ),
        .executableTarget(
            name: "dMLX",
            dependencies: ["DMLXKit"]
        ),
        .testTarget(
            name: "DMLXKitTests",
            dependencies: ["DMLXKit"]
        ),
    ],
    swiftLanguageModes: [.v5]
)
