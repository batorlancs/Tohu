// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Tohu",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "TohuCore", targets: ["TohuCore"]),
        .executable(name: "tohu", targets: ["TohuCLI"]),
        .executable(name: "TohuApp", targets: ["TohuApp"]),
    ],
    targets: [
        // Core ML model + decoding, a port of tohu/restore.py. Resources (models/export/) are loaded
        // from a directory at runtime, so the same code serves the app bundle, the CLI and the tests.
        .target(name: "TohuCore"),
        .executableTarget(name: "TohuCLI", dependencies: ["TohuCore"]),
        // Menu-bar app: keeps the model loaded, serves the local HTTP API, hotkeys, writer panel.
        // Packaged into Tohu.app by scripts/build-app.sh.
        .executableTarget(name: "TohuApp", dependencies: ["TohuCore"]),
        .testTarget(name: "TohuCoreTests", dependencies: ["TohuCore"]),
    ]
)
