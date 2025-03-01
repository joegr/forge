// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MetalNeuralKit",
    platforms: [
        .macOS(.v14)  // Targeting macOS 14 (Sonoma) or later with M series chips
    ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "MetalNeuralKit",
            targets: ["MetalNeuralKit"]),
        .executable(
            name: "MetalNeuralKitDemo",
            targets: ["MetalNeuralKitDemo"])
    ],
    dependencies: [
        // No external dependencies initially, we'll use system frameworks
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "MetalNeuralKit",
            dependencies: [],
            resources: [
                .process("Resources") // For any model resources we might need
            ]),
        .executableTarget(
            name: "MetalNeuralKitDemo",
            dependencies: ["MetalNeuralKit"]),
        .testTarget(
            name: "MetalNeuralKitTests",
            dependencies: ["MetalNeuralKit"]
        ),
    ]
)
