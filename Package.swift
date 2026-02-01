// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "MetalLIC",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "MetalLIC", targets: ["MetalLIC"]),
    ],
    targets: [
        .target(
            name: "MetalLIC",
            resources: [.copy("Shaders")]
        ),
        .testTarget(
            name: "MetalLICTests",
            dependencies: ["MetalLIC"],
            exclude: ["Fixtures"]
        ),
    ]
)
