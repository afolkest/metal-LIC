# metal-LIC

GPU implementation of Line Integral Convolution on Apple Silicon, written in Swift + Metal. 
Capable of real-time performance and 4k resolution. 

## What it does

Takes an input texture and a vector field, traces streamlines via RK2 integration with a Hann-windowed kernel, and outputs a grayscale LIC image. The entire pipeline runs as a Metal compute shader with no CPU readback.

## Performance

Measured GPU time on M1 Pro, single pass, kernel length 30 pixels:

| Resolution | Uniform field | Vortex field |
|---|---|---|
| 1080p | 238 fps | 179 fps |
| 2048x2048 | 118 fps | 88 fps |
| 3840x2160 | 60 fps | 46 fps |

Throughput scales linearly with pixel count. Multi-pass (3x) at 2K runs at ~35 fps.

## Usage

Swift Package Manager library. Add as a dependency:

```swift
.package(url: "https://github.com/<you>/metal-LIC", branch: "main")
```

The package provides `LICEncoder` for single-shot encoding and `LICDispatcher` for pipelined multi-buffered dispatch.

## Demo app

`LICDemo` is a SwiftUI app that pairs the LIC engine with a forest fire cellular automaton to give time-dependent noise.
This lets you run real-time simulations. Run it from Xcode or `swift run LICDemo`.

## Tests

```bash
swift test                                          # core tests
RUN_BENCHMARKS=1 swift test --filter LICBenchmarkTests  # GPU benchmarks
RUN_PARITY=1 swift test --filter LICParityTests         # bryLIC reference comparison
```
