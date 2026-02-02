<video src="https://github.com/user-attachments/assets/ebbd32f1-bf2a-4ad8-a226-b7ed347f0582" autoplay loop muted playsinline></video>

# metal-LIC

GPU implementation of Line Integral Convolution on Apple Silicon, written in Swift + Metal. 
Capable of real-time performance at 4k resolution (~40 fps in simple tests on my Apple M1 Pro machine). 

## What it does

It takes an input texture and a vector field, convolves the texture along the streamlines of the vector field, and outputs a grayscale LIC image. The entire pipeline runs as a Metal compute shader.

Thanks to the performance of the implementation, it is possible to assign time-dynamics to the input texture that is getting convolved, giving
a host of possible dynamic visualizations of the (static) vector field that do not flicker.  

## Performance

Measured GPU time on M1 Pro, single convolution pass, kernel length 30 pixels:

| Resolution | Uniform field | Vortex field |
|---|---|---|
| 1080p | 238 fps | 179 fps |
| 2048x2048 | 118 fps | 88 fps |
| 3840x2160 | 60 fps | 46 fps |

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
