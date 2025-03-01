# MetalNeuralKit

A Swift library for building and optimizing neural networks on Apple Silicon using the Metal Neural Engine and Metal Performance Shaders.

## Features

- Optimized for Apple Silicon M-series chips, especially the M4 and Neural Engine
- Architecture tracking: Records and tracks neural network structure changes over time
- Performance benchmarking: Compares Metal performance against NVIDIA CUDA implementations
- Training and inference optimizations for M4 architecture

## Requirements

- macOS 14.0+ (Sonoma or newer)
- Apple Silicon Mac with M-series chip (optimized for M4)
- Xcode 15.0+
- Swift 6.0+

## Installation

### Swift Package Manager

Add the following to your `Package.swift` file:

```swift
dependencies: [
    .package(url: "https://github.com/yourusername/MetalNeuralKit.git", from: "0.1.0")
]
```

## Architecture

MetalNeuralKit consists of several key components:

1. **Core Neural Network**: Metal-optimized neural network primitives leveraging MPS and the Neural Engine
2. **Architecture Tracking**: Tools to record and analyze neural network structure over time
3. **Performance Benchmarking**: Utilities to compare performance with CUDA implementations
4. **M4 Optimizations**: Specific optimizations for the M4 Neural Engine

## Usage

```swift
import MetalNeuralKit

// Create a neural network
let network = MetalNeuralNetwork()

// Add layers
network.add(layer: ConvolutionLayer(inputChannels: 3, outputChannels: 16, kernelSize: 3))
network.add(layer: ReLULayer())
network.add(layer: PoolingLayer(type: .max, kernelSize: 2, stride: 2))
network.add(layer: FullyConnectedLayer(inputSize: 16, outputSize: 10))

// Train the network
network.train(data: trainData, labels: trainLabels, epochs: 10)

// Compare with CUDA implementation
let benchmarker = PerformanceBenchmarker()
let results = benchmarker.compare(network: network, cudaNetwork: cudaNetwork)
```

## License

This library is available under the MIT license. See the LICENSE file for more info. 