import Foundation
import Metal
import MetalPerformanceShaders

/// A comprehensive example demonstrating key features of MetalNeuralKit
public class ComprehensiveExample {
    /// Metal device
    private let device: MTLDevice
    
    /// Command queue
    private let commandQueue: MTLCommandQueue
    
    /// Neural network
    private var network: NeuralNetwork
    
    /// M4 optimizer
    private let m4Optimizer = M4Optimizer.shared
    
    /// Architecture tracker
    private let architectureTracker = ArchitectureTracker.shared
    
    /// CUDA bridge
    private let cudaBridge = CUDABridge.shared
    
    /// Output directory for saving results
    private let outputDirectory: String
    
    /// Example name
    private let exampleName = "ComprehensiveExample"
    
    /// Initialize the example
    /// - Parameter outputDirectory: Directory to save results
    public init(outputDirectory: String = "./output") {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        
        self.device = device
        
        guard let commandQueue = device.makeCommandQueue() else {
            fatalError("Failed to create command queue")
        }
        
        self.commandQueue = commandQueue
        
        // Create neural network
        self.network = NeuralNetwork(name: "ComprehensiveNetwork", device: device)
        
        // Set output directory
        self.outputDirectory = outputDirectory
        
        // Create output directory if it doesn't exist
        let fileManager = FileManager.default
        if !fileManager.fileExists(atPath: outputDirectory) {
            do {
                try fileManager.createDirectory(atPath: outputDirectory, withIntermediateDirectories: true)
            } catch {
                print("Error creating output directory: \(error)")
            }
        }
    }
    
    /// Run the comprehensive example
    public func run() {
        print("🚀 Starting MetalNeuralKit Comprehensive Example")
        print("===============================================")
        
        // Check device capabilities
        checkDeviceCapabilities()
        
        // Build a complex network
        buildComplexNetwork()
        
        // Track architecture changes
        trackArchitectureChanges()
        
        // Optimize for M4
        optimizeForM4()
        
        // Run inference
        runInference()
        
        // Compare with CUDA (simulated)
        compareToCUDA()
        
        // Export results
        exportResults()
        
        print("\n✅ Comprehensive Example Completed")
        print("===============================================")
    }
    
    /// Check device capabilities
    private func checkDeviceCapabilities() {
        print("\n📊 Device Capabilities:")
        print("-----------------------------------------------")
        print("Device: \(device.name)")
        print("Has Neural Engine: \(device.hasNeuralEngine)")
        
        let m4CapabilityInfo = m4Optimizer.getM4CapabilityInfo()
        print("Is M4 Chip: \(m4CapabilityInfo["isM4Chip"] as? Bool ?? false)")
        print("Recommended Working Set Size: \(ByteCountFormatter.string(fromByteCount: Int64(m4CapabilityInfo["recommendedMaxWorkingSetSize"] as? Int ?? 0), countStyle: .file))")
        print("Supports Unified Memory: \(m4CapabilityInfo["supportsUnifiedMemory"] as? Bool ?? false)")
        
        print("\nOptimization Tips:")
        let tips = m4Optimizer.getM4OptimizationTips()
        for (index, tip) in tips.enumerated() {
            print("  \(index + 1). \(tip)")
        }
    }
    
    /// Build a complex neural network
    private func buildComplexNetwork() {
        print("\n🏗️ Building Complex Network:")
        print("-----------------------------------------------")
        
        // Initial snapshot
        architectureTracker.recordSnapshot(network: network, reason: .initial)
        
        // Add a convolution layer
        let conv1 = ConvolutionLayer(
            name: "conv1",
            inputChannels: 3,
            outputChannels: 16,
            kernelSize: (3, 3),
            stride: (1, 1),
            padding: (1, 1),
            useNeuralEngine: true,
            device: device
        )
        network.addLayer(conv1)
        architectureTracker.recordSnapshot(network: network, reason: .layerAdded)
        print("Added convolution layer: \(conv1.name)")
        
        // Add a pooling layer
        let pool1 = PoolingLayer(
            name: "pool1",
            kernelSize: (2, 2),
            stride: (2, 2),
            type: .max,
            device: device
        )
        network.addLayer(pool1)
        architectureTracker.recordSnapshot(network: network, reason: .layerAdded)
        print("Added pooling layer: \(pool1.name)")
        
        // Add another convolution layer
        let conv2 = ConvolutionLayer(
            name: "conv2",
            inputChannels: 16,
            outputChannels: 32,
            kernelSize: (3, 3),
            stride: (1, 1),
            padding: (1, 1),
            useNeuralEngine: true,
            device: device
        )
        network.addLayer(conv2)
        architectureTracker.recordSnapshot(network: network, reason: .layerAdded)
        print("Added convolution layer: \(conv2.name)")
        
        // Add another pooling layer
        let pool2 = PoolingLayer(
            name: "pool2",
            kernelSize: (2, 2),
            stride: (2, 2),
            type: .max,
            device: device
        )
        network.addLayer(pool2)
        architectureTracker.recordSnapshot(network: network, reason: .layerAdded)
        print("Added pooling layer: \(pool2.name)")
        
        // Add a fully connected layer
        let fc1 = FullyConnectedLayer(
            name: "fc1",
            inputSize: 32 * 8 * 8, // Assuming input was 32x32
            outputSize: 128,
            useNeuralEngine: true,
            device: device
        )
        network.addLayer(fc1)
        architectureTracker.recordSnapshot(network: network, reason: .layerAdded)
        print("Added fully connected layer: \(fc1.name)")
        
        // Add a fully connected layer for output
        let fc2 = FullyConnectedLayer(
            name: "fc2",
            inputSize: 128,
            outputSize: 10, // 10 classes
            useNeuralEngine: true,
            device: device
        )
        network.addLayer(fc2)
        architectureTracker.recordSnapshot(network: network, reason: .layerAdded)
        print("Added fully connected layer: \(fc2.name)")
        
        print("Network built with \(network.layers.count) layers")
    }
    
    /// Track architecture changes
    private func trackArchitectureChanges() {
        print("\n📝 Tracking Architecture Changes:")
        print("-----------------------------------------------")
        
        // Get all snapshots
        let snapshots = architectureTracker.getSnapshots(networkId: network.id)
        print("Total snapshots: \(snapshots.count)")
        
        // Print snapshot timeline
        print("\nSnapshot Timeline:")
        for (index, snapshot) in snapshots.enumerated() {
            let dateFormatter = DateFormatter()
            dateFormatter.dateFormat = "HH:mm:ss.SSS"
            let timeString = dateFormatter.string(from: snapshot.timestamp)
            
            print("  \(index + 1). [\(timeString)] \(snapshot.reason.rawValue) - Layers: \(snapshot.layerCount)")
        }
        
        // Modify a layer to demonstrate tracking
        if let conv1 = network.getLayer(at: 0) as? ConvolutionLayer {
            // Change kernel size
            conv1.parameters["kernelSize"] = (5, 5)
            architectureTracker.recordSnapshot(network: network, reason: .layerModified)
            print("\nModified conv1 layer: changed kernel size to (5, 5)")
        }
        
        // Get evolution data
        if let evolution = architectureTracker.getEvolutionData(networkId: network.id) {
            print("\nNetwork Evolution:")
            print("  Created: \(evolution.createdAt)")
            print("  Last Modified: \(evolution.lastModifiedAt)")
            print("  Layer Changes: \(evolution.layerChanges.count)")
        }
    }
    
    /// Optimize for M4
    private func optimizeForM4() {
        print("\n⚡ Optimizing for M4:")
        print("-----------------------------------------------")
        
        // Configure M4 optimizer
        m4Optimizer.configure(useHalfPrecision: true)
        print("Configured M4 optimizer with half-precision")
        
        // Optimize network
        let result = m4Optimizer.optimizeNetwork(network)
        
        if result {
            print("Successfully optimized network for M4")
            
            // Get new snapshots
            let snapshots = architectureTracker.getSnapshots(networkId: network.id)
            
            // Print last snapshot info
            if let lastSnapshot = snapshots.last {
                print("Last snapshot reason: \(lastSnapshot.reason.rawValue)")
                print("Network now has \(lastSnapshot.layerCount) layers")
            }
        } else {
            print("Failed to optimize network (likely not running on M4 chip)")
        }
    }
    
    /// Run inference
    private func runInference() {
        print("\n🧠 Running Inference:")
        print("-----------------------------------------------")
        
        // Create a command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            print("Failed to create command buffer")
            return
        }
        
        // Create a sample input (32x32 RGB image)
        let inputWidth = 32
        let inputHeight = 32
        let inputChannels = 3
        
        let inputDesc = MPSImageDescriptor(
            channelFormat: .float16,
            width: inputWidth,
            height: inputHeight,
            featureChannels: inputChannels
        )
        
        guard let inputImage = MPSImage(device: device, imageDescriptor: inputDesc) else {
            print("Failed to create input image")
            return
        }
        
        // Fill with random data
        let inputCount = inputWidth * inputHeight * inputChannels
        var inputData = [Float](repeating: 0, count: inputCount)
        for i in 0..<inputCount {
            inputData[i] = Float.random(in: 0...1)
        }
        
        inputImage.writeBytes(
            UnsafeRawPointer(inputData),
            dataLayout: .featureChannelsxHeightxWidth,
            imageIndex: 0
        )
        
        // Measure inference time
        let start = Date()
        
        var currentImage = inputImage
        var outputImage: MPSImage? = nil
        
        // Forward pass through each layer
        for (index, layer) in network.layers.enumerated() {
            if outputImage == nil || index == network.layers.count - 1 {
                // Create output image for the layer
                let outputDesc: MPSImageDescriptor
                
                if let convLayer = layer as? ConvolutionLayer {
                    // Calculate output dimensions for convolution
                    let padH = convLayer.parameters["padding"] as? (Int, Int) ?? (0, 0)
                    let strideH = convLayer.parameters["stride"] as? (Int, Int) ?? (1, 1)
                    let kernelH = convLayer.parameters["kernelSize"] as? (Int, Int) ?? (3, 3)
                    let outputChannels = convLayer.parameters["outputChannels"] as? Int ?? 1
                    
                    let outWidth = (inputWidth + 2 * padH.0 - kernelH.0) / strideH.0 + 1
                    let outHeight = (inputHeight + 2 * padH.1 - kernelH.1) / strideH.1 + 1
                    
                    outputDesc = MPSImageDescriptor(
                        channelFormat: .float16,
                        width: outWidth,
                        height: outHeight,
                        featureChannels: outputChannels
                    )
                } else if let poolLayer = layer as? PoolingLayer {
                    // Calculate output dimensions for pooling
                    let kernelH = poolLayer.parameters["kernelSize"] as? (Int, Int) ?? (2, 2)
                    let strideH = poolLayer.parameters["stride"] as? (Int, Int) ?? (2, 2)
                    
                    let outWidth = (inputWidth - kernelH.0) / strideH.0 + 1
                    let outHeight = (inputHeight - kernelH.1) / strideH.1 + 1
                    
                    outputDesc = MPSImageDescriptor(
                        channelFormat: .float16,
                        width: outWidth,
                        height: outHeight,
                        featureChannels: currentImage.featureChannels
                    )
                } else if let fcLayer = layer as? FullyConnectedLayer {
                    // Output dimensions for fully connected
                    let outputSize = fcLayer.parameters["outputSize"] as? Int ?? 10
                    
                    outputDesc = MPSImageDescriptor(
                        channelFormat: .float16,
                        width: 1,
                        height: 1,
                        featureChannels: outputSize
                    )
                } else {
                    // Default output
                    outputDesc = MPSImageDescriptor(
                        channelFormat: .float16,
                        width: currentImage.width,
                        height: currentImage.height,
                        featureChannels: currentImage.featureChannels
                    )
                }
                
                outputImage = MPSImage(device: device, imageDescriptor: outputDesc)
            }
            
            guard let output = outputImage else {
                print("Failed to create output image")
                return
            }
            
            // Perform the layer forward pass
            // In a real implementation, this would use the layer's forward method
            // Here we'll just simulate it
            print("  Processing layer: \(layer.name)")
            
            // Simulate processing delay
            usleep(5000) // 5ms
            
            // Update for next iteration
            currentImage = output
        }
        
        let end = Date()
        let inferenceTime = end.timeIntervalSince(start)
        
        print("\nInference completed in \(inferenceTime * 1000) ms")
        
        // Print output dimensions
        if let finalOutput = outputImage {
            print("Output shape: \(finalOutput.width)x\(finalOutput.height)x\(finalOutput.featureChannels)")
        }
    }
    
    /// Compare with CUDA (simulated)
    private func compareToCUDA() {
        print("\n🔍 Comparing with CUDA (Simulated):")
        print("-----------------------------------------------")
        
        print("CUDA available: \(cudaBridge.isCUDAAvailable() ? "Yes" : "No (simulated)")")
        
        // Create a dummy comparison
        let comparison = cudaBridge.createDummyComparisonResult(networkName: network.name)
        
        // Print comparison results
        print("\nComparison Results:")
        print("  Metal Device: \(comparison.metalDeviceInfo.name)")
        print("  CUDA Device: \(comparison.cudaDeviceInfo.name)")
        print("  Metal Avg Time: \(comparison.averageMetalExecutionTime * 1000) ms")
        print("  CUDA Avg Time: \(comparison.averageCUDAExecutionTime * 1000) ms")
        print("  Speedup Factor: \(comparison.speedupFactor)x")
        print("  Metal Memory: \(ByteCountFormatter.string(fromByteCount: comparison.metalMemoryUsage, countStyle: .file))")
        print("  CUDA Memory: \(ByteCountFormatter.string(fromByteCount: comparison.cudaMemoryUsage, countStyle: .file))")
        print("  Memory Efficiency: \(comparison.memoryEfficiencyFactor)x")
        
        // Export comparison
        let comparisonPath = "\(outputDirectory)/\(exampleName)_comparison.json"
        if cudaBridge.exportComparisonReport(result: comparison, filePath: comparisonPath) {
            print("\nExported comparison report to: \(comparisonPath)")
        }
    }
    
    /// Export results
    private func exportResults() {
        print("\n📦 Exporting Results:")
        print("-----------------------------------------------")
        
        // Export snapshots
        let snapshotsPath = "\(outputDirectory)/\(exampleName)_snapshots.json"
        if architectureTracker.exportSnapshots(networkId: network.id, filePath: snapshotsPath) {
            print("Exported architecture snapshots to: \(snapshotsPath)")
        }
        
        // Export evolution data
        let evolutionPath = "\(outputDirectory)/\(exampleName)_evolution.json"
        if architectureTracker.exportEvolutionData(networkId: network.id, filePath: evolutionPath) {
            print("Exported evolution data to: \(evolutionPath)")
        }
        
        print("\nAll results exported to \(outputDirectory)")
    }
}

/// Pooling layer implementation for example
class PoolingLayer: BaseLayer {
    init(name: String, kernelSize: (Int, Int), stride: (Int, Int), type: PoolingType, device: MTLDevice) {
        super.init(name: name, type: .pooling)
        
        // Set parameters
        parameters["kernelSize"] = kernelSize
        parameters["stride"] = stride
        parameters["type"] = type.rawValue
    }
    
    enum PoolingType: String {
        case max
        case average
    }
}

/// Fully connected layer implementation for example
class FullyConnectedLayer: BaseLayer {
    init(name: String, inputSize: Int, outputSize: Int, useNeuralEngine: Bool, device: MTLDevice) {
        super.init(name: name, type: .fullyConnected)
        
        // Set parameters
        parameters["inputSize"] = inputSize
        parameters["outputSize"] = outputSize
        parameters["useNeuralEngine"] = useNeuralEngine
    }
}

/// Main entry point for running the example
public func runComprehensiveExample() {
    let example = ComprehensiveExample()
    example.run()
} 