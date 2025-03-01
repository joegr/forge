import Foundation
import Metal
import MetalPerformanceShaders

/// Example usage of the MetalNeuralKit library
public class SimpleExample {
    /// Run a simple example of using the library
    public static func runExample() {
        print("Starting MetalNeuralKit simple example...")
        
        // Get Metal device
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Metal is not supported on this device")
            return
        }
        
        // Check for Neural Engine support
        var hasNeuralEngine = false
        if #available(macOS 14.0, *) {
            if let accelerationInfo = device.accelerationInfo {
                hasNeuralEngine = accelerationInfo.supportsAppleNeuralEngine
                print("Neural Engine support: \(hasNeuralEngine)")
            }
        }
        
        // Create a neural network
        let network = NeuralNetwork(name: "SimpleConvNet", device: device)
        
        // Add layers to the network
        network.add(layer: ConvolutionLayer(
            device: device,
            inputChannels: 3,
            outputChannels: 16,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            useNeuralEngine: true
        ))
        
        // ...additional layers would be added here
        
        // Create a sample input image
        let imageWidth = 224
        let imageHeight = 224
        let channelCount = 3
        
        let imageDesc = MPSImageDescriptor(
            channelFormat: .float16,
            width: imageWidth,
            height: imageHeight,
            featureChannels: channelCount
        )
        
        let inputImage = MPSImage(device: device, imageDescriptor: imageDesc)
        
        // Create dummy data for the input image
        let imageByteSize = imageWidth * imageHeight * channelCount * MemoryLayout<Float16>.size
        let imageBytes = UnsafeMutablePointer<Float16>.allocate(capacity: imageByteSize / MemoryLayout<Float16>.size)
        
        // Fill with random data
        for i in 0..<(imageByteSize / MemoryLayout<Float16>.size) {
            imageBytes[i] = Float16(Float.random(in: 0...1))
        }
        
        // Copy data to the image
        let region = MTLRegion(
            origin: MTLOrigin(x: 0, y: 0, z: 0),
            size: MTLSize(width: imageWidth, height: imageHeight, depth: 1)
        )
        
        inputImage.texture.replace(
            region: region,
            mipmapLevel: 0,
            withBytes: imageBytes,
            bytesPerRow: imageWidth * channelCount * MemoryLayout<Float16>.size
        )
        
        // Free memory
        imageBytes.deallocate()
        
        // Run inference
        print("Running forward pass...")
        let startTime = Date()
        let outputImage = network.forward(input: inputImage)
        let inferenceTime = Date().timeIntervalSince(startTime)
        
        print("Forward pass completed in \(inferenceTime * 1000) ms")
        print("Output image size: \(outputImage.width)x\(outputImage.height)x\(outputImage.featureChannels)")
        
        // Set up benchmarking if desired
        if CommandLine.arguments.contains("--benchmark") {
            print("Running benchmarks...")
            
            // Create benchmarker
            let benchmarker = PerformanceBenchmarker(device: device)
            
            // Benchmark Metal implementation
            let metalMetrics = benchmarker.benchmarkMetal(network: network, input: inputImage, iterations: 100)
            
            print("Metal benchmarking results:")
            print("  - Average execution time: \(metalMetrics.executionTime * 1000) ms")
            print("  - Estimated memory usage: \(Double(metalMetrics.memoryUsage) / 1_000_000) MB")
            print("  - Hardware: \(metalMetrics.hardwareInfo.name)")
            print("  - Neural Engine: \(metalMetrics.hardwareInfo.capabilities["hasNeuralEngine"] as? Bool ?? false)")
            
            // In a real application, you would also benchmark CUDA implementation
            // and compare the results
        }
        
        // Export network structure if desired
        if CommandLine.arguments.contains("--export") {
            print("Exporting network structure...")
            
            let networkConfig = network.export()
            
            if let networkData = try? JSONSerialization.data(withJSONObject: networkConfig, options: .prettyPrinted),
               let networkString = String(data: networkData, encoding: .utf8) {
                print("Network structure:")
                print(networkString)
            }
            
            // Get network history
            let history = network.getHistory()
            print("Network history: \(history.count) snapshots")
            
            for (index, snapshot) in history.enumerated() {
                print("  \(index): \(snapshot.timestamp) - \(snapshot.reason)")
            }
        }
        
        print("Example completed successfully")
    }
}

/// Extension to allow running the example from command line
@main
extension SimpleExample {
    /// Main entry point
    public static func main() {
        runExample()
    }
} 