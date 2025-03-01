import Foundation
import MetalNeuralKit

/// Print a welcome banner
func printWelcomeBanner() {
    print("""
    ------------------------------------------------
    |  MetalNeuralKit - Neural Network Library     |
    |  Optimized for Apple Silicon M4 and Neural   |
    |  Engine with Architecture Tracking           |
    ------------------------------------------------
    
    A Swift library for building, training, and optimizing
    neural networks on Apple Silicon devices
    
    """)
}

/// Print the menu
func printMenu() {
    print("""
    Please select an option:
    
    1. Run Simple Example
    2. Run Comprehensive Example
    3. Show Device Capabilities
    4. Run M4 Optimization Test
    5. Run CUDA Comparison (simulated)
    
    0. Exit
    
    Enter your choice: 
    """, terminator: "")
}

/// Show device capabilities
func showDeviceCapabilities() {
    print("\n📊 Device Capabilities:")
    print("-----------------------------------------------")
    
    // Get M4 optimizer
    let m4Optimizer = M4Optimizer.shared
    
    // Get capability info
    let capabilityInfo = m4Optimizer.getM4CapabilityInfo()
    
    // Print device info
    print("Device Name: \(capabilityInfo["deviceName"] as? String ?? "Unknown")")
    print("Is M4 Chip: \(capabilityInfo["isM4Chip"] as? Bool ?? false)")
    print("Has Neural Engine: \(capabilityInfo["hasNeuralEngine"] as? Bool ?? false)")
    print("Supports Unified Memory: \(capabilityInfo["supportsUnifiedMemory"] as? Bool ?? false)")
    let memorySize = Int64(capabilityInfo["recommendedMaxWorkingSetSize"] as? Int ?? 0)
    print("Recommended Working Set Size: \(ByteCountFormatter.string(fromByteCount: memorySize, countStyle: .file))")
    
    // Print optimization tips
    print("\nOptimization Tips for M4:")
    let tips = m4Optimizer.getM4OptimizationTips()
    for (index, tip) in tips.enumerated() {
        print("  \(index + 1). \(tip)")
    }
    
    print("\nPress Enter to continue...")
    _ = readLine()
}

/// Run M4 optimization test
func runM4OptimizationTest() {
    print("\n⚡ M4 Optimization Test:")
    print("-----------------------------------------------")
    
    // Get optimizer
    let m4Optimizer = M4Optimizer.shared
    
    // Check if we're on M4
    let isM4 = m4Optimizer.getM4CapabilityInfo()["isM4Chip"] as? Bool ?? false
    
    if !isM4 {
        print("Warning: Not running on M4 chip. Some optimizations will not be available.")
    }
    
    // Create a small test network
    guard let device = MTLCreateSystemDefaultDevice() else {
        print("Error: Metal device not available")
        return
    }
    
    let network = NeuralNetwork(name: "OptimizationTestNetwork", device: device)
    
    // Add some layers
    print("Creating test network...")
    
    // Add convolution layer
    let conv = ConvolutionLayer(
        name: "conv1",
        inputChannels: 3,
        outputChannels: 16,
        kernelSize: (3, 3),
        stride: (1, 1),
        padding: (1, 1),
        useNeuralEngine: false, // Initially false
        device: device
    )
    network.addLayer(conv)
    
    // Run optimization
    print("Running M4 optimization...")
    let optimized = m4Optimizer.optimizeNetwork(network)
    
    if optimized {
        print("Network successfully optimized for M4!")
        
        // Check if Neural Engine is enabled after optimization
        if let convLayer = network.getLayer(at: 0) as? ConvolutionLayer {
            print("Neural Engine is \(convLayer.useNeuralEngine ? "enabled" : "disabled") after optimization")
            
            // Print other optimization parameters
            let halfPrecision = convLayer.useHalfPrecision
            print("Half precision is \(halfPrecision ? "enabled" : "disabled")")
            
            if let groups = convLayer.parameters["groups"] as? Int {
                print("Convolution groups: \(groups)")
            }
            
            if let notes = convLayer.parameters["optimizationNotes"] as? [String] {
                print("\nOptimization notes:")
                for note in notes {
                    print("- \(note)")
                }
            }
        }
    } else {
        print("Optimization failed (likely not running on M4 chip)")
    }
    
    print("\nPress Enter to continue...")
    _ = readLine()
}

/// Run CUDA comparison test
func runCUDAComparisonTest() {
    print("\n🔍 CUDA Comparison Test (Simulated):")
    print("-----------------------------------------------")
    
    // Get CUDA bridge
    let cudaBridge = CUDABridge.shared
    
    // Check CUDA availability (will be simulated)
    print("CUDA available: \(cudaBridge.isCUDAAvailable() ? "Yes" : "No (simulated)")")
    
    // List available CUDA devices
    let devices = cudaBridge.getAvailableCUDADevices()
    print("\nAvailable CUDA devices (simulated):")
    for device in devices {
        print("- \(device.name) (\(ByteCountFormatter.string(fromByteCount: device.memorySize, countStyle: .file)) memory)")
    }
    
    // Create a dummy comparison
    print("\nGenerating comparison with Metal...")
    let comparison = cudaBridge.createDummyComparisonResult(networkName: "TestNetwork")
    
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
    let outputDirectory = "./output"
    let fileManager = FileManager.default
    if !fileManager.fileExists(atPath: outputDirectory) {
        do {
            try fileManager.createDirectory(atPath: outputDirectory, withIntermediateDirectories: true)
        } catch {
            print("Error creating output directory: \(error)")
        }
    }
    
    let comparisonPath = "\(outputDirectory)/cuda_comparison_test.json"
    if cudaBridge.exportComparisonReport(result: comparison, filePath: comparisonPath) {
        print("\nExported comparison report to: \(comparisonPath)")
    }
    
    print("\nPress Enter to continue...")
    _ = readLine()
}

/// Main function
func main() {
    printWelcomeBanner()
    
    var running = true
    
    while running {
        printMenu()
        
        guard let input = readLine() else {
            print("Error reading input. Please try again.")
            continue
        }
        
        guard let choice = Int(input) else {
            print("Invalid input. Please enter a number.")
            continue
        }
        
        switch choice {
        case 0:
            running = false
            print("Exiting. Thank you for using MetalNeuralKit!")
            
        case 1:
            print("\nRunning Simple Example...\n")
            
            // Run the simple example
            let example = SimpleExample()
            example.runExample()
            
            print("\nSimple Example completed. Press Enter to continue...")
            _ = readLine()
            
        case 2:
            print("\nRunning Comprehensive Example...\n")
            
            // Run the comprehensive example
            runComprehensiveExample()
            
            print("\nComprehensive Example completed. Press Enter to continue...")
            _ = readLine()
            
        case 3:
            showDeviceCapabilities()
            
        case 4:
            runM4OptimizationTest()
            
        case 5:
            runCUDAComparisonTest()
            
        default:
            print("Invalid choice. Please enter a number between 0 and 5.")
        }
    }
}

// Run the main function
main() 