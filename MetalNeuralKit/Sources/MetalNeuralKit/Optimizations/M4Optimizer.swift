import Foundation
import Metal
import MetalPerformanceShaders

/// Optimizer for Apple M4 architecture neural network operations
public class M4Optimizer {
    /// Singleton instance
    public static let shared = M4Optimizer()
    
    /// Metal device
    private let device: MTLDevice
    
    /// Command queue
    private let commandQueue: MTLCommandQueue
    
    /// Whether the Neural Engine is available
    private let isNeuralEngineAvailable: Bool
    
    /// Whether this is running on an M4 chip
    private let isM4Chip: Bool
    
    /// Whether to use half-precision (Float16) for operations
    private var useHalfPrecision: Bool = true
    
    /// Private initializer
    private init() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        
        self.device = device
        
        guard let commandQueue = device.makeCommandQueue() else {
            fatalError("Failed to create command queue")
        }
        
        self.commandQueue = commandQueue
        
        // Check for Neural Engine
        self.isNeuralEngineAvailable = device.supportsFamily(.apple7)
        
        // Check for M4 chip - this is a simplified check
        // In a real implementation, this would use more sophisticated detection
        let deviceName = device.name.lowercased()
        self.isM4Chip = deviceName.contains("m4") || deviceName.contains("apple 4")
        
        print("M4Optimizer initialized on device: \(device.name)")
        print("Neural Engine available: \(isNeuralEngineAvailable)")
        print("Detected as M4 chip: \(isM4Chip)")
    }
    
    /// Configure optimization settings
    /// - Parameter useHalfPrecision: Whether to use half-precision (Float16)
    public func configure(useHalfPrecision: Bool) {
        self.useHalfPrecision = useHalfPrecision
    }
    
    /// Optimize a neural network for M4 architecture
    /// - Parameter network: The neural network to optimize
    /// - Returns: True if optimization was successful
    public func optimizeNetwork(_ network: NeuralNetwork) -> Bool {
        guard isM4Chip else {
            print("M4 optimizations are only available on M4 chips")
            return false
        }
        
        print("Optimizing network '\(network.name)' for M4 architecture...")
        
        // Record a snapshot before optimization
        ArchitectureTracker.shared.recordSnapshot(
            network: network,
            reason: .performanceOptimization
        )
        
        // Optimize each layer
        for layer in network.layers {
            optimizeLayer(layer)
        }
        
        // Record a snapshot after optimization
        ArchitectureTracker.shared.recordSnapshot(
            network: network,
            reason: .performanceOptimization
        )
        
        return true
    }
    
    /// Optimize a layer for M4 architecture
    /// - Parameter layer: The layer to optimize
    private func optimizeLayer(_ layer: Layer) {
        switch layer.type {
        case .convolution:
            optimizeConvolutionLayer(layer)
        case .fullyConnected:
            optimizeFullyConnectedLayer(layer)
        case .batchNormalization:
            optimizeBatchNormalizationLayer(layer)
        case .pooling:
            optimizePoolingLayer(layer)
        default:
            // Other layer types may not need specific optimizations
            break
        }
    }
    
    /// Optimize a convolution layer for M4 architecture
    /// - Parameter layer: The convolution layer to optimize
    private func optimizeConvolutionLayer(_ layer: Layer) {
        guard let convLayer = layer as? ConvolutionLayer else { return }
        
        // Enable Neural Engine if available
        if isNeuralEngineAvailable {
            convLayer.useNeuralEngine = true
        }
        
        // Set precision based on configuration
        convLayer.useHalfPrecision = useHalfPrecision
        
        // Adjust kernel parameters for M4 architecture
        // In a real implementation, this would involve more sophisticated
        // adjustments based on M4-specific optimizations
        
        // For demonstration purposes, we'll just adjust some parameters
        
        // Optimize convolution groups
        if let inputChannels = convLayer.parameters["inputChannels"] as? Int,
           let outputChannels = convLayer.parameters["outputChannels"] as? Int {
            
            // Check if we can use grouped convolutions
            if inputChannels == outputChannels && inputChannels.isMultiple(of: 4) {
                convLayer.parameters["groups"] = 4
            }
        }
        
        // Add a note about optimization
        var notes = convLayer.parameters["optimizationNotes"] as? [String] ?? []
        notes.append("Optimized for M4 Neural Engine on \(Date())")
        convLayer.parameters["optimizationNotes"] = notes
    }
    
    /// Optimize a fully connected layer for M4 architecture
    /// - Parameter layer: The fully connected layer to optimize
    private func optimizeFullyConnectedLayer(_ layer: Layer) {
        // In a real implementation, this would apply M4-specific optimizations
        
        // Add a note about optimization
        var notes = layer.parameters["optimizationNotes"] as? [String] ?? []
        notes.append("Optimized for M4 on \(Date())")
        layer.parameters["optimizationNotes"] = notes
    }
    
    /// Optimize a batch normalization layer for M4 architecture
    /// - Parameter layer: The batch normalization layer to optimize
    private func optimizeBatchNormalizationLayer(_ layer: Layer) {
        // In a real implementation, this would apply M4-specific optimizations
        
        // Add a note about optimization
        var notes = layer.parameters["optimizationNotes"] as? [String] ?? []
        notes.append("Optimized for M4 on \(Date())")
        layer.parameters["optimizationNotes"] = notes
    }
    
    /// Optimize a pooling layer for M4 architecture
    /// - Parameter layer: The pooling layer to optimize
    private func optimizePoolingLayer(_ layer: Layer) {
        // In a real implementation, this would apply M4-specific optimizations
        
        // Add a note about optimization
        var notes = layer.parameters["optimizationNotes"] as? [String] ?? []
        notes.append("Optimized for M4 on \(Date())")
        layer.parameters["optimizationNotes"] = notes
    }
    
    /// Create an optimized graph for the neural network
    /// - Parameter network: The neural network
    /// - Returns: An optimized MPSGraph
    public func createOptimizedGraph(for network: NeuralNetwork) -> MPSGraph {
        let graph = MPSGraph()
        
        // Configure graph options
        if isM4Chip {
            // M4-specific options
            graph.options = [
                .synchronizeResults: true,
                .debugMode: false
            ]
            
            if useHalfPrecision {
                // Use half precision
                graph.options[.dataTypeMode] = MPSGraphOptions.DataTypeMode.preferFast.rawValue
            }
        }
        
        // In a real implementation, this would build a complete graph
        // based on the network's layers
        
        return graph
    }
    
    /// Get M4-specific optimization tips
    /// - Returns: Array of optimization tips
    public func getM4OptimizationTips() -> [String] {
        return [
            "Use half-precision (Float16) for most operations to leverage enhanced M4 half-precision performance",
            "Enable Neural Engine for convolution and fully connected layers",
            "Use grouped convolutions where possible",
            "Batch operations to minimize CPU-GPU synchronization",
            "Use MPSGraph for complex operations to leverage automatic optimization",
            "Minimize data transfers between CPU and GPU",
            "Consider fusing operations (like convolution + bias + activation) for performance",
            "Allocate persistent buffers to avoid Metal resource creation overhead"
        ]
    }
    
    /// Get M4 capability information
    /// - Returns: Dictionary of capability information
    public func getM4CapabilityInfo() -> [String: Any] {
        return [
            "isM4Chip": isM4Chip,
            "hasNeuralEngine": isNeuralEngineAvailable,
            "deviceName": device.name,
            "registryID": device.registryID,
            "supportsUnifiedMemory": device.hasUnifiedMemory,
            "recommendedMaxWorkingSetSize": device.recommendedMaxWorkingSetSize,
            "supportsMPSGraph": true
        ]
    }
    
    /// Get the optimal thread group size for compute kernels on M4
    /// - Returns: MTLSize with optimal thread group dimensions
    public func getOptimalM4ThreadGroupSize() -> MTLSize {
        // On M4, optimal thread group sizes depend on the specific compute unit
        // but these are generally good starting points
        return MTLSizeMake(32, 4, 1)
    }
    
    /// Create an optimized compute pipeline state for M4
    /// - Parameters:
    ///   - functionName: The Metal function name
    ///   - library: The Metal library
    /// - Returns: An optimized compute pipeline state
    public func createOptimizedComputePipelineState(
        functionName: String,
        library: MTLLibrary
    ) -> MTLComputePipelineState? {
        guard let function = library.makeFunction(name: functionName) else {
            print("Failed to create function \(functionName)")
            return nil
        }
        
        // Create pipeline state with optimization options
        let descriptor = MTLComputePipelineDescriptor()
        descriptor.computeFunction = function
        
        // Enable threadgroup memory length optimization
        descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true
        
        // Enable buffer data types
        descriptor.buffers[0].mutability = .immutable
        
        do {
            // Create optimized pipeline state
            return try device.makeComputePipelineState(
                descriptor: descriptor,
                options: [.argumentInfo, .bufferTypeInfo],
                reflection: nil
            )
        } catch {
            print("Failed to create compute pipeline state: \(error)")
            return nil
        }
    }
}

/// Extension on MTLDevice to check for Neural Engine support
extension MTLDevice {
    /// Check if the Neural Engine is available
    public var hasNeuralEngine: Bool {
        return self.supportsFamily(.apple7)
    }
} 