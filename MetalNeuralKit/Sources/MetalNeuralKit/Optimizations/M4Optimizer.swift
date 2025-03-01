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
        var optimizationSuccess = true
        for i in 0..<network.getLayers().count {
            let layer = network.getLayer(at: i)
            let layerSuccess = layer.optimizeForM4(useNeuralEngine: isNeuralEngineAvailable)
            optimizationSuccess = optimizationSuccess && layerSuccess
        }
        
        // Record a snapshot after optimization
        ArchitectureTracker.shared.recordSnapshot(
            network: network,
            reason: .performanceOptimization
        )
        
        return optimizationSuccess
    }
    
    /// Create an MPSGraph optimized for M4
    /// - Returns: An optimized MPSGraph instance
    public func createOptimizedGraph() -> MPSGraph {
        let graph = MPSGraph()
        
        // Set optimization options if running on M4
        if isM4Chip {
            // Enable optimizations
            graph.options = MPSGraphOptions.synchronizeResults
            
            // Enable half-precision if requested
            if useHalfPrecision {
                // This would enable FP16 computations when supported
                // Implementation would depend on MPSGraph capabilities
            }
        }
        
        return graph
    }
    
    /// Get M4 optimization capabilities
    /// - Returns: A dictionary of capabilities
    public func getM4Capabilities() -> [String: Any] {
        return [
            "isM4Chip": isM4Chip,
            "hasNeuralEngine": isNeuralEngineAvailable,
            "supportsHalfPrecision": true,
            "deviceName": device.name
        ]
    }
    
    /// Get optimization recommendations for current hardware
    /// - Returns: Array of recommendations
    public func getOptimizationTips() -> [String] {
        var tips: [String] = []
        
        if isM4Chip {
            tips.append("Use half-precision (Float16) operations for better performance")
            
            if isNeuralEngineAvailable {
                tips.append("Enable Neural Engine for compatible layers")
                tips.append("Batch operations for better Neural Engine utilization")
            }
            
            tips.append("Minimize CPU-GPU data transfers")
            tips.append("Use MPSGraph for complex operations")
        } else {
            tips.append("M4-specific optimizations not available on this device")
        }
        
        return tips
    }
}

/// Extension on MTLDevice to check for Neural Engine support
extension MTLDevice {
    /// Check if the Neural Engine is available
    public var hasNeuralEngine: Bool {
        return self.supportsFamily(.apple7)
    }
} 