import Foundation
import Metal
import MetalPerformanceShaders

/// Main neural network class that contains layers and manages execution
public class NeuralNetwork {
    // Core properties
    public let id = UUID()
    public var name: String
    public let createdAt = Date()
    public var modifiedAt: Date
    
    // Metal objects
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    
    // Network structure
    private var layers: [Layer] = []
    
    // History tracking
    private var history: [NetworkSnapshot] = []
    private var trackingEnabled: Bool
    private var snapshotInterval: TimeInterval
    private var lastSnapshotTime: Date?
    
    /// Initialize a new neural network
    public init(name: String, device: MTLDevice? = nil, trackHistory: Bool = true, snapshotInterval: TimeInterval = 3600) {
        self.name = name
        self.modifiedAt = createdAt
        self.trackingEnabled = trackHistory
        self.snapshotInterval = snapshotInterval
        
        // Get the default Metal device if none provided
        if let providedDevice = device {
            self.device = providedDevice
        } else {
            guard let defaultDevice = MTLCreateSystemDefaultDevice() else {
                fatalError("Metal is not supported on this device")
            }
            self.device = defaultDevice
        }
        
        // Create command queue
        guard let queue = self.device.makeCommandQueue() else {
            fatalError("Could not create Metal command queue")
        }
        self.commandQueue = queue
        
        // Create initial snapshot
        if trackingEnabled {
            createSnapshot(reason: "Initial network creation")
        }
    }
    
    /// Add a layer to the network
    @discardableResult
    public func add(layer: Layer) -> Self {
        layers.append(layer)
        modifiedAt = Date()
        
        // Create snapshot if tracking is enabled
        if trackingEnabled {
            createSnapshot(reason: "Added \(layer.name) layer")
        }
        
        return self
    }
    
    /// Remove a layer at the specified index
    @discardableResult
    public func removeLayer(at index: Int) -> Self {
        guard index >= 0 && index < layers.count else {
            fatalError("Layer index out of bounds")
        }
        
        let removedLayer = layers.remove(at: index)
        modifiedAt = Date()
        
        // Create snapshot if tracking is enabled
        if trackingEnabled {
            createSnapshot(reason: "Removed \(removedLayer.name) layer")
        }
        
        return self
    }
    
    /// Get all layers in the network
    public func getLayers() -> [Layer] {
        return layers
    }
    
    /// Get a layer at a specific index
    public func getLayer(at index: Int) -> Layer {
        guard index >= 0 && index < layers.count else {
            fatalError("Layer index out of bounds")
        }
        
        return layers[index]
    }
    
    /// Forward pass through the entire network
    public func forward(input: MPSImage) -> MPSImage {
        guard !layers.isEmpty else {
            return input // No transformation if there are no layers
        }
        
        // Create a command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            fatalError("Could not create command buffer")
        }
        
        var currentOutput = input
        
        // Pass through each layer
        for layer in layers {
            currentOutput = layer.forward(input: currentOutput, commandBuffer: commandBuffer)
        }
        
        // Commit the command buffer
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return currentOutput
    }
    
    /// Backward pass for training (computes gradients)
    public func backward(outputGradient: MPSImage) -> MPSImage {
        guard !layers.isEmpty else {
            return outputGradient // No transformation if there are no layers
        }
        
        // Create a command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            fatalError("Could not create command buffer")
        }
        
        var currentGradient = outputGradient
        
        // Pass through each layer in reverse
        for layer in layers.reversed() {
            currentGradient = layer.backward(inputGradient: currentGradient, commandBuffer: commandBuffer)
        }
        
        // Commit the command buffer
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return currentGradient
    }
    
    /// Update network parameters (weights, biases) during training
    public func updateParameters(learningRate: Float) {
        for layer in layers {
            layer.updateParameters(learningRate: learningRate)
        }
        
        modifiedAt = Date()
        
        // Check if we should create a snapshot based on time interval
        if trackingEnabled, let lastSnapshot = lastSnapshotTime {
            let timeSinceLastSnapshot = modifiedAt.timeIntervalSince(lastSnapshot)
            if timeSinceLastSnapshot >= snapshotInterval {
                createSnapshot(reason: "Periodic update after training")
            }
        }
    }
    
    /// Train the network on a batch of data
    public func train(input: MPSImage, target: MPSImage, learningRate: Float) {
        // Forward pass
        let output = forward(input: input)
        
        // Compute loss gradient - this is simplified and would need a proper loss function
        // In a real implementation, you would compute the gradient of your loss function here
        
        // For simplicity, we'll just use the difference between output and target as the gradient
        // Create a command buffer for computing the loss gradient
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            fatalError("Could not create command buffer")
        }
        
        // TODO: Implement proper loss function and gradient computation
        // For now, we'll just use a placeholder
        let outputGradient = output // This should be the gradient of the loss function
        
        // Backward pass to compute gradients
        _ = backward(outputGradient: outputGradient)
        
        // Update parameters with computed gradients
        updateParameters(learningRate: learningRate)
    }
    
    /// Create a snapshot of the current network state
    private func createSnapshot(reason: String) {
        // Create deep copies of all layers
        let layerCopies = layers.map { $0.copy() }
        
        let snapshot = NetworkSnapshot(
            networkId: id,
            timestamp: Date(),
            reason: reason,
            layers: layerCopies
        )
        
        history.append(snapshot)
        lastSnapshotTime = snapshot.timestamp
    }
    
    /// Get the history of network snapshots
    public func getHistory() -> [NetworkSnapshot] {
        return history
    }
    
    /// Export the network to a dictionary
    public func export() -> [String: Any] {
        let layerConfigs = layers.map { $0.export() }
        
        return [
            "id": id.uuidString,
            "name": name,
            "createdAt": createdAt.timeIntervalSince1970,
            "modifiedAt": modifiedAt.timeIntervalSince1970,
            "layers": layerConfigs,
            "history": history.map { $0.export() }
        ]
    }
    
    /// Import network from a dictionary
    public func `import`(configuration: [String: Any]) {
        if let name = configuration["name"] as? String {
            self.name = name
        }
        
        if let modifiedAt = configuration["modifiedAt"] as? TimeInterval {
            self.modifiedAt = Date(timeIntervalSince1970: modifiedAt)
        }
        
        // Clear existing layers
        layers.removeAll()
        
        // Import layers
        if let layerConfigs = configuration["layers"] as? [[String: Any]] {
            for layerConfig in layerConfigs {
                // Create appropriate layer type based on configuration
                if let layerType = layerConfig["type"] as? String,
                   let type = LayerType(rawValue: layerType) {
                    
                    var layer: Layer?
                    
                    switch type {
                    case .convolution:
                        if let inputChannels = layerConfig["inputChannels"] as? Int,
                           let outputChannels = layerConfig["outputChannels"] as? Int,
                           let kernelSize = layerConfig["kernelSize"] as? Int {
                            
                            layer = ConvolutionLayer(
                                device: device,
                                inputChannels: inputChannels,
                                outputChannels: outputChannels,
                                kernelSize: kernelSize
                            )
                        }
                        
                    // Handle other layer types here
                    default:
                        break
                    }
                    
                    // Import the layer configuration
                    if let layer = layer {
                        layer.import(configuration: layerConfig)
                        layers.append(layer)
                    }
                }
            }
        }
        
        // Import history if tracking is enabled
        if trackingEnabled, let historyConfigs = configuration["history"] as? [[String: Any]] {
            history.removeAll()
            
            for historyConfig in historyConfigs {
                if let snapshot = NetworkSnapshot.import(configuration: historyConfig, device: device) {
                    history.append(snapshot)
                }
            }
            
            lastSnapshotTime = history.last?.timestamp
        }
    }
}

/// Snapshot of a neural network at a specific point in time
public struct NetworkSnapshot {
    public let networkId: UUID
    public let timestamp: Date
    public let reason: String
    public let layers: [Layer]
    
    /// Export the snapshot to a dictionary
    public func export() -> [String: Any] {
        return [
            "networkId": networkId.uuidString,
            "timestamp": timestamp.timeIntervalSince1970,
            "reason": reason,
            "layers": layers.map { $0.export() }
        ]
    }
    
    /// Import snapshot from a dictionary
    public static func `import`(configuration: [String: Any], device: MTLDevice) -> NetworkSnapshot? {
        guard let networkIdString = configuration["networkId"] as? String,
              let networkId = UUID(uuidString: networkIdString),
              let timestamp = configuration["timestamp"] as? TimeInterval,
              let reason = configuration["reason"] as? String,
              let layerConfigs = configuration["layers"] as? [[String: Any]] else {
            return nil
        }
        
        var layers: [Layer] = []
        
        // Import layers
        for layerConfig in layerConfigs {
            if let layerType = layerConfig["type"] as? String,
               let type = LayerType(rawValue: layerType) {
                
                var layer: Layer?
                
                switch type {
                case .convolution:
                    if let inputChannels = layerConfig["inputChannels"] as? Int,
                       let outputChannels = layerConfig["outputChannels"] as? Int,
                       let kernelSize = layerConfig["kernelSize"] as? Int {
                        
                        layer = ConvolutionLayer(
                            device: device,
                            inputChannels: inputChannels,
                            outputChannels: outputChannels,
                            kernelSize: kernelSize
                        )
                    }
                    
                // Handle other layer types here
                default:
                    break
                }
                
                // Import the layer configuration
                if let layer = layer {
                    layer.import(configuration: layerConfig)
                    layers.append(layer)
                }
            }
        }
        
        return NetworkSnapshot(
            networkId: networkId,
            timestamp: Date(timeIntervalSince1970: timestamp),
            reason: reason,
            layers: layers
        )
    }
} 