import Foundation
import Metal
import MetalPerformanceShaders

/// Protocol defining a neural network layer
public protocol Layer: AnyObject {
    /// Unique identifier for the layer
    var id: UUID { get }
    
    /// Name of the layer for tracking and debugging
    var name: String { get }
    
    /// Type of the layer
    var type: LayerType { get }
    
    /// Parameters of the layer that can be trained/updated
    var parameters: [String: Any] { get }
    
    /// Forward pass logic
    func forward(input: MPSImage, commandBuffer: MTLCommandBuffer) -> MPSImage
    
    /// Backward pass logic for training
    func backward(inputGradient: MPSImage, commandBuffer: MTLCommandBuffer) -> MPSImage
    
    /// Update parameters during training
    func updateParameters(learningRate: Float)
    
    /// Create a deep copy of the layer
    func copy() -> Layer
    
    /// Export the layer configuration to a dictionary
    func export() -> [String: Any]
    
    /// Import layer configuration from a dictionary
    func `import`(configuration: [String: Any])
    
    /// Optimize the layer for M4 hardware if applicable
    func optimizeForM4(useNeuralEngine: Bool) -> Bool
}

/// Type of neural network layer
public enum LayerType: String, Codable {
    case convolution
    case pooling
    case fullyConnected
    case activation
    case batchNormalization
    case custom
}

/// Base implementation for Layer protocol
open class BaseLayer: Layer {
    public let id = UUID()
    public var name: String
    public let type: LayerType
    public let createdAt = Date()
    public var modifiedAt: Date
    
    public var parameters: [String: Any] {
        return [:]
    }
    
    public init(name: String, type: LayerType) {
        self.name = name
        self.type = type
        self.modifiedAt = createdAt
    }
    
    open func forward(input: MPSImage, commandBuffer: MTLCommandBuffer) -> MPSImage {
        fatalError("Subclass must implement forward")
    }
    
    open func backward(inputGradient: MPSImage, commandBuffer: MTLCommandBuffer) -> MPSImage {
        fatalError("Subclass must implement backward")
    }
    
    open func updateParameters(learningRate: Float) {
        // Default implementation does nothing
    }
    
    open func copy() -> Layer {
        fatalError("Subclass must implement copy")
    }
    
    open func export() -> [String: Any] {
        return [
            "id": id.uuidString,
            "name": name,
            "type": type.rawValue,
            "createdAt": createdAt.timeIntervalSince1970,
            "modifiedAt": modifiedAt.timeIntervalSince1970
        ]
    }
    
    open func `import`(configuration: [String: Any]) {
        if let name = configuration["name"] as? String {
            self.name = name
        }
        
        if let modifiedAt = configuration["modifiedAt"] as? TimeInterval {
            self.modifiedAt = Date(timeIntervalSince1970: modifiedAt)
        }
    }
    
    open func optimizeForM4(useNeuralEngine: Bool) -> Bool {
        // Default implementation: no M4-specific optimizations
        return false
    }
} 