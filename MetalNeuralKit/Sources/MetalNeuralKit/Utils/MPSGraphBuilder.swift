import Foundation
import Metal
import MetalPerformanceShaders

/// Utility class for creating MPSGraph instances optimized for M4 Neural Engine
public class MPSGraphBuilder {
    /// Metal device
    private let device: MTLDevice
    
    /// Whether to use the Neural Engine if available
    private let useNeuralEngine: Bool
    
    /// Initialize the graph builder
    public init(device: MTLDevice? = nil, useNeuralEngine: Bool = true) {
        // Get the default Metal device if none provided
        if let providedDevice = device {
            self.device = providedDevice
        } else {
            guard let defaultDevice = MTLCreateSystemDefaultDevice() else {
                fatalError("Metal is not supported on this device")
            }
            self.device = defaultDevice
        }
        
        self.useNeuralEngine = useNeuralEngine
    }
    
    /// Create an optimized MPSGraph instance
    public func createGraph() -> MPSGraph {
        let graph = MPSGraph()
        
        // Set optimization options
        if #available(macOS 14.0, *), useNeuralEngine {
            // Configure the graph to use the Neural Engine if available
            if let accelerationInfo = device.accelerationInfo,
               accelerationInfo.supportsAppleNeuralEngine {
                
                // Enable Neural Engine optimization
                let options: MPSGraphOptions = [.enableAppleNeuralEngine]
                graph.options = options
            }
        }
        
        return graph
    }
    
    /// Create an MPSGraphExecutable from the provided graph, optimized for the current device
    public func createExecutable(graph: MPSGraph, feeds: [MPSGraphTensor: MPSGraphShapedType], targetTensors: [MPSGraphTensor]) -> MPSGraphExecutable {
        let executionDescriptor = MPSGraphExecutionDescriptor()
        
        // Set optimization options for the execution
        if #available(macOS 14.0, *), useNeuralEngine {
            // Configure to use the Neural Engine if available
            if let accelerationInfo = device.accelerationInfo,
               accelerationInfo.supportsAppleNeuralEngine {
                
                // Enable Neural Engine optimization for execution
                executionDescriptor.options = [.enableAppleNeuralEngine]
                
                // M4-specific optimizations
                if device.name.contains("Apple M4") {
                    // M4 has 38 trillion ops/sec Neural Engine
                    // Set appropriate optimization parameters
                    
                    // Enable 16-bit floating point operations where possible
                    // This is particularly effective for the Neural Engine
                    executionDescriptor.options.insert(.enableFP16Execution)
                    
                    // Add M4-specific optimizations here as they become available
                }
            }
        }
        
        // Create the executable with optimization
        return graph.compileAsynchronous(feeds: feeds, 
                                         targetTensors: targetTensors,
                                         targetOperations: nil,
                                         executionDescriptor: executionDescriptor)
    }
    
    /// Create a convolution operation in the graph
    public func createConvolution(graph: MPSGraph,
                                 input: MPSGraphTensor,
                                 weights: MPSGraphTensor,
                                 biases: MPSGraphTensor?,
                                 stride: [Int] = [1, 1],
                                 padding: MPSGraphPaddingMode = .valid,
                                 dataFormat: MPSGraphTensorNamedDataLayout = .NHWC) -> MPSGraphTensor {
        
        var result: MPSGraphTensor
        
        // Create convolution operation
        result = graph.convolution2D(input: input,
                                     weights: weights,
                                     descriptor: MPSGraphConvolution2DOpDescriptor(
                                        strideInX: stride[0],
                                        strideInY: stride[1],
                                        dilationRateInX: 1,
                                        dilationRateInY: 1,
                                        groups: 1,
                                        paddingMode: padding,
                                        dataLayout: dataFormat,
                                        weightsLayout: .HWIO))
        
        // Add biases if provided
        if let biases = biases {
            result = graph.addition(result, biases, name: nil)
        }
        
        return result
    }
    
    /// Create a fully connected operation in the graph
    public func createFullyConnected(graph: MPSGraph,
                                    input: MPSGraphTensor,
                                    weights: MPSGraphTensor,
                                    biases: MPSGraphTensor?) -> MPSGraphTensor {
        
        // Reshape input if needed
        var reshapedInput = input
        let inputShape = graph.shape(of: input)
        
        // If input has more than 2 dimensions, flatten it to 2D
        if graph.rankOfTensor(input) > 2 {
            let shape = graph.shape(of: input)
            let batchSize = graph.sliceTensor(shape, dimension: 0, start: 0, length: 1, name: nil)
            let remainingDims = graph.sliceTensor(shape, dimension: 1, start: 0, length: graph.rankOfTensor(input) - 1, name: nil)
            let flattenedSize = graph.reduceProduct(remainingDims, axis: 0, name: nil)
            let newShape = graph.concatTensors([batchSize, flattenedSize], dimension: 0, name: nil)
            
            reshapedInput = graph.reshape(input, withShape: newShape, name: nil)
        }
        
        // Matrix multiplication
        var result = graph.matrixMultiplication(primary: reshapedInput, secondary: weights, name: nil)
        
        // Add biases if provided
        if let biases = biases {
            result = graph.addition(result, biases, name: nil)
        }
        
        return result
    }
    
    /// Create an activation function in the graph
    public func createActivation(graph: MPSGraph,
                                input: MPSGraphTensor,
                                type: ActivationType) -> MPSGraphTensor {
        
        switch type {
        case .relu:
            return graph.reLU(with: input, name: nil)
        case .leakyRelu(let alpha):
            return graph.leakyReLU(with: input, alpha: alpha, name: nil)
        case .sigmoid:
            return graph.sigmoid(with: input, name: nil)
        case .tanh:
            return graph.tanh(with: input, name: nil)
        case .softmax:
            return graph.softMax(with: input, axis: -1, name: nil)
        }
    }
    
    /// Create a pooling operation in the graph
    public func createPooling(graph: MPSGraph,
                             input: MPSGraphTensor,
                             poolingType: PoolingType,
                             kernelSize: [Int],
                             stride: [Int],
                             padding: MPSGraphPaddingMode = .valid,
                             dataFormat: MPSGraphTensorNamedDataLayout = .NHWC) -> MPSGraphTensor {
        
        let descriptor = MPSGraphPooling2DOpDescriptor(
            kernelWidth: kernelSize[0],
            kernelHeight: kernelSize[1],
            strideInX: stride[0],
            strideInY: stride[1],
            paddingMode: padding,
            dataLayout: dataFormat
        )
        
        switch poolingType {
        case .max:
            return graph.maxPooling2D(withSourceTensor: input, descriptor: descriptor, name: nil)
        case .average:
            return graph.avgPooling2D(withSourceTensor: input, descriptor: descriptor, name: nil)
        }
    }
    
    /// Create a batch normalization operation in the graph
    public func createBatchNorm(graph: MPSGraph,
                               input: MPSGraphTensor,
                               mean: MPSGraphTensor,
                               variance: MPSGraphTensor,
                               gamma: MPSGraphTensor,
                               beta: MPSGraphTensor,
                               epsilon: Float = 1e-5) -> MPSGraphTensor {
        
        return graph.normalization(
            input,
            mean: mean,
            variance: variance,
            gamma: gamma,
            beta: beta,
            epsilon: MPSGraphTensor(scalar: Float(epsilon), graph: graph),
            name: nil
        )
    }
}

/// Type of activation function
public enum ActivationType {
    case relu
    case leakyRelu(alpha: Float)
    case sigmoid
    case tanh
    case softmax
}

/// Type of pooling operation
public enum PoolingType {
    case max
    case average
} 