import Foundation
import Metal
import MetalPerformanceShaders

/// Convolution layer using Metal Performance Shaders
public class ConvolutionLayer: BaseLayer {
    // Layer configuration
    public let inputChannels: Int
    public let outputChannels: Int
    public let kernelSize: Int
    public let stride: Int
    public let padding: Int
    
    // MPS objects
    private var convolution: MPSCNNConvolution?
    private var convolutionGradient: MPSCNNConvolutionGradient?
    private var convolutionTranspose: MPSCNNConvolutionTranspose?
    
    // Weight and bias data
    private var weights: [Float]
    private var biases: [Float]
    private var weightGradients: [Float]
    private var biasGradients: [Float]
    
    // Metal resources
    private var device: MTLDevice
    private var weightsBuffer: MTLBuffer?
    private var biasesBuffer: MTLBuffer?
    private var weightGradientBuffer: MTLBuffer?
    private var biasGradientBuffer: MTLBuffer?
    
    // Neural Engine optimization flags
    private var useNeuralEngine: Bool
    
    public override var parameters: [String: Any] {
        return [
            "weights": weights,
            "biases": biases,
            "inputChannels": inputChannels,
            "outputChannels": outputChannels,
            "kernelSize": kernelSize,
            "stride": stride,
            "padding": padding
        ]
    }
    
    public init(device: MTLDevice,
                inputChannels: Int,
                outputChannels: Int,
                kernelSize: Int,
                stride: Int = 1,
                padding: Int = 0,
                useNeuralEngine: Bool = true) {
        
        self.device = device
        self.inputChannels = inputChannels
        self.outputChannels = outputChannels
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
        self.useNeuralEngine = useNeuralEngine
        
        // Initialize weights with Xavier initialization
        let weightsCount = outputChannels * inputChannels * kernelSize * kernelSize
        let scale = sqrt(2.0 / Float(inputChannels * kernelSize * kernelSize))
        
        self.weights = [Float](repeating: 0, count: weightsCount)
        for i in 0..<weightsCount {
            // Random value between -scale and scale
            self.weights[i] = (Float.random(in: 0..<1) * 2 - 1) * scale
        }
        
        self.biases = [Float](repeating: 0, count: outputChannels)
        self.weightGradients = [Float](repeating: 0, count: weightsCount)
        self.biasGradients = [Float](repeating: 0, count: outputChannels)
        
        super.init(name: "Convolution", type: .convolution)
        
        setupMetal()
    }
    
    private func setupMetal() {
        // Create Metal buffers for weights and biases
        let weightsSize = weights.count * MemoryLayout<Float>.size
        let biasesSize = biases.count * MemoryLayout<Float>.size
        
        weightsBuffer = device.makeBuffer(bytes: weights, length: weightsSize, options: .storageModeShared)
        biasesBuffer = device.makeBuffer(bytes: biases, length: biasesSize, options: .storageModeShared)
        weightGradientBuffer = device.makeBuffer(length: weightsSize, options: .storageModeShared)
        biasGradientBuffer = device.makeBuffer(length: biasesSize, options: .storageModeShared)
        
        // Create convolution descriptor
        let convDesc = MPSCNNConvolutionDescriptor(
            kernelWidth: kernelSize,
            kernelHeight: kernelSize,
            inputFeatureChannels: inputChannels,
            outputFeatureChannels: outputChannels
        )
        
        convDesc.strideInPixelsX = stride
        convDesc.strideInPixelsY = stride
        
        // Use Neural Engine if requested and available
        if #available(macOS 14.0, *), useNeuralEngine {
            // M4 specific optimizations
            if let accelerationInfo = device.accelerationInfo,
               accelerationInfo.supportsAppleNeuralEngine {
                convDesc.options = [.enableNeuralEngine]
            }
        }
        
        // Create the convolution kernel
        convolution = MPSCNNConvolution(
            device: device,
            convolutionDescriptor: convDesc,
            kernelWeights: weights,
            biasTerms: biases,
            flags: []
        )
        
        if let conv = convolution {
            conv.offset = MPSOffset(x: padding, y: padding, z: 0)
            conv.edgeMode = .zero
        }
        
        // Setup gradient kernels for training
        convolutionGradient = MPSCNNConvolutionGradient(
            device: device,
            convolutionDescriptor: convDesc,
            kernelWeights: weights,
            biasTerms: biases,
            flags: []
        )
        
        convolutionTranspose = MPSCNNConvolutionTranspose(
            device: device,
            convolutionDescriptor: convDesc,
            kernelWeights: weights,
            biasTerms: biases,
            flags: []
        )
    }
    
    // Forward pass implementation
    public override func forward(input: MPSImage, commandBuffer: MTLCommandBuffer) -> MPSImage {
        guard let convolution = convolution else {
            fatalError("Convolution kernel not initialized")
        }
        
        // Create output image descriptor
        let inputWidth = input.width
        let inputHeight = input.height
        
        let outputWidth = (inputWidth - kernelSize + 2 * padding) / stride + 1
        let outputHeight = (inputHeight - kernelSize + 2 * padding) / stride + 1
        
        let outputDesc = MPSImageDescriptor(
            channelFormat: input.featureChannelFormat,
            width: outputWidth,
            height: outputHeight,
            featureChannels: outputChannels
        )
        
        let outputImage = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: outputDesc)
        
        // Encode convolution
        convolution.encode(commandBuffer: commandBuffer, sourceImage: input, destinationImage: outputImage)
        
        return outputImage
    }
    
    // Backward pass implementation for training
    public override func backward(inputGradient: MPSImage, commandBuffer: MTLCommandBuffer) -> MPSImage {
        guard let convolutionGradient = convolutionGradient,
              let convolutionTranspose = convolutionTranspose else {
            fatalError("Gradient kernels not initialized")
        }
        
        // Create output image descriptor for gradients
        let outputDesc = MPSImageDescriptor(
            channelFormat: inputGradient.featureChannelFormat,
            width: inputGradient.width * stride,
            height: inputGradient.height * stride,
            featureChannels: inputChannels
        )
        
        let outputGradient = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: outputDesc)
        
        // Encode gradient computation
        convolutionTranspose.encode(commandBuffer: commandBuffer, sourceImage: inputGradient, destinationImage: outputGradient)
        
        // Update weight gradients
        if let weightGradBuffer = weightGradientBuffer, 
           let biasGradBuffer = biasGradientBuffer {
            
            // TODO: Implement weight gradient computation using Metal compute shader
            // This would encode the computation to update weightGradients and biasGradients
        }
        
        return outputGradient
    }
    
    // Update weights and biases based on computed gradients
    public override func updateParameters(learningRate: Float) {
        // Apply gradients to weights and biases
        for i in 0..<weights.count {
            weights[i] -= learningRate * weightGradients[i]
        }
        
        for i in 0..<biases.count {
            biases[i] -= learningRate * biasGradients[i]
        }
        
        // Reset gradients
        weightGradients = [Float](repeating: 0, count: weights.count)
        biasGradients = [Float](repeating: 0, count: biases.count)
        
        // Update MPS kernels with new weights
        setupMetal()
        
        // Update modification timestamp
        modifiedAt = Date()
    }
    
    // Create a deep copy of the layer
    public override func copy() -> Layer {
        let copy = ConvolutionLayer(
            device: device,
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: padding,
            useNeuralEngine: useNeuralEngine
        )
        
        // Copy weights and biases
        copy.weights = self.weights
        copy.biases = self.biases
        
        // Copy timestamps
        copy.modifiedAt = self.modifiedAt
        
        // Reinitialize Metal objects with copied weights
        copy.setupMetal()
        
        return copy
    }
    
    // Export layer configuration
    public override func export() -> [String: Any] {
        var config = super.export()
        config["inputChannels"] = inputChannels
        config["outputChannels"] = outputChannels
        config["kernelSize"] = kernelSize
        config["stride"] = stride
        config["padding"] = padding
        config["useNeuralEngine"] = useNeuralEngine
        config["weights"] = weights
        config["biases"] = biases
        
        return config
    }
    
    // Import layer configuration
    public override func `import`(configuration: [String: Any]) {
        super.import(configuration: configuration)
        
        if let weights = configuration["weights"] as? [Float] {
            self.weights = weights
        }
        
        if let biases = configuration["biases"] as? [Float] {
            self.biases = biases
        }
        
        if let useNeuralEngine = configuration["useNeuralEngine"] as? Bool {
            self.useNeuralEngine = useNeuralEngine
        }
        
        // Reinitialize Metal objects with imported weights
        setupMetal()
    }
} 