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
    
    // Weight and bias data
    private var weights: [Float]
    private var biases: [Float]
    
    // Metal resources
    private var device: MTLDevice
    private var weightsBuffer: MTLBuffer?
    private var biasesBuffer: MTLBuffer?
    
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
        
        super.init(name: "Convolution", type: .convolution)
        
        setupMetal()
    }
    
    private func setupMetal() {
        // Create Metal buffers for weights and biases
        let weightsSize = weights.count * MemoryLayout<Float>.size
        let biasesSize = biases.count * MemoryLayout<Float>.size
        
        weightsBuffer = device.makeBuffer(bytes: weights, length: weightsSize, options: .storageModeShared)
        biasesBuffer = device.makeBuffer(bytes: biases, length: biasesSize, options: .storageModeShared)
        
        // Create convolution descriptor
        let convDesc = MPSCNNConvolutionDescriptor(
            kernelWidth: kernelSize,
            kernelHeight: kernelSize,
            inputFeatureChannels: inputChannels,
            outputFeatureChannels: outputChannels
        )
        
        convDesc.strideInPixelsX = stride
        convDesc.strideInPixelsY = stride
        
        // Set padding
        if padding > 0 {
            convDesc.setEdgeMode(.zero)
        }
        
        // Create convolution kernel
        if let weights = weightsBuffer, let biases = biasesBuffer {
            convolution = MPSCNNConvolution(
                device: device,
                convolutionDescriptor: convDesc,
                kernelWeights: weights.contents().bindMemory(to: Float.self, capacity: self.weights.count),
                biasTerms: biases.contents().bindMemory(to: Float.self, capacity: self.biases.count),
                flags: useNeuralEngine ? .enableNeuralNetworkGraph : []
            )
            
            convolution?.padding = MPSNNPaddingMethod.custom(
                MPSNNPaddingCustom(
                    leftPadding: padding,
                    rightPadding: padding,
                    topPadding: padding,
                    bottomPadding: padding
                )
            )
        }
    }
    
    public override func forward(input: MPSImage, commandBuffer: MTLCommandBuffer) -> MPSImage {
        guard let convolution = convolution else {
            fatalError("Convolution not initialized")
        }
        
        // Create output image
        let outputWidth = (input.width - kernelSize + 2 * padding) / stride + 1
        let outputHeight = (input.height - kernelSize + 2 * padding) / stride + 1
        
        let outputImageDescriptor = MPSImageDescriptor(
            channelFormat: input.featureChannelFormat,
            width: outputWidth,
            height: outputHeight,
            featureChannels: outputChannels
        )
        
        let outputImage = MPSImage(device: device, imageDescriptor: outputImageDescriptor)
        
        // Encode convolution
        convolution.encode(commandBuffer: commandBuffer, sourceImage: input, destinationImage: outputImage)
        
        return outputImage
    }
    
    public override func backward(inputGradient: MPSImage, commandBuffer: MTLCommandBuffer) -> MPSImage {
        // For simplified version, return input gradient directly
        return inputGradient
    }
    
    public override func updateParameters(learningRate: Float) {
        // Simplified version: no parameter updates
    }
    
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
        
        // Setup metal again with copied weights
        copy.setupMetal()
        
        return copy
    }
    
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
        
        // Reset Metal setup with new parameters
        setupMetal()
    }
    
    public override func optimizeForM4(useNeuralEngine: Bool) -> Bool {
        // Update Neural Engine usage flag
        self.useNeuralEngine = useNeuralEngine
        
        // Reinitialize the convolution with Neural Engine support
        setupMetal()
        
        return true
    }
} 