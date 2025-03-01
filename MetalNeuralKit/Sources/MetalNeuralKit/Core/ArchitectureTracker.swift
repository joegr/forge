import Foundation

/// A class for tracking changes to neural network architecture over time
public class ArchitectureTracker {
    /// Singleton instance
    public static let shared = ArchitectureTracker()
    
    /// History of network snapshots
    private var networkSnapshots: [UUID: [NetworkSnapshot]] = [:]
    
    /// Network evolution data
    private var networkEvolution: [UUID: NetworkEvolution] = [:]
    
    /// Private initializer
    private init() {}
    
    /// Record a snapshot of a neural network
    /// - Parameters:
    ///   - network: The neural network to snapshot
    ///   - reason: The reason for taking the snapshot
    public func recordSnapshot(network: NeuralNetwork, reason: SnapshotReason) {
        let snapshot = NetworkSnapshot(
            id: UUID(),
            networkId: network.id,
            timestamp: Date(),
            name: network.name,
            layerCount: network.layers.count,
            reason: reason,
            architecture: serializeArchitecture(network: network)
        )
        
        if networkSnapshots[network.id] == nil {
            networkSnapshots[network.id] = []
        }
        
        networkSnapshots[network.id]?.append(snapshot)
        
        // Update evolution data
        updateEvolutionData(network: network, snapshot: snapshot)
    }
    
    /// Get all snapshots for a network
    /// - Parameter networkId: The network ID
    /// - Returns: An array of snapshots ordered by timestamp
    public func getSnapshots(networkId: UUID) -> [NetworkSnapshot] {
        return networkSnapshots[networkId]?.sorted(by: { $0.timestamp < $1.timestamp }) ?? []
    }
    
    /// Get snapshots for a network within a time range
    /// - Parameters:
    ///   - networkId: The network ID
    ///   - startDate: The start date
    ///   - endDate: The end date
    /// - Returns: An array of snapshots within the time range
    public func getSnapshots(networkId: UUID, startDate: Date, endDate: Date) -> [NetworkSnapshot] {
        return networkSnapshots[networkId]?.filter({ 
            $0.timestamp >= startDate && $0.timestamp <= endDate
        }).sorted(by: { $0.timestamp < $1.timestamp }) ?? []
    }
    
    /// Get the evolution data for a network
    /// - Parameter networkId: The network ID
    /// - Returns: The network evolution data
    public func getEvolutionData(networkId: UUID) -> NetworkEvolution? {
        return networkEvolution[networkId]
    }
    
    /// Export snapshots to JSON
    /// - Parameters:
    ///   - networkId: The network ID
    ///   - filePath: The file path to save to
    /// - Returns: True if successful
    public func exportSnapshots(networkId: UUID, filePath: String) -> Bool {
        guard let snapshots = networkSnapshots[networkId] else {
            return false
        }
        
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = .prettyPrinted
            encoder.dateEncodingStrategy = .iso8601
            
            let data = try encoder.encode(snapshots)
            try data.write(to: URL(fileURLWithPath: filePath))
            
            return true
        } catch {
            print("Error exporting snapshots: \(error)")
            return false
        }
    }
    
    /// Export evolution data to JSON
    /// - Parameters:
    ///   - networkId: The network ID
    ///   - filePath: The file path to save to
    /// - Returns: True if successful
    public func exportEvolutionData(networkId: UUID, filePath: String) -> Bool {
        guard let evolution = networkEvolution[networkId] else {
            return false
        }
        
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = .prettyPrinted
            encoder.dateEncodingStrategy = .iso8601
            
            let data = try encoder.encode(evolution)
            try data.write(to: URL(fileURLWithPath: filePath))
            
            return true
        } catch {
            print("Error exporting evolution data: \(error)")
            return false
        }
    }
    
    /// Import snapshots from JSON
    /// - Parameters:
    ///   - filePath: The file path to load from
    /// - Returns: True if successful
    public func importSnapshots(filePath: String) -> Bool {
        do {
            let data = try Data(contentsOf: URL(fileURLWithPath: filePath))
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            
            let snapshots = try decoder.decode([NetworkSnapshot].self, from: data)
            
            guard let networkId = snapshots.first?.networkId else {
                return false
            }
            
            networkSnapshots[networkId] = snapshots
            
            // Rebuild evolution data
            for snapshot in snapshots {
                if let network = deserializeNetwork(snapshot: snapshot) {
                    updateEvolutionData(network: network, snapshot: snapshot)
                }
            }
            
            return true
        } catch {
            print("Error importing snapshots: \(error)")
            return false
        }
    }
    
    /// Serialize network architecture to dictionary
    private func serializeArchitecture(network: NeuralNetwork) -> [String: Any] {
        var architecture: [String: Any] = [
            "id": network.id.uuidString,
            "name": network.name,
            "createdAt": network.createdAt.timeIntervalSince1970,
            "modifiedAt": network.modifiedAt.timeIntervalSince1970,
            "layers": []
        ]
        
        var layers: [[String: Any]] = []
        
        for layer in network.layers {
            var layerDict: [String: Any] = [
                "id": layer.id.uuidString,
                "name": layer.name,
                "type": layer.type.rawValue,
                "createdAt": layer.createdAt.timeIntervalSince1970,
                "modifiedAt": layer.modifiedAt.timeIntervalSince1970
            ]
            
            // Add layer-specific parameters
            var parameters: [String: Any] = [:]
            for (key, value) in layer.parameters {
                // Convert parameter values to string or numeric representation
                if let value = value as? CustomStringConvertible {
                    parameters[key] = value.description
                } else {
                    parameters[key] = String(describing: value)
                }
            }
            layerDict["parameters"] = parameters
            
            layers.append(layerDict)
        }
        
        architecture["layers"] = layers
        
        return architecture
    }
    
    /// Deserialize network from snapshot (simplified)
    private func deserializeNetwork(snapshot: NetworkSnapshot) -> NeuralNetwork? {
        // This is a simplified version that would need to be expanded
        // to fully reconstruct a network from snapshot data
        guard let device = MTLCreateSystemDefaultDevice() else {
            return nil
        }
        
        let network = NeuralNetwork(id: snapshot.networkId, name: snapshot.name, device: device)
        
        // In a real implementation, we would reconstruct each layer
        // based on the architecture data
        
        return network
    }
    
    /// Update evolution data based on a new snapshot
    private func updateEvolutionData(network: NeuralNetwork, snapshot: NetworkSnapshot) {
        if networkEvolution[network.id] == nil {
            networkEvolution[network.id] = NetworkEvolution(
                networkId: network.id,
                networkName: network.name,
                createdAt: network.createdAt,
                lastModifiedAt: network.modifiedAt,
                layerChanges: [],
                performanceChanges: []
            )
        }
        
        var evolution = networkEvolution[network.id]!
        
        // Update last modified date
        evolution.lastModifiedAt = snapshot.timestamp
        
        // Record layer changes
        let layerChange = LayerChange(
            timestamp: snapshot.timestamp,
            layerCount: snapshot.layerCount,
            reason: snapshot.reason
        )
        evolution.layerChanges.append(layerChange)
        
        // In a real implementation, we would also track performance changes
        
        networkEvolution[network.id] = evolution
    }
}

/// Reason for taking a snapshot
public enum SnapshotReason: String, Codable {
    case initial = "Initial Creation"
    case layerAdded = "Layer Added"
    case layerRemoved = "Layer Removed"
    case layerModified = "Layer Modified"
    case hyperparameterChange = "Hyperparameter Change"
    case trainingCheckpoint = "Training Checkpoint"
    case performanceOptimization = "Performance Optimization"
    case manual = "Manual Snapshot"
    case other = "Other"
}

/// A snapshot of a neural network
public struct NetworkSnapshot: Codable {
    /// Snapshot ID
    public let id: UUID
    
    /// Network ID
    public let networkId: UUID
    
    /// Timestamp when the snapshot was taken
    public let timestamp: Date
    
    /// Network name
    public let name: String
    
    /// Number of layers
    public let layerCount: Int
    
    /// Reason for taking the snapshot
    public let reason: SnapshotReason
    
    /// Architecture as JSON dictionary
    public let architecture: [String: Any]
    
    enum CodingKeys: String, CodingKey {
        case id, networkId, timestamp, name, layerCount, reason, architecture
    }
    
    public init(id: UUID, networkId: UUID, timestamp: Date, name: String, layerCount: Int, reason: SnapshotReason, architecture: [String: Any]) {
        self.id = id
        self.networkId = networkId
        self.timestamp = timestamp
        self.name = name
        self.layerCount = layerCount
        self.reason = reason
        self.architecture = architecture
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        networkId = try container.decode(UUID.self, forKey: .networkId)
        timestamp = try container.decode(Date.self, forKey: .timestamp)
        name = try container.decode(String.self, forKey: .name)
        layerCount = try container.decode(Int.self, forKey: .layerCount)
        reason = try container.decode(SnapshotReason.self, forKey: .reason)
        
        // Decode architecture from JSON string
        let architectureString = try container.decode(String.self, forKey: .architecture)
        if let data = architectureString.data(using: .utf8),
           let jsonObject = try? JSONSerialization.jsonObject(with: data, options: []),
           let architectureDict = jsonObject as? [String: Any] {
            architecture = architectureDict
        } else {
            architecture = [:]
        }
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(networkId, forKey: .networkId)
        try container.encode(timestamp, forKey: .timestamp)
        try container.encode(name, forKey: .name)
        try container.encode(layerCount, forKey: .layerCount)
        try container.encode(reason, forKey: .reason)
        
        // Encode architecture to JSON string
        if let jsonData = try? JSONSerialization.data(withJSONObject: architecture, options: [.prettyPrinted]),
           let jsonString = String(data: jsonData, encoding: .utf8) {
            try container.encode(jsonString, forKey: .architecture)
        } else {
            try container.encode("{}", forKey: .architecture)
        }
    }
}

/// A record of layer changes
public struct LayerChange: Codable {
    /// Timestamp when the change occurred
    public let timestamp: Date
    
    /// Number of layers after the change
    public let layerCount: Int
    
    /// Reason for the change
    public let reason: SnapshotReason
}

/// A record of performance changes
public struct PerformanceChange: Codable {
    /// Timestamp when the change occurred
    public let timestamp: Date
    
    /// Forward pass time in seconds
    public let forwardTime: Double
    
    /// Backward pass time in seconds
    public let backwardTime: Double
    
    /// Memory usage in bytes
    public let memoryUsage: Int64
    
    /// Notes about the performance change
    public let notes: String?
}

/// A record of network evolution over time
public struct NetworkEvolution: Codable {
    /// Network ID
    public let networkId: UUID
    
    /// Network name
    public let networkName: String
    
    /// Network creation date
    public let createdAt: Date
    
    /// Network last modified date
    public var lastModifiedAt: Date
    
    /// Layer changes over time
    public var layerChanges: [LayerChange]
    
    /// Performance changes over time
    public var performanceChanges: [PerformanceChange]
}

// Required for compilation
import Metal 