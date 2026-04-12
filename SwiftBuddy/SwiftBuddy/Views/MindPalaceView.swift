import SwiftUI
import SwiftData

struct MindPalaceView: View {
    @Environment(\.dismiss) private var dismiss
    @Query var triples: [KnowledgeGraphTriple]
    
    // Physics Simulation State
    @State private var nodes: [String: ForceNode] = [:]
    @State private var edges: [ForceEdge] = []
    @State private var isSimulating = false
    @State private var isPhysicsSettled = false
    @State private var draggedNodeId: String? = nil
    
    // Timer
    private let timer = Timer.publish(every: 1.0 / 60.0, on: .main, in: .common).autoconnect()
    
    var body: some View {
        ZStack(alignment: .topTrailing) {
            SwiftBuddyTheme.background.ignoresSafeArea()
            
            if triples.isEmpty {
                VStack(spacing: 20) {
                    Image(systemName: "network")
                        .font(.system(size: 60))
                        .foregroundStyle(.secondary)
                    Text("No Synaptic Triples Formed")
                        .font(.title2).bold()
                    Text("Converse with a persona to trigger Synaptic Synthesis.")
                        .foregroundStyle(.secondary)
                }
            } else {
                GeometryReader { proxy in
                    let canvasCenter = CGPoint(x: proxy.size.width / 2, y: proxy.size.height / 2)
                    
                    Canvas { context, size in
                        // Draw Edges
                        for edge in edges {
                            if let source = nodes[edge.sourceId], let target = nodes[edge.targetId] {
                                var path = Path()
                                path.move(to: source.position)
                                path.addLine(to: target.position)
                                
                                context.stroke(path, with: .color(.secondary.opacity(0.3)), lineWidth: 1.5)
                                
                                // Midpoint for label
                                let midX = (source.position.x + target.position.x) / 2
                                let midY = (source.position.y + target.position.y) / 2
                                let angle = atan2(target.position.y - source.position.y, target.position.x - source.position.x)
                                
                                context.translateBy(x: midX, y: midY)
                                context.rotate(by: Angle(radians: Double(angle)))
                                // Draw edge predicate label with a slightly larger font
                                context.draw(Text(edge.predicate).font(.caption).bold().foregroundColor(SwiftBuddyTheme.accent), at: .zero)
                                context.rotate(by: Angle(radians: -Double(angle)))
                                context.translateBy(x: -midX, y: -midY)
                            }
                        }
                        
                        // Draw Nodes
                        for (_, node) in nodes {
                            let rect = CGRect(x: node.position.x - 10, y: node.position.y - 10, width: 20, height: 20)
                            context.fill(Path(ellipseIn: rect), with: .color(SwiftBuddyTheme.accent))
                            
                            // Let's drop a little blurred halo for the cyberpunk glow
                            context.stroke(Path(ellipseIn: rect.insetBy(dx: -3, dy: -3)), with: .color(SwiftBuddyTheme.accent.opacity(0.4)), lineWidth: 2)
                            
                            context.draw(Text(node.name).font(.callout.bold()), at: CGPoint(x: node.position.x, y: node.position.y + 26))
                        }
                    }
                    .gesture(
                        DragGesture(minimumDistance: 0)
                            .onChanged { value in
                                // Find nearest
                                if draggedNodeId == nil {
                                    if let nearest = nodes.values.min(by: { distance($0.position, value.location) < distance($1.position, value.location) }) {
                                        if distance(nearest.position, value.location) < 30 {
                                            draggedNodeId = nearest.id
                                            isPhysicsSettled = false
                                        }
                                    }
                                }
                                
                                if let id = draggedNodeId {
                                    nodes[id]?.position = value.location
                                    nodes[id]?.velocity = .zero
                                }
                            }
                            .onEnded { _ in
                                draggedNodeId = nil
                            }
                    )
                    .onReceive(timer) { _ in
                        if !isPhysicsSettled {
                            stepSimulation(center: canvasCenter)
                        }
                    }
                }
            }
            #if os(macOS)
            Button(action: { dismiss() }) {
                Image(systemName: "xmark.circle.fill")
                    .font(.system(size: 24))
                    .foregroundStyle(.secondary.opacity(0.8))
            }
            .buttonStyle(.plain)
            .padding(20)
            #endif
        }
        .onAppear {
            initializeGraph()
        }
        .onChange(of: triples.count) { _ in
            initializeGraph() // Re-init if new edges arrive!
        }
        #if os(macOS)
        .navigationTitle("Mind Palace")
        #endif
    }
    
    // MARK: - Simulation Engine
    
    private func initializeGraph() {
        var newNodes: [String: ForceNode] = [:]
        var newEdges: [ForceEdge] = []
        
        let cx = NSScreen.main?.frame.width ?? 800
        let cy = NSScreen.main?.frame.height ?? 600
        
        for triple in triples {
            let sId = triple.subject.lowercased()
            let oId = triple.object.lowercased()
            
            if newNodes[sId] == nil {
                newNodes[sId] = ForceNode(id: sId, name: triple.subject, position: randomPoint(around: CGPoint(x: cx/2, y: cy/2)))
            }
            if newNodes[oId] == nil {
                newNodes[oId] = ForceNode(id: oId, name: triple.object, position: randomPoint(around: CGPoint(x: cx/2, y: cy/2)))
            }
            
            newEdges.append(ForceEdge(sourceId: sId, targetId: oId, predicate: triple.predicate))
        }
        
        self.nodes = newNodes
        self.edges = newEdges
        self.isPhysicsSettled = false
    }
    
    private func randomPoint(around center: CGPoint) -> CGPoint {
        let r = CGFloat.random(in: 0...200)
        let theta = CGFloat.random(in: 0...(2 * .pi))
        return CGPoint(x: center.x + r * cos(theta), y: center.y + r * sin(theta))
    }
    
    private func distance(_ p1: CGPoint, _ p2: CGPoint) -> CGFloat {
        hypot(p1.x - p2.x, p1.y - p2.y)
    }
    
    private func stepSimulation(center: CGPoint) {
        var totalDisplacement: CGFloat = 0
        let k: CGFloat = 0.5 // Spring constant
        let repulsion: CGFloat = 12000 // Increased Node repulsion to spread text
        let damping: CGFloat = 0.85
        let centerGravity: CGFloat = 0.005 // Reduced pull to center to allow spread
        
        // 1. Calculate Repulsion (Coulomb)
        let nodeValues = Array(nodes.values)
        for i in 0..<nodeValues.count {
            for j in (i+1)..<nodeValues.count {
                let n1 = nodeValues[i]
                let n2 = nodeValues[j]
                
                let dx = n1.position.x - n2.position.x
                let dy = n1.position.y - n2.position.y
                let dist = max(hypot(dx, dy), 1) // Prevent div 0
                
                let force = repulsion / (dist * dist)
                let fx = (dx / dist) * force
                let fy = (dy / dist) * force
                
                if n1.id != draggedNodeId {
                    nodes[n1.id]?.velocity.x += fx
                    nodes[n1.id]?.velocity.y += fy
                }
                if n2.id != draggedNodeId {
                    nodes[n2.id]?.velocity.x -= fx
                    nodes[n2.id]?.velocity.y -= fy
                }
            }
        }
        
        // 2. Calculate Spring Attraction (Hooke's)
        for edge in edges {
            guard let n1 = nodes[edge.sourceId], let n2 = nodes[edge.targetId] else { continue }
            
            let dx = n2.position.x - n1.position.x
            let dy = n2.position.y - n1.position.y
            let dist = max(hypot(dx, dy), 1)
            
            let force = (dist - 280) * k // Increased Target length to 280 to prevent word bunching
            let fx = (dx / dist) * force
            let fy = (dy / dist) * force
            
            if n1.id != draggedNodeId {
                nodes[n1.id]?.velocity.x += fx
                nodes[n1.id]?.velocity.y += fy
            }
            if n2.id != draggedNodeId {
                nodes[n2.id]?.velocity.x -= fx
                nodes[n2.id]?.velocity.y -= fy
            }
        }
        
        // 3. Center Gravity & Integration
        for (id, node) in nodes {
            if id == draggedNodeId { continue }
            
            // Gravity
            nodes[id]?.velocity.x += (center.x - node.position.x) * centerGravity
            nodes[id]?.velocity.y += (center.y - node.position.y) * centerGravity
            
            // Damping & application
            nodes[id]?.velocity.x *= damping
            nodes[id]?.velocity.y *= damping
            
            nodes[id]?.position.x += (nodes[id]?.velocity.x ?? 0)
            nodes[id]?.position.y += (nodes[id]?.velocity.y ?? 0)
            
            totalDisplacement += abs(nodes[id]?.velocity.x ?? 0) + abs(nodes[id]?.velocity.y ?? 0)
        }
        
        // Settle condition
        if totalDisplacement < 0.5 {
            isPhysicsSettled = true
        }
    }
}

// Data structures
fileprivate struct ForceNode {
    let id: String
    let name: String
    var position: CGPoint
    var velocity: CGPoint = .zero
}

fileprivate struct ForceEdge {
    let sourceId: String
    let targetId: String
    let predicate: String
}
