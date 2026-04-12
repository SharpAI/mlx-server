// PalaceVisualizerView.swift
import SwiftUI
import SwiftData

struct PalaceVisualizerView: View {
    @Environment(\.dismiss) private var dismiss
    @Query(sort: \PalaceWing.createdDate) var wings: [PalaceWing]
    @State private var expandedRooms: Set<String> = []
    
    var body: some View {
        ZStack(alignment: .topTrailing) {
        ScrollView([.horizontal, .vertical], showsIndicators: true) {
            VStack(alignment: .leading, spacing: 0) {
                ForEach(wings) { wing in
                    VStack(alignment: .leading, spacing: 0) {
                        
                        // Tunnel Connection from previous Wing
                        if wing.name != wings.first?.name {
                            TunnelConnector()
                        }
                        
                        // Main Wing Container
                        WingNodeView(wing: wing, expandedRooms: $expandedRooms)
                    }
                }
                
                if wings.isEmpty {
                    VStack(spacing: 20) {
                        Image(systemName: "brain.head.profile")
                            .font(.system(size: 60))
                            .foregroundStyle(.secondary)
                        Text("No Architectures Built")
                            .font(.title2).bold()
                        Text("Use the Inspector Registry to download a Persona.")
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .padding(.top, 100)
                }
            }
            .padding(60)
        }
        .background(SwiftBuddyTheme.background.ignoresSafeArea())
        
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
        #if os(macOS)
        .navigationTitle("Palace Visualizer")
        #endif
    }
}

struct TunnelConnector: View {
    var body: some View {
        HStack {
            Rectangle()
                .fill(Color.secondary.opacity(0.4))
                .frame(width: 2, height: 40)
                .padding(.leading, 60) // Align to the visual 'tunnel' anchor
            Text("tunnel")
                .font(.caption2.monospaced())
                .foregroundStyle(.secondary)
                .padding(.leading, 4)
            Spacer()
        }
    }
}

struct WingNodeView: View {
    let wing: PalaceWing
    @Binding var expandedRooms: Set<String>
    
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            headerView
            roomsLayoutView
        }
        .padding(30)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(nsColor: .controlBackgroundColor).opacity(0.6))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(SwiftBuddyTheme.divider, lineWidth: 1)
        )
    }
    
    private var headerView: some View {
        HStack {
            Image(systemName: "building.2.crop.circle.fill")
                .foregroundStyle(SwiftBuddyTheme.accent)
            Text("WING: \(wing.name)")
                .font(.headline).bold()
                .monospaced()
        }
        .padding(.bottom, 10)
    }
    
    private var roomsLayoutView: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(alignment: .top, spacing: 0) {
                ForEach(wing.rooms) { room in
                    HStack(alignment: .top, spacing: 0) {
                        RoomNodeView(room: room, isExpanded: Binding(
                            get: { expandedRooms.contains(room.name) } ,
                            set: { exp in 
                                if exp { expandedRooms.insert(room.name) }
                                else { expandedRooms.remove(room.name) }
                            }
                        ))
                        
                        if room.name != wing.rooms.last?.name {
                            hallConnector
                        }
                    }
                }
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 30)
        }
    }
    
    private var hallConnector: some View {
        HStack(spacing: 0) {
            Rectangle()
                .fill(Color.secondary.opacity(0.3))
                .frame(width: 20, height: 2)
            Text("hall")
                .font(.system(size: 9, design: .monospaced))
                .foregroundStyle(.secondary)
                .padding(.horizontal, 4)
            Rectangle()
                .fill(Color.secondary.opacity(0.3))
                .frame(width: 20, height: 2)
        }
        .padding(.top, 25)
    }
}

struct RoomNodeView: View {
    let room: PalaceRoom
    @Binding var isExpanded: Bool
    
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Main Room Block
            Button(action: {
                withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
                    isExpanded.toggle()
                }
            }) {
                VStack(spacing: 8) {
                    Text("Room: \(room.name)")
                        .font(.subheadline)
                        .bold()
                    Text("\(room.memories.count) facts")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                .frame(width: 140, height: 50)
                .background(Color(nsColor: .controlBackgroundColor))
                .cornerRadius(6)
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(isExpanded ? SwiftBuddyTheme.accent : Color.secondary.opacity(0.4), lineWidth: 1)
                )
            }
            .buttonStyle(.plain)
            
            // Closet Dropdown Expansion
            if isExpanded {
                VStack(alignment: .center, spacing: 0) {
                    // Vertical arrow pointing to Closet
                    Rectangle()
                        .fill(Color.secondary.opacity(0.4))
                        .frame(width: 1, height: 20)
                    
                    Image(systemName: "arrowtriangle.down.fill")
                        .font(.system(size: 8))
                        .foregroundStyle(.secondary)
                        .padding(.bottom, 4)
                    
                    // Closet / Drawer Block
                    VStack(alignment: .leading, spacing: 14) {
                        HStack {
                            Image(systemName: "cabinet.fill")
                                .foregroundStyle(SwiftBuddyTheme.accent.opacity(0.8))
                            Text("Closet")
                                .font(.caption).bold()
                        }
                        
                        Divider()
                        
                        // Drawers (Individual Memories) limiting to top 15 for rendering performance
                        ForEach(room.memories.prefix(15)) { memory in
                            HStack(alignment: .top) {
                                Image(systemName: "arrow.turn.down.right")
                                    .font(.system(size: 10))
                                    .foregroundStyle(.secondary)
                                    .padding(.top, 2)
                                
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("Drawer: \(memory.hallType)")
                                        .font(.system(size: 9, weight: .bold, design: .monospaced))
                                        .foregroundStyle(SwiftBuddyTheme.accent)
                                    
                                    Text(memory.text)
                                        .font(.caption2)
                                        .foregroundStyle(.primary.opacity(0.8))
                                        .lineLimit(4)
                                }
                            }
                            .padding(.bottom, 6)
                        }
                        
                        if room.memories.count > 15 {
                            Text("+ \(room.memories.count - 15) more hidden text fragments")
                                .font(.caption2).italic()
                                .foregroundStyle(.secondary)
                        }
                    }
                    .padding(16)
                    .frame(width: 280)
                    .background(Color(nsColor: .windowBackgroundColor).opacity(0.8))
                    .cornerRadius(8)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.secondary.opacity(0.2), lineWidth: 1)
                    )
                }
                // shift closet to align visually with the center of the room block
                .padding(.leading, 70 - 140) 
            }
        }
        // Force width footprint minimum to not collapse the hall spacing when closet is expanded
        .frame(width: 140) 
    }
}
