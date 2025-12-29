# Spatial Navigation

The spatial system provides the agent with a sense of "where" it is, supporting both physical navigation and abstract cognitive mapping.

## Grid Cells (Entorhinal Cortex)

Grid cells provide a metric coordinate system for the brain.

*   **Firing Pattern**: Hexagonal lattice. A single cell fires at multiple locations forming a perfect grid.
*   **Mechanism**: Implemented via **Path Integration**.
    *   Input: Velocity vector $(v_x, v_y)$.
    *   Math: Sum of 3 cosine waves oscillating at $60^{\circ}$ intervals.
*   **Modules**: We simulate multiple grid modules with different scales (spacings).
    *   Small scale: ~30cm.
    *   Large scale: ~10m.
    *   Scale ratio: $\approx 1.42$ (optimal for resolution).

## Place Cells (Hippocampus)

Place cells represent specific locations in a specific context.

*   **Firing Pattern**: Gaussian field. Fires only when the agent is in a specific spot.
*   **Formation**: Derived from the intersection of multiple Grid Cell inputs.
*   **Remapping**: When the environment changes (e.g., lights go out, room shape changes), Place Cells "remap" (assign to new random locations), creating a unique orthogonal code for that context.

## Semantic Spaces (The "Cognitive Map")

Recent research suggests the brain uses grid/place codes for non-spatial data too. NeuroxAI implements this **Semantic Grid**.

*   **Concept Mapping**: Words and concepts are mapped into a 2D/3D manifold.
*   **Navigation**: "Thinking" becomes navigation through concept space.
*   **Distance**: Semantic similarity = Euclidean distance in grid space.
