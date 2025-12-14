import { useRef, useMemo, useLayoutEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stars, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';
import type { NeuronTopology } from './useBrainSimulation';

interface BrainVizProps {
  topology: NeuronTopology[];
  activeIndices: number[];
}

const REGION_COLORS: Record<string, string> = {
  'cortex': '#4b0082', // Royal Purple
  'basal_ganglia': '#ffff00', // Yellow (Dopamine)
  'hippocampus': '#00ff00', // Green
  'thalamus': '#ff00ff', // Magenta
  'amygdala': '#ff0000', // Red
  'cerebellum': '#00ffff', // Cyan
  'interneurons': '#ffffff', // White
};

function Neurons({ topology, activeIndices }: BrainVizProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const tempObject = new THREE.Object3D();
  const tempColor = new THREE.Color();
  const baseColors = useMemo(() => new Float32Array(topology.length * 3), [topology]);

  // Initialize base colors once
  useLayoutEffect(() => {
    if (!meshRef.current) return;

    topology.forEach((n, i) => {
      tempObject.position.set(n.x, n.z, n.y); // Swap Y/Z for better camera angle
      tempObject.scale.setScalar(1.5); // Neuron size
      tempObject.updateMatrix();
      meshRef.current!.setMatrixAt(i, tempObject.matrix);

      const color = new THREE.Color(REGION_COLORS[n.region] || '#888888');
      color.toArray(baseColors, i * 3);
      meshRef.current!.setColorAt(i, color);
    });
    meshRef.current.instanceMatrix.needsUpdate = true;
    meshRef.current.instanceColor!.needsUpdate = true;
  }, [topology]);

  // Update activity each frame (or whenever stats change)
  // Note: For 60fps smoothing, we could decay colors over time in useFrame
  useFrame(() => {
    if (!meshRef.current) return;

    // Reset all to base color (dimmed)
    // Optimization: Only reset active ones from previous frame if possible,
    // but iterating 3000 items is fast enough in JS.
    for (let i = 0; i < topology.length; i++) {
        // Base dim color
        tempColor.fromArray(baseColors, i * 3).multiplyScalar(0.2);
        meshRef.current.setColorAt(i, tempColor);
    }

    // Light up active neurons
    for (const idx of activeIndices) {
      if (idx < topology.length) {
        // Flash bright
        tempColor.fromArray(baseColors, idx * 3).multiplyScalar(5.0).addScalar(0.2);
        meshRef.current.setColorAt(idx, tempColor);
      }
    }

    meshRef.current.instanceColor!.needsUpdate = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, topology.length]}>
      <sphereGeometry args={[1, 8, 8]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  );
}

// Synapses (Connections) - Static for now to save FPS
// We draw lines between nearby neurons in same region to simulate density
function Synapses({ topology }: { topology: NeuronTopology[] }) {
    const lines = useMemo(() => {
        const points = [];
        const maxDist = 15.0; // Connection distance
        // Sample connections (too many to draw all n^2)
        // Connect each neuron to 2 nearest neighbors
        for (let i = 0; i < topology.length; i+=2) { // Skip every other for perf
            const n1 = topology[i];
            let connections = 0;
            for (let j = i + 1; j < topology.length; j+=5) {
                if (connections >= 2) break;
                const n2 = topology[j];
                // Only connect within region
                if (n1.region !== n2.region) continue;

                const dx = n1.x - n2.x;
                const dy = n1.y - n2.y;
                const dz = n1.z - n2.z;
                const distSq = dx*dx + dy*dy + dz*dz;

                if (distSq < maxDist * maxDist) {
                    points.push(new THREE.Vector3(n1.x, n1.z, n1.y));
                    points.push(new THREE.Vector3(n2.x, n2.z, n2.y));
                    connections++;
                }
            }
        }
        return new THREE.BufferGeometry().setFromPoints(points);
    }, [topology]);

    return (
        <lineSegments geometry={lines}>
            <lineBasicMaterial color="#333333" transparent opacity={0.3} />
        </lineSegments>
    );
}

export default function BrainScene({ topology, activeIndices }: BrainVizProps) {
  return (
    <Canvas>
      <PerspectiveCamera makeDefault position={[0, 100, 250]} />
      <OrbitControls autoRotate autoRotateSpeed={0.5} />
      <color attach="background" args={['#050505']} />
      <Stars radius={300} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={1} />

      {topology.length > 0 && (
        <>
            <Neurons topology={topology} activeIndices={activeIndices} />
            <Synapses topology={topology} />
        </>
      )}
    </Canvas>
  );
}
