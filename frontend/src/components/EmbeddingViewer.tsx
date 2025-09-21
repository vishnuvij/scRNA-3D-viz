import { OrbitControls } from '@react-three/drei';
import { Canvas } from '@react-three/fiber';
import { Suspense, useMemo } from 'react';
import * as THREE from 'three';

import type { EmbeddingPoint } from '../types';

const CLUSTER_COLORS = ['#ef476f', '#ffd166', '#06d6a0', '#118ab2', '#073b4c', '#8338ec'];

interface EmbeddingViewerProps {
  points: EmbeddingPoint[];
  loading: boolean;
  onSelect: (cellId: string | null) => void;
  selectedCellId?: string | null;
}

interface PointsCloudProps {
  points: EmbeddingPoint[];
  onSelect: (cellId: string | null) => void;
  selectedCellId?: string | null;
}

const PointsCloud = ({ points, onSelect, selectedCellId }: PointsCloudProps) => {
  const { positions, colors } = useMemo(() => {
    const positions = new Float32Array(points.length * 3);
    const colors = new Float32Array(points.length * 3);
    const colorCache = new Map<number, [number, number, number]>();
    for (let i = 0; i < points.length; i += 1) {
      const point = points[i];
      positions[i * 3] = point.x;
      positions[i * 3 + 1] = point.y;
      positions[i * 3 + 2] = point.z;

      const cached = colorCache.get(point.cluster);
      let color: [number, number, number];
      if (cached) {
        color = cached;
      } else {
        const hex = CLUSTER_COLORS[point.cluster % CLUSTER_COLORS.length];
        const rgb = new THREE.Color(hex).toArray() as [number, number, number];
        colorCache.set(point.cluster, rgb);
        color = rgb;
      }

      colors[i * 3] = color[0];
      colors[i * 3 + 1] = color[1];
      colors[i * 3 + 2] = color[2];
    }
    return { positions, colors };
  }, [points]);

  const selectedPosition = useMemo(() => {
    if (!selectedCellId) {
      return null;
    }
    const match = points.find((point) => point.cell_id === selectedCellId);
    return match ? [match.x, match.y, match.z] : null;
  }, [points, selectedCellId]);

  return (
    <>
      <points
        onPointerDown={(event) => {
          event.stopPropagation();
          const index = event.index ?? 0;
          const point = points[index];
          if (point) {
            onSelect(point.cell_id);
          }
        }}
      >
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" array={positions} count={points.length} itemSize={3} />
          <bufferAttribute attach="attributes-color" array={colors} count={points.length} itemSize={3} />
        </bufferGeometry>
        <pointsMaterial vertexColors size={0.08} sizeAttenuation />
      </points>
      {selectedPosition && (
        <mesh position={selectedPosition}>
          <sphereGeometry args={[0.18, 16, 16]} />
          <meshBasicMaterial color="#ffffff" />
        </mesh>
      )}
    </>
  );
};

export const EmbeddingViewer = ({ points, loading, onSelect, selectedCellId }: EmbeddingViewerProps) => (
  <div className="viewer">
    {loading && <div className="loading-overlay">Loading embedding…</div>}
    <Canvas camera={{ position: [3, 3, 3], fov: 50 }}>
      <color attach="background" args={[0.02, 0.02, 0.04]} />
      <ambientLight intensity={0.8} />
      <pointLight position={[10, 10, 10]} intensity={0.6} />
      <Suspense fallback={null}>
        <PointsCloud points={points} loading={loading} onSelect={onSelect} selectedCellId={selectedCellId} />
      </Suspense>
      <OrbitControls enablePan enableRotate enableZoom />
    </Canvas>
  </div>
);
