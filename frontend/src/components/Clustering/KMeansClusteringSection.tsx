// src/components/KMeansClusteringSection.tsx
import React, { useState } from 'react';

import KMeansClusteringChart from '@/components/Clustering/KMeansClusteringChart';

interface KMeansClusteringSectionProps {
  title: string;
  defaultConfig: {
    nClusters: number;
    nCenters: number;
    randomState: number;
    nSamples: number;
    nFeatures: number;
  };
}

const KMeansClusteringSection: React.FC<KMeansClusteringSectionProps> = ({
  title,
  defaultConfig,
}) => {
  const [nClusters, setNClusters] = useState<number>(defaultConfig.nClusters);
  const [nCenters, setNCenters] = useState<number>(defaultConfig.nCenters);
  const [randomState, setRandomState] = useState<number>(
    defaultConfig.randomState
  );
  const [nSamples, setNSamples] = useState<number>(defaultConfig.nSamples);
  const [nFeatures, setNFeatures] = useState<number>(defaultConfig.nFeatures);

  return (
    <div className="p-4 border border-gray-300 rounded">
      <h2 className="text-2xl font-bold mb-4">{title}</h2>
      {/* Contenedor de formulario con altura fija y los campos alineados al inicio */}
      <div style={{ height: '200px' }}>
        <div className="grid grid-cols-2 gap-y-2 gap-x-4 mb-4">
          <div className="flex flex-col">
            <label className="mb-1">Número de clusters:</label>
            <input
              type="number"
              value={nClusters}
              onChange={(e) => setNClusters(parseInt(e.target.value, 10))}
              min={1}
              className="p-1 border border-gray-300 rounded"
            />
          </div>
          <div className="flex flex-col">
            <label className="mb-1">Número de centros:</label>
            <input
              type="number"
              value={nCenters}
              onChange={(e) => setNCenters(parseInt(e.target.value, 10))}
              min={1}
              className="p-1 border border-gray-300 rounded"
            />
          </div>
          <div className="flex flex-col">
            <label className="mb-1">Número de puntos:</label>
            <input
              type="number"
              value={nSamples}
              onChange={(e) => setNSamples(parseInt(e.target.value, 10))}
              min={10}
              className="p-1 border border-gray-300 rounded"
            />
          </div>
          <div className="flex flex-col">
            <label className="mb-1">Número de características:</label>
            <input
              type="number"
              value={nFeatures}
              onChange={(e) => setNFeatures(parseInt(e.target.value, 10))}
              min={2}
              max={5}
              className="p-1 border border-gray-300 rounded"
            />
          </div>
          <div className="flex flex-col">
            <label className="mb-1">Semilla:</label>
            <input
              type="number"
              value={randomState}
              onChange={(e) => setRandomState(parseInt(e.target.value, 10))}
              className="p-1 border border-gray-300 rounded"
            />
          </div>
        </div>
      </div>

      {/* Contenedor del gráfico */}
      <KMeansClusteringChart
        nClusters={nClusters}
        nCenters={nCenters}
        randomState={randomState}
        nSamples={nSamples}
        nFeatures={nFeatures}
      />
    </div>
  );
};

export default KMeansClusteringSection;
