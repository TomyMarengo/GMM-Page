// src/components/GmmClusteringSection.tsx
import React, { useState } from 'react';

import GmmClusteringChart from '@/components/Clustering/GmmClusteringChart';

interface GmmClusteringSectionProps {
  title: string;
  defaultConfig: {
    nComponents: number;
    nCenters: number;
    covarianceType: string;
    randomState: number;
    nSamples: number;
    nFeatures: number;
  };
}

const GmmClusteringSection: React.FC<GmmClusteringSectionProps> = ({
  title,
  defaultConfig,
}) => {
  const [nComponents, setNComponents] = useState<number>(
    defaultConfig.nComponents
  );
  const [nCenters, setNCenters] = useState<number>(defaultConfig.nCenters);
  const [covarianceType, setCovarianceType] = useState<string>(
    defaultConfig.covarianceType
  );
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
              value={nComponents}
              onChange={(e) => setNComponents(parseInt(e.target.value, 10))}
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
          <div className="flex flex-col">
            <label className="mb-1">Tipo de covarianza:</label>
            <select
              value={covarianceType}
              onChange={(e) => setCovarianceType(e.target.value)}
              className="p-1 border border-gray-300 rounded"
            >
              <option value="full">full</option>
              <option value="tied">tied</option>
              <option value="diag">diag</option>
              <option value="spherical">spherical</option>
            </select>
          </div>
        </div>
      </div>
      {/* Contenedor del gráfico */}
      <GmmClusteringChart
        nComponents={nComponents}
        nCenters={nCenters}
        covarianceType={covarianceType}
        randomState={randomState}
        nSamples={nSamples}
        nFeatures={nFeatures}
      />
    </div>
  );
};

export default GmmClusteringSection;
