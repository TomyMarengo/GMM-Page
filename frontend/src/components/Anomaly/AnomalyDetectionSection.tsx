// src/components/AnomalyDetectionSection.tsx
import React, { useState } from 'react';

import AnomalyDetectionChart from '@/components/Anomaly/AnomalyDetectionChart';

interface AnomalyDetectionSectionProps {
  title: string;
  defaultConfig: {
    algorithm: 'GMM' | 'IsolationForest';
    contamination: number;
    nComponents?: number;
    randomState: number;
    nSamples: number;
    nFeatures: number;
  };
}

const AnomalyDetectionSection: React.FC<AnomalyDetectionSectionProps> = ({
  title,
  defaultConfig,
}) => {
  const [algorithm, setAlgorithm] = useState<'GMM' | 'IsolationForest'>(
    defaultConfig.algorithm
  );
  const [contamination, setContamination] = useState<number>(
    defaultConfig.contamination
  );
  const [nComponents, setNComponents] = useState<number>(
    defaultConfig.nComponents || 3
  );
  const [randomState, setRandomState] = useState<number>(
    defaultConfig.randomState
  );
  const [nSamples, setNSamples] = useState<number>(defaultConfig.nSamples);
  const [nFeatures, setNFeatures] = useState<number>(defaultConfig.nFeatures);

  return (
    <div className="p-4 border border-gray-300 rounded">
      <h2 className="text-2xl font-bold mb-4">
        {title}
        {algorithm === 'GMM' && (
          <>
            <label className="mb-1 mx-2 font-bold">con componentes:</label>
            <input
              type="number"
              value={nComponents}
              onChange={(e) => setNComponents(parseInt(e.target.value, 10))}
              min={1}
              className="p-1 border border-gray-300 rounded"
            />
          </>
        )}
      </h2>
      {/* Contenedor de formulario con altura fija y los campos alineados al inicio */}
      <div style={{ height: '250px' }}>
        <div className="grid grid-cols-2 gap-y-2 gap-x-4 mb-4">
          <div className="flex flex-col">
            <label className="mb-1">Algoritmo:</label>
            <select
              value={algorithm}
              onChange={(e) =>
                setAlgorithm(e.target.value as 'GMM' | 'IsolationForest')
              }
              className="p-1 border border-gray-300 rounded"
            >
              <option value="GMM">GMM</option>
              <option value="IsolationForest">Isolation Forest</option>
            </select>
          </div>
          <div className="flex flex-col">
            <label className="mb-1">Contaminación (%):</label>
            <input
              type="number"
              value={contamination}
              onChange={(e) => setContamination(parseFloat(e.target.value))}
              min={0}
              max={0.5}
              step={0.01}
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
      <AnomalyDetectionChart
        algorithm={algorithm}
        contamination={contamination}
        nComponents={nComponents}
        randomState={randomState}
        nSamples={nSamples}
        nFeatures={nFeatures}
      />
    </div>
  );
};

export default AnomalyDetectionSection;
