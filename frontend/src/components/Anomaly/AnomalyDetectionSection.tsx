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
    nEstimators?: number; // Solo para Isolation Forest
    maxSamples?: string | number; // Solo para Isolation Forest
  };
}

const AnomalyDetectionSection: React.FC<AnomalyDetectionSectionProps> = ({
  title,
  defaultConfig,
}) => {
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

  // Nuevos estados para Isolation Forest
  const [nEstimators, setNEstimators] = useState<number>(
    defaultConfig.nEstimators || 100
  );
  const [maxSamples, setMaxSamples] = useState<string | number>(
    defaultConfig.maxSamples || 'auto'
  );

  const handleMaxSamplesChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    if (value.toLowerCase() === 'auto') {
      setMaxSamples('auto');
    } else {
      const num = Number(value);
      if (!isNaN(num)) {
        setMaxSamples(num);
      } else {
        setMaxSamples('auto');
      }
    }
  };

  return (
    <div className="p-4 border border-gray-300 rounded">
      <h2 className="text-2xl gap-2 font-bold mb-4 flex items-center flex-wrap">
        {title}
      </h2>
      {/* Contenedor de formulario con altura fija y los campos alineados al inicio */}
      <div className="grid grid-cols-2 gap-y-2 gap-x-4 mb-4">
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

        {/* Campos adicionales para Isolation Forest */}
        {defaultConfig.algorithm === 'IsolationForest' ? (
          <>
            <div className="flex flex-col">
              <label className="mb-1">Número de estimadores:</label>
              <input
                type="number"
                value={nEstimators}
                onChange={(e) => setNEstimators(parseInt(e.target.value, 10))}
                min={50}
                className="p-1 border border-gray-300 rounded"
              />
            </div>
            <div className="flex flex-col">
              <label className="mb-1">Máximo de muestras:</label>
              <input
                type="text"
                value={maxSamples}
                onChange={handleMaxSamplesChange}
                placeholder="'auto' o número"
                className="p-1 border border-gray-300 rounded"
              />
            </div>
          </>
        ) : (
          <div className="flex flex-col">
            <label className="mb-1"> GMM componentes:</label>
            <input
              type="number"
              value={nComponents}
              onChange={(e) => setNComponents(parseInt(e.target.value, 10))}
              min={1}
              className="p-1 border border-gray-300 rounded"
            />
          </div>
        )}
      </div>

      {/* Contenedor del gráfico */}
      <AnomalyDetectionChart
        algorithm={defaultConfig.algorithm}
        contamination={contamination}
        nComponents={nComponents}
        randomState={randomState}
        nSamples={nSamples}
        nFeatures={nFeatures}
        {...(defaultConfig.algorithm === 'IsolationForest' && {
          nEstimators,
          maxSamples,
        })}
      />
    </div>
  );
};

export default AnomalyDetectionSection;
