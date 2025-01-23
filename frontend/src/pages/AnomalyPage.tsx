// src/pages/AnomalyPage.tsx

import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';

import AnomalyDetectionSection from '@/components/Anomaly/AnomalyDetectionSection';
import markdownComponents from '@/components/MarkdownComponents';
import { AnomalyAlgorithm, anomalyDescriptions } from '@/constants';

const AnomalyPage: React.FC = () => {
  const [algorithm1, setAlgorithm1] = useState<AnomalyAlgorithm>('GMM');
  const [algorithm2, setAlgorithm2] =
    useState<AnomalyAlgorithm>('IsolationForest');

  // Configuraciones por defecto para cada algoritmo
  const getDefaultConfig = (algorithm: AnomalyAlgorithm) => {
    return algorithm === 'GMM'
      ? {
          algorithm: 'GMM' as const,
          contamination: 0.05,
          randomState: 42,
          nSamples: 150,
          nFeatures: 2,
          nComponents: 3,
        }
      : {
          algorithm: 'IsolationForest' as const,
          contamination: 0.05,
          randomState: 42,
          nSamples: 150,
          nFeatures: 2,
        };
  };

  return (
    <div className="p-4">
      {/* Grid de Descripciones */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Configuración 1 - Descripción */}
        <div className="p-4 border border-gray-300 rounded flex flex-col">
          <h2 className="text-2xl font-bold mb-4">Configuración 1</h2>
          <div className="mb-4 flex items-center gap-2">
            <label className="block mb-1 font-semibold">Algoritmo:</label>
            <select
              value={algorithm1}
              onChange={(e) =>
                setAlgorithm1(e.target.value as AnomalyAlgorithm)
              }
              className="p-2 border border-gray-300 rounded"
            >
              <option value="GMM">GMM</option>
              <option value="IsolationForest">Isolation Forest</option>
            </select>
          </div>

          {/* Descripción Dinámica */}
          <div className="text-gray-700 flex-1">
            <ReactMarkdown components={markdownComponents}>
              {anomalyDescriptions[algorithm1]}
            </ReactMarkdown>
          </div>
        </div>

        {/* Configuración 2 - Descripción */}
        <div className="p-4 border border-gray-300 rounded flex flex-col">
          <h2 className="text-2xl font-bold mb-4">Configuración 2</h2>
          <div className="mb-4 flex items-center gap-2">
            <label className="block mb-1 font-semibold">Algoritmo:</label>
            <select
              value={algorithm2}
              onChange={(e) =>
                setAlgorithm2(e.target.value as AnomalyAlgorithm)
              }
              className="p-2 border border-gray-300 rounded"
            >
              <option value="GMM">GMM</option>
              <option value="IsolationForest">Isolation Forest</option>
            </select>
          </div>

          {/* Descripción Dinámica */}
          <div className="text-gray-700 flex-1">
            <ReactMarkdown components={markdownComponents}>
              {anomalyDescriptions[algorithm2]}
            </ReactMarkdown>
          </div>
        </div>
      </div>

      {/* Grid de Detección de Anomalías */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
        {/* Configuración 1 - Detección */}
        <div className="flex flex-col">
          <AnomalyDetectionSection
            title={algorithm1}
            defaultConfig={getDefaultConfig(algorithm1)}
          />
        </div>

        {/* Configuración 2 - Detección */}
        <div className="flex flex-col">
          <AnomalyDetectionSection
            title={algorithm2}
            defaultConfig={getDefaultConfig(algorithm2)}
          />
        </div>
      </div>
    </div>
  );
};

export default AnomalyPage;
