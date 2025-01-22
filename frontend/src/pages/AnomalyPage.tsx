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

  return (
    <div className="p-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Sección 1 */}
        <div className="border border-gray-300 rounded p-4 flex flex-col">
          <h2 className="text-2xl font-bold mb-4">Configuración 1</h2>
          <div className="mb-4">
            <label className="mr-2 font-semibold">Algoritmo:</label>
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
          <div className="mb-4 text-gray-700 h-72">
            <ReactMarkdown components={markdownComponents}>
              {anomalyDescriptions[algorithm1]}
            </ReactMarkdown>
          </div>

          {/* Sección de Detección de Anomalías */}
          <AnomalyDetectionSection
            title={algorithm1}
            defaultConfig={
              algorithm1 === 'GMM'
                ? {
                    algorithm: 'GMM',
                    contamination: 0.05,
                    randomState: 42,
                    nSamples: 150,
                    nFeatures: 2,
                    nComponents: 3,
                  }
                : {
                    algorithm: 'IsolationForest',
                    contamination: 0.05,
                    randomState: 42,
                    nSamples: 150,
                    nFeatures: 2,
                  }
            }
          />
        </div>

        {/* Sección 2 */}
        <div className="border border-gray-300 rounded p-4 flex flex-col">
          <h2 className="text-2xl font-bold mb-4">Configuración 2</h2>
          <div className="mb-4">
            <label className="mr-2 font-semibold">Algoritmo:</label>
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
          <div className="mb-4 text-gray-700 h-72">
            <ReactMarkdown components={markdownComponents}>
              {anomalyDescriptions[algorithm2]}
            </ReactMarkdown>
          </div>

          {/* Sección de Detección de Anomalías */}
          <AnomalyDetectionSection
            title={algorithm2}
            defaultConfig={
              algorithm2 === 'GMM'
                ? {
                    algorithm: 'GMM',
                    contamination: 0.05,
                    randomState: 42,
                    nSamples: 150,
                    nFeatures: 2,
                    nComponents: 3,
                  }
                : {
                    algorithm: 'IsolationForest',
                    contamination: 0.05,
                    randomState: 42,
                    nSamples: 150,
                    nFeatures: 2,
                  }
            }
          />
        </div>
      </div>
    </div>
  );
};

export default AnomalyPage;
