import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';

import GmmClusteringSection from '@/components/Clustering/GmmClusteringSection';
import KMeansClusteringSection from '@/components/Clustering/KMeansClusteringSection';
import markdownComponents from '@/components/MarkdownComponents';
import { ClusteringAlgorithm, clusteringDescriptions } from '@/constants';

const ClusteringPage: React.FC = () => {
  const [algorithm1, setAlgorithm1] = useState<ClusteringAlgorithm>('GMM');
  const [algorithm2, setAlgorithm2] = useState<ClusteringAlgorithm>('KMeans');

  // Configuraciones por defecto para cada algoritmo
  const getDefaultConfigGMM = () => ({
    nComponents: 3,
    nCenters: 3,
    randomState: 42,
    nSamples: 150,
    nFeatures: 2,
    covarianceType: 'full',
  });

  const getDefaultConfigKMeans = () => ({
    nClusters: 3,
    nCenters: 3,
    randomState: 42,
    nSamples: 150,
    nFeatures: 2,
  });

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
                setAlgorithm1(e.target.value as ClusteringAlgorithm)
              }
              className="p-2 border border-gray-300 rounded"
            >
              <option value="GMM">GMM</option>
              <option value="KMeans">KMeans</option>
            </select>
          </div>

          {/* Descripción Dinámica */}
          <div className="text-gray-700 flex-1">
            <ReactMarkdown components={markdownComponents}>
              {clusteringDescriptions[algorithm1]}
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
                setAlgorithm2(e.target.value as ClusteringAlgorithm)
              }
              className="p-2 border border-gray-300 rounded"
            >
              <option value="GMM">GMM</option>
              <option value="KMeans">KMeans</option>
            </select>
          </div>

          {/* Descripción Dinámica */}
          <div className="text-gray-700 flex-1">
            <ReactMarkdown components={markdownComponents}>
              {clusteringDescriptions[algorithm2]}
            </ReactMarkdown>
          </div>
        </div>
      </div>

      {/* Grid de Sección de Clustering */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
        {/* Configuración 1 - Clustering */}
        <div className="flex flex-col">
          {algorithm1 === 'GMM' ? (
            <GmmClusteringSection
              title="GMM"
              defaultConfig={getDefaultConfigGMM()}
            />
          ) : (
            <KMeansClusteringSection
              title="KMeans"
              defaultConfig={getDefaultConfigKMeans()}
            />
          )}
        </div>

        {/* Configuración 2 - Clustering */}
        <div className="flex flex-col">
          {algorithm2 === 'GMM' ? (
            <GmmClusteringSection
              title="GMM"
              defaultConfig={getDefaultConfigGMM()}
            />
          ) : (
            <KMeansClusteringSection
              title="KMeans"
              defaultConfig={getDefaultConfigKMeans()}
            />
          )}
        </div>
      </div>
    </div>
  );
};

export default ClusteringPage;
