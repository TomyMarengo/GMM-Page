// src/pages/ClusteringPage.tsx

import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';

import GmmClusteringSection from '@/components/Clustering/GmmClusteringSection';
import KMeansClusteringSection from '@/components/Clustering/KMeansClusteringSection';
import markdownComponents from '@/components/MarkdownComponents';
import { ClusteringAlgorithm, clusteringDescriptions } from '@/constants';

const ClusteringPage: React.FC = () => {
  const [algorithm1, setAlgorithm1] = useState<ClusteringAlgorithm>('GMM');
  const [algorithm2, setAlgorithm2] = useState<ClusteringAlgorithm>('KMeans');

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
                setAlgorithm1(e.target.value as ClusteringAlgorithm)
              }
              className="p-2 border border-gray-300 rounded"
            >
              <option value="GMM">GMM</option>
              <option value="KMeans">KMeans</option>
            </select>
          </div>

          {/* Descripción Dinámica */}
          <div className="mb-4 text-gray-700 h-80">
            <ReactMarkdown components={markdownComponents}>
              {clusteringDescriptions[algorithm1]}
            </ReactMarkdown>
          </div>

          {/* Sección de Clustering */}
          {algorithm1 === 'GMM' ? (
            <GmmClusteringSection
              title="GMM"
              defaultConfig={{
                nComponents: 3,
                nCenters: 3,
                randomState: 42,
                nSamples: 150,
                nFeatures: 2,
                covarianceType: 'full',
              }}
            />
          ) : (
            <KMeansClusteringSection
              title="KMeans"
              defaultConfig={{
                nClusters: 3,
                nCenters: 3,
                randomState: 42,
                nSamples: 150,
                nFeatures: 2,
              }}
            />
          )}
        </div>

        {/* Sección 2 */}
        <div className="border border-gray-300 rounded p-4 flex flex-col">
          <h2 className="text-2xl font-bold mb-4">Configuración 2</h2>
          <div className="mb-4">
            <label className="mr-2 font-semibold">Algoritmo:</label>
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
          <div className="mb-4 text-gray-700 h-80">
            <ReactMarkdown components={markdownComponents}>
              {clusteringDescriptions[algorithm2]}
            </ReactMarkdown>
          </div>

          {/* Sección de Clustering */}
          {algorithm2 === 'GMM' ? (
            <GmmClusteringSection
              title="GMM"
              defaultConfig={{
                nComponents: 3,
                nCenters: 3,
                randomState: 42,
                nSamples: 150,
                nFeatures: 2,
                covarianceType: 'full',
              }}
            />
          ) : (
            <KMeansClusteringSection
              title="KMeans"
              defaultConfig={{
                nClusters: 3,
                nCenters: 3,
                randomState: 42,
                nSamples: 150,
                nFeatures: 2,
              }}
            />
          )}
        </div>
      </div>
    </div>
  );
};

export default ClusteringPage;
