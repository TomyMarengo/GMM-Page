import React, { useState } from 'react';

import GmmClusteringChart from '@/components/GmmClusteringChart';

interface GmmClusteringSectionProps {
  title: string;
  defaultConfig?: {
    nComponents: number;
    covarianceType: string;
    randomState: number;
    nSamples: number;
    nFeatures: number;
  };
}

const GmmClusteringSection: React.FC<GmmClusteringSectionProps> = ({
  title,
  defaultConfig = {
    nComponents: 3,
    covarianceType: 'full',
    randomState: 42,
    nSamples: 150,
    nFeatures: 2,
  },
}) => {
  const [nComponents, setNComponents] = useState<number>(
    defaultConfig.nComponents
  );
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
      <div className="mb-2">
        <label className="mr-2">Número de clusters:</label>
        <input
          type="number"
          value={nComponents}
          onChange={(e) => setNComponents(parseInt(e.target.value, 10))}
          min={1}
          className="p-1 border border-gray-300 rounded"
        />
      </div>
      <div className="mb-2">
        <label className="mr-2">Número de puntos:</label>
        <input
          type="number"
          value={nSamples}
          onChange={(e) => setNSamples(parseInt(e.target.value, 10))}
          min={10}
          className="p-1 border border-gray-300 rounded"
        />
      </div>
      <div className="mb-2">
        <label className="mr-2">Número de características:</label>
        <input
          type="number"
          value={nFeatures}
          onChange={(e) => setNFeatures(parseInt(e.target.value, 10))}
          min={2}
          max={5}
          className="p-1 border border-gray-300 rounded"
        />
      </div>
      <div className="mb-2">
        <label className="mr-2">Semilla:</label>
        <input
          type="number"
          value={randomState}
          onChange={(e) => setRandomState(parseInt(e.target.value, 10))}
          className="p-1 border border-gray-300 rounded"
        />
      </div>
      <div className="mb-2">
        <label className="mr-2">Tipo de covarianza:</label>
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
      <GmmClusteringChart
        nComponents={nComponents}
        covarianceType={covarianceType}
        randomState={randomState}
        nSamples={nSamples}
        nFeatures={nFeatures}
      />
    </div>
  );
};

export default GmmClusteringSection;
