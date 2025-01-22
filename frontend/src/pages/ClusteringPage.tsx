import React from 'react';

import GmmClusteringSection from '@/components/GmmClusteringSection';

const ClusteringPage: React.FC = () => {
  return (
    <div className="grid grid-cols-2 gap-4 p-4">
      <GmmClusteringSection
        title="Configuración 1"
        defaultConfig={{
          nComponents: 3,
          covarianceType: 'full',
          randomState: 42,
          nSamples: 150,
          nFeatures: 2,
        }}
      />
      <GmmClusteringSection
        title="Configuración 2"
        defaultConfig={{
          nComponents: 4,
          covarianceType: 'diag',
          randomState: 123,
          nSamples: 200,
          nFeatures: 2,
        }}
      />
    </div>
  );
};

export default ClusteringPage;
