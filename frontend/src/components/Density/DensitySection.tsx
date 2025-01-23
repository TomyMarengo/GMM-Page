// src/components/Density/DensitySection.tsx

import React, { useState } from 'react';

import DensityChart from '@/components/Density/DensityChart';

interface DensitySectionProps {
  title: string;
  defaultConfig: {
    nComponents: number;
    covarianceType: string;
    randomState: number;
    dataset: string;
  };
}

const DensitySection: React.FC<DensitySectionProps> = ({
  title,
  defaultConfig,
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

  return (
    <div className="p-4 border border-gray-300 rounded">
      <h2 className="text-2xl font-bold mb-4">{title}</h2>

      {/* Formulario de configuración */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className="block font-semibold mb-2">N° Componentes:</label>
          <input
            type="number"
            value={nComponents}
            onChange={(e) => setNComponents(parseInt(e.target.value, 10))}
            min={1}
            className="block w-full p-2 border border-gray-300 rounded"
          />
        </div>

        <div>
          <label className="block font-semibold mb-2">
            Tipo de Covarianza:
          </label>
          <select
            value={covarianceType}
            onChange={(e) => setCovarianceType(e.target.value)}
            className="block w-full p-2 border border-gray-300 rounded"
          >
            <option value="full">Full</option>
            <option value="diag">Diagonal</option>
            <option value="tied">Tied</option>
            <option value="spherical">Spherical</option>
          </select>
        </div>

        <div>
          <label className="block font-semibold mb-2">Semilla:</label>
          <input
            type="number"
            value={randomState}
            onChange={(e) => setRandomState(parseInt(e.target.value, 10))}
            className="block w-full p-2 border border-gray-300 rounded"
          />
        </div>
      </div>

      {/* Gráfico de densidad */}
      <DensityChart
        nComponents={nComponents}
        covarianceType={covarianceType}
        randomState={randomState}
        dataset={defaultConfig.dataset}
      />
    </div>
  );
};

export default DensitySection;
