// src/pages/DensityPage.tsx

import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';

import DensitySection from '@/components/Density/DensitySection';
import markdownComponents from '@/components/MarkdownComponents';
import { DensityDataset, densityDescriptions } from '@/constants';

const DensityPage: React.FC = () => {
  const [dataset1, setDataset1] = useState<DensityDataset>('iris');
  const [dataset2, setDataset2] = useState<DensityDataset>('housing');

  // Configuraciones por defecto para cada dataset
  const getDefaultConfig = (dataset: DensityDataset) => ({
    nComponents: 3,
    covarianceType: 'full',
    randomState: 42,
    dataset,
  });

  return (
    <div className="p-4">
      {/* Grid de Descripciones */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Configuración 1 - Descripción */}
        <div className="p-4 border border-gray-300 rounded flex flex-col">
          <h2 className="text-2xl font-bold mb-4">Configuración 1</h2>
          <div className="mb-4 flex items-center gap-2">
            <label className="block mb-1 font-semibold">Dataset:</label>
            <select
              value={dataset1}
              onChange={(e) => setDataset1(e.target.value as DensityDataset)}
              className="p-2 border border-gray-300 rounded"
            >
              <option value="iris">Iris</option>
              <option value="housing">California Housing</option>
            </select>
          </div>

          {/* Descripción Dinámica */}
          <div className="text-gray-700 flex-1">
            <ReactMarkdown components={markdownComponents}>
              {densityDescriptions[dataset1]}
            </ReactMarkdown>
          </div>
        </div>

        {/* Configuración 2 - Descripción */}
        <div className="p-4 border border-gray-300 rounded flex flex-col">
          <h2 className="text-2xl font-bold mb-4">Configuración 2</h2>
          <div className="mb-4 flex items-center gap-2">
            <label className="block mb-1 font-semibold">Dataset:</label>
            <select
              value={dataset2}
              onChange={(e) => setDataset2(e.target.value as DensityDataset)}
              className="p-2 border border-gray-300 rounded"
            >
              <option value="iris">Iris</option>
              <option value="housing">California Housing</option>
            </select>
          </div>

          {/* Descripción Dinámica */}
          <div className="text-gray-700 flex-1">
            <ReactMarkdown components={markdownComponents}>
              {densityDescriptions[dataset2]}
            </ReactMarkdown>
          </div>
        </div>
      </div>

      {/* Grid de Modelado de Densidad */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
        {/* Configuración 1 - Densidad */}
        <div className="flex flex-col">
          <DensitySection
            title={`Modelado de Densidad - ${dataset1}`}
            defaultConfig={getDefaultConfig(dataset1)}
          />
        </div>

        {/* Configuración 2 - Densidad */}
        <div className="flex flex-col">
          <DensitySection
            title={`Modelado de Densidad - ${dataset2}`}
            defaultConfig={getDefaultConfig(dataset2)}
          />
        </div>
      </div>
    </div>
  );
};

export default DensityPage;
