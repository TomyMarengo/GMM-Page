// src/components/Segmentation/SegmentationSection.tsx

import React, { useState } from 'react';

import SegmentationChart from '@/components/Segmentation/SegmentationChart';

interface SegmentationSectionProps {
  title: string;
  defaultConfig: {
    algorithm: 'GMM' | 'KMeans';
    nComponents: number;
    resizeShape: [number, number];
    imageUrl: string | null;
    imageFile?: File | null;
  };
}

const SegmentationSection: React.FC<SegmentationSectionProps> = ({
  title,
  defaultConfig,
}) => {
  const [nComponents, setNComponents] = useState<number>(
    defaultConfig.nComponents
  );

  return (
    <div className="p-4 border border-gray-300 rounded">
      <h2 className="text-2xl font-bold mb-4">{title}</h2>

      {/* Formulario de configuración */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="flex flex-col">
          <label className="mb-1">Número de clusters:</label>
          <input
            type="number"
            value={nComponents}
            onChange={(e) => setNComponents(parseInt(e.target.value, 10))}
            min={1}
            className="p-2 border border-gray-300 rounded"
          />
        </div>
      </div>

      {/* Gráfico de segmentación */}
      <SegmentationChart
        algorithm={defaultConfig.algorithm}
        nComponents={nComponents}
        resizeShape={defaultConfig.resizeShape}
        imageUrl={defaultConfig.imageUrl}
        imageFile={defaultConfig.imageFile}
      />
    </div>
  );
};

export default SegmentationSection;
