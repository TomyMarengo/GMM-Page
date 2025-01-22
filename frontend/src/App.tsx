// src/App.tsx

import React, { useState } from 'react';

import Sidebar from '@/components/Sidebar';
import ClusteringPage from '@/pages/ClusteringPage';

const App: React.FC = () => {
  const [activeView, setActiveView] = useState<string>('clustering');

  const menuItems = [
    { key: 'clustering', label: 'Clustering' },
    { key: 'anomalies', label: 'Detección de anomalías' },
    { key: 'density', label: 'Modelado de densidad' },
    { key: 'segmentation', label: 'Segmentación de imágenes' },
    { key: 'patterns', label: 'Reconocimiento de patrones' },
  ];

  const renderContent = () => {
    switch (activeView) {
      case 'clustering':
        return <ClusteringPage />;
      // Agrega las otras vistas conforme se implementen.
      default:
        return <div>Selecciona una opción en el menú</div>;
    }
  };

  return (
    <div className="flex">
      <Sidebar
        items={menuItems}
        activeKey={activeView}
        onSelect={setActiveView}
      />

      <div className="ml-12 p-4 w-full transition-all duration-300">
        {renderContent()}
      </div>
    </div>
  );
};

export default App;
