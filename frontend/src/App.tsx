// src/App.tsx

import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';

import Sidebar from '@/components/Sidebar';
import AnomalyPage from '@/pages/AnomalyPage';
import ClusteringPage from '@/pages/ClusteringPage';
import DensityPage from '@/pages/DensityPage';
import HomePage from '@/pages/HomePage';
import SegmentationPage from '@/pages/SegmentationPage';

const App: React.FC = () => {
  const menuItems = [
    { key: 'home', label: 'Inicio', path: '/' },
    { key: 'clustering', label: 'Clustering', path: '/clustering' },
    { key: 'anomalies', label: 'Detección de Anomalías', path: '/anomalies' },
    { key: 'density', label: 'Modelado de Densidad', path: '/density' },
    {
      key: 'segmentation',
      label: 'Segmentación de Imágenes',
      path: '/segmentation',
    },
  ];

  const renderContent = () => {
    return (
      <Routes>
        <Route path="/" element={<HomePage />} />{' '}
        <Route path="/clustering" element={<ClusteringPage />} />
        <Route path="/anomalies" element={<AnomalyPage />} />
        <Route path="/density" element={<DensityPage />} />
        <Route path="/segmentation" element={<SegmentationPage />} />
        <Route path="*" element={<div>Pagina no encontrada</div>} />
      </Routes>
    );
  };

  return (
    <Router>
      <Sidebar items={menuItems} />
      <div className="ml-60 p-4 h-screen overflow-auto transition-all duration-300">
        {renderContent()}
      </div>
    </Router>
  );
};

export default App;
