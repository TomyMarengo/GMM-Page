// src/pages/SegmentationPage.tsx

import axios from 'axios';
import React, { useEffect, useMemo, useState } from 'react';
import ReactMarkdown from 'react-markdown';

import markdownComponents from '@/components/MarkdownComponents';
import SegmentationSection from '@/components/Segmentation/SegmentationSection';
import {
  segmentationAlgorithmDescription,
  segmentationDatasetDescription,
} from '@/constants';

const SegmentationPage: React.FC = () => {
  const [algorithm1, setAlgorithm1] = useState<'GMM' | 'KMeans'>('GMM');
  const [algorithm2, setAlgorithm2] = useState<'GMM' | 'KMeans'>('KMeans');
  const [selectedDataset, setSelectedDataset] = useState<'MNIST' | 'CIFAR10'>(
    'MNIST'
  );
  const [selectedImage, setSelectedImage] = useState<string | null>(null);

  const [mnistImages, setMnistImages] = useState<
    { url: string; label: number }[]
  >([]);
  const [cifarImages, setCifarImages] = useState<
    { url: string; label: number }[]
  >([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    setIsLoading(true);
    Promise.all([
      axios.get('http://localhost:5000/datasets/mnist'),
      axios.get('http://localhost:5000/datasets/cifar10'),
    ])
      .then(([mnistResponse, cifarResponse]) => {
        setMnistImages(mnistResponse.data);
        setCifarImages(cifarResponse.data);
      })
      .catch((error) => console.error('Error al cargar las imágenes:', error))
      .finally(() => setIsLoading(false));
  }, []);

  const images = selectedDataset === 'MNIST' ? mnistImages : cifarImages;

  const getDefaultConfig = (
    algorithm: 'GMM' | 'KMeans'
  ): {
    algorithm: 'GMM' | 'KMeans';
    nComponents: number;
    resizeShape: [number, number];
    imageUrl: string;
  } => {
    return {
      algorithm,
      nComponents: 3,
      resizeShape: [256, 256],
      imageUrl: selectedImage || '',
    };
  };

  const defaultConfig1 = useMemo(
    () => getDefaultConfig(algorithm1),
    [algorithm1, selectedImage]
  );

  const defaultConfig2 = useMemo(
    () => getDefaultConfig(algorithm2),
    [algorithm2, selectedImage]
  );

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Segmentación de Imágenes</h1>

      {/* Selector de dataset */}
      <div className="mb-4 flex items-center gap-4">
        <label className="text-lg font-semibold">
          Selecciona el conjunto de datos:
        </label>
        <select
          value={selectedDataset}
          onChange={(e) =>
            setSelectedDataset(e.target.value as 'MNIST' | 'CIFAR10')
          }
          className="p-2 border border-gray-300 rounded"
        >
          <option value="MNIST">MNIST</option>
          <option value="CIFAR10">CIFAR-10</option>
        </select>

        <div className="text-gray-700 flex-1">
          <ReactMarkdown components={markdownComponents}>
            {segmentationDatasetDescription[selectedDataset]}
          </ReactMarkdown>
        </div>
      </div>

      {/* Selector de imágenes */}
      {isLoading ? (
        <div className="flex h-screen mt-10 ml-10">
          <p className="text-gray-500 text-2xl">Cargando imágenes...</p>
        </div>
      ) : (
        <div className="mb-4">
          <h2 className="text-lg font-semibold mb-2">Selecciona una imagen:</h2>
          <div className="grid grid-cols-10 gap-4">
            {images.map((image, idx) => (
              <img
                key={idx}
                src={image.url}
                alt={`${selectedDataset} ${idx}`}
                className={`cursor-pointer border-4 rounded-lg ${
                  selectedImage === image.url
                    ? 'border-blue-500'
                    : 'border-transparent'
                }`}
                onClick={() => setSelectedImage(image.url)}
                style={{
                  width: '100px',
                  height: '100px',
                  objectFit: 'contain',
                }}
              />
            ))}
          </div>
        </div>
      )}

      {/* Grid de Configuración */}
      {selectedImage ? (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Configuración 1 */}
            <div className="p-4 border border-gray-300 rounded">
              <h2 className="text-2xl font-bold mb-4">Configuración 1</h2>
              <div className="mb-4 flex items-center gap-2">
                <label className="block mb-2 font-semibold">Algoritmo:</label>
                <select
                  value={algorithm1}
                  onChange={(e) =>
                    setAlgorithm1(e.target.value as 'GMM' | 'KMeans')
                  }
                  className="p-2 border border-gray-300 rounded"
                >
                  <option value="GMM">Gaussian Mixture Model (GMM)</option>
                  <option value="KMeans">KMeans</option>
                </select>
              </div>
              <div className="text-gray-700 flex-1">
                <ReactMarkdown components={markdownComponents}>
                  {segmentationAlgorithmDescription[algorithm1]}
                </ReactMarkdown>
              </div>
            </div>

            {/* Configuración 2 */}
            <div className="p-4 border border-gray-300 rounded">
              <h2 className="text-2xl font-bold mb-4">Configuración 2</h2>
              <div className="mb-4 flex items-center gap-2">
                <label className="block mb-2 font-semibold">Algoritmo:</label>

                <select
                  value={algorithm2}
                  onChange={(e) =>
                    setAlgorithm2(e.target.value as 'GMM' | 'KMeans')
                  }
                  className="p-2 border border-gray-300 rounded"
                >
                  <option value="GMM">Gaussian Mixture Model (GMM)</option>
                  <option value="KMeans">KMeans</option>
                </select>
              </div>
              <div className="text-gray-700 flex-1">
                <ReactMarkdown components={markdownComponents}>
                  {segmentationAlgorithmDescription[algorithm2]}
                </ReactMarkdown>
              </div>
            </div>
          </div>

          {/* Grid de Segmentación */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
            {/* Segmentación 1 */}
            <SegmentationSection
              title={`Segmentación con ${algorithm1}`}
              defaultConfig={defaultConfig1}
            />

            {/* Segmentación 2 */}
            <SegmentationSection
              title={`Segmentación con ${algorithm2}`}
              defaultConfig={defaultConfig2}
            />
          </div>
        </>
      ) : (
        <p className="text-gray-500">
          Selecciona una imagen para comenzar la segmentación.
        </p>
      )}
    </div>
  );
};

export default SegmentationPage;
