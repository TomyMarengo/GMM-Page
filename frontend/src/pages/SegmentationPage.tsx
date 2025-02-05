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
  const [selectedDataset, setSelectedDataset] = useState<
    'MNIST' | 'CIFAR10' | 'Upload'
  >('MNIST');
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [uploadedImageFile, setUploadedImageFile] = useState<File | null>(null);
  const [imageDimensions, setImageDimensions] = useState<[number, number]>([
    256, 256,
  ]);

  const [mnistImages, setMnistImages] = useState<
    { url: string; label: number }[]
  >([]);
  const [cifarImages, setCifarImages] = useState<
    { url: string; label: number }[]
  >([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  const API_URL = process.env.REACT_APP_API_URL;

  useEffect(() => {
    if (selectedDataset === 'Upload') {
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    Promise.all([
      axios.get(`${API_URL}/datasets/mnist`),
      axios.get(`${API_URL}/datasets/cifar10`),
    ])
      .then(([mnistResponse, cifarResponse]) => {
        setMnistImages(mnistResponse.data);
        setCifarImages(cifarResponse.data);
      })
      .catch((error) => console.error('Error al cargar las imágenes:', error))
      .finally(() => setIsLoading(false));
  }, [selectedDataset]);

  // Para resetear estados al cambiar de dataset
  useEffect(() => {
    if (selectedDataset !== 'Upload') {
      setUploadedImageFile(null);
      setSelectedImage(null);
    }
  }, [selectedDataset]);

  const images =
    selectedDataset === 'MNIST'
      ? mnistImages
      : selectedDataset === 'CIFAR10'
        ? cifarImages
        : [];

  const getDefaultConfig = (
    algorithm: 'GMM' | 'KMeans'
  ): {
    algorithm: 'GMM' | 'KMeans';
    nComponents: number;
    resizeShape: [number, number];
    imageUrl: string | null;
    imageFile?: File | null;
  } => {
    return {
      algorithm,
      nComponents: 3,
      resizeShape: imageDimensions,
      imageUrl:
        selectedDataset === 'Upload'
          ? selectedImage || null
          : selectedImage || null,
      imageFile:
        selectedDataset === 'Upload' ? uploadedImageFile || null : null,
    };
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setUploadedImageFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        const result = reader.result as string;
        setSelectedImage(result);

        // Cargar la imagen para obtener sus dimensiones
        const img = new Image();
        img.onload = () => {
          const maxDimension = 512; // Define una dimensión máxima
          let width = img.naturalWidth;
          let height = img.naturalHeight;

          if (width > height) {
            if (width > maxDimension) {
              height = Math.round((height * maxDimension) / width);
              width = maxDimension;
            }
          } else {
            if (height > maxDimension) {
              width = Math.round((width * maxDimension) / height);
              height = maxDimension;
            }
          }

          setImageDimensions([width, height]);
        };
        img.src = result;
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDatasetImageSelect = (imageUrl: string) => {
    setSelectedImage(imageUrl);
    setUploadedImageFile(null);

    // Cargar la imagen para obtener sus dimensiones
    const img = new Image();
    img.onload = () => {
      const maxDimension = 512; // Define una dimensión máxima
      let width = img.naturalWidth;
      let height = img.naturalHeight;

      if (width > height) {
        if (width > maxDimension) {
          height = Math.round((height * maxDimension) / width);
          width = maxDimension;
        }
      } else {
        if (height > maxDimension) {
          width = Math.round((width * maxDimension) / height);
          height = maxDimension;
        }
      }

      setImageDimensions([width, height]);
    };
    img.src = imageUrl;
  };

  const defaultConfig1Memo = useMemo(
    () => getDefaultConfig(algorithm1),
    [
      algorithm1,
      selectedImage,
      uploadedImageFile,
      selectedDataset,
      imageDimensions,
    ]
  );

  const defaultConfig2Memo = useMemo(
    () => getDefaultConfig(algorithm2),
    [
      algorithm2,
      selectedImage,
      uploadedImageFile,
      selectedDataset,
      imageDimensions,
    ]
  );

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Segmentación de Imágenes</h1>

      {/* Selector de dataset */}
      <div className="mb-4 flex flex-col md:flex-row items-start md:items-center gap-4">
        <label className="text-lg font-semibold">
          Selecciona el conjunto de datos:
        </label>
        <select
          value={selectedDataset}
          onChange={(e) =>
            setSelectedDataset(e.target.value as 'MNIST' | 'CIFAR10' | 'Upload')
          }
          className="p-2 border border-gray-300 rounded"
        >
          <option value="MNIST">MNIST</option>
          <option value="CIFAR10">CIFAR-10</option>
          <option value="Upload">Subir Imagen</option>
        </select>

        <div className="text-gray-700 flex-1">
          <ReactMarkdown components={markdownComponents}>
            {segmentationDatasetDescription[selectedDataset]}
          </ReactMarkdown>
        </div>
      </div>

      {/* Selector de imágenes o subida */}
      {selectedDataset === 'Upload' ? (
        <div className="mb-4">
          <h2 className="text-lg font-semibold mb-2">Sube una imagen:</h2>
          <input
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            className="p-2 border border-gray-300 rounded"
          />
        </div>
      ) : isLoading ? (
        <div className="flex justify-center items-center h-64">
          <p className="text-gray-500 text-2xl">Cargando imágenes...</p>
        </div>
      ) : (
        <div className="mb-4">
          <h2 className="text-lg font-semibold mb-2">Selecciona una imagen:</h2>
          <div className="grid grid-cols-5 md:grid-cols-10 gap-4">
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
                onClick={() => handleDatasetImageSelect(image.url)}
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
              defaultConfig={defaultConfig1Memo}
            />

            {/* Segmentación 2 */}
            <SegmentationSection
              title={`Segmentación con ${algorithm2}`}
              defaultConfig={defaultConfig2Memo}
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
