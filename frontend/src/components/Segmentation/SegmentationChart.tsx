// src/components/Segmentation/SegmentationChart.tsx

import React, { useEffect, useMemo, useRef, useState } from 'react';

import { useFetchSegmentationMutation } from '@/services/slices/segmentationApiSlice';

interface SegmentationChartProps {
  algorithm: 'GMM' | 'KMeans';
  nComponents: number;
  resizeShape: [number, number];
  imageUrl: string | null; // Puede ser null si se sube una imagen
  imageFile?: File | null; // Opcional, solo si se sube una imagen
}

const SegmentationChart: React.FC<SegmentationChartProps> = ({
  algorithm,
  nComponents,
  resizeShape,
  imageUrl,
  imageFile,
}) => {
  const [fetchSegmentation, { data, isLoading, isError }] =
    useFetchSegmentationMutation();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hoverInfo, setHoverInfo] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<string | null>(null);
  const [originalImageUrl, setOriginalImageUrl] = useState<string | null>(null);

  const clusterColors = useMemo(
    () =>
      data?.cluster_centers.map((_, index) => {
        const hue = (index * 360) / data.cluster_centers.length;
        return [
          Math.round(((hue % 360) * 255) / 360), // R
          Math.round((((hue + 120) % 360) * 255) / 360), // G
          Math.round((((hue + 240) % 360) * 255) / 360), // B
        ];
      }),
    [data]
  );

  useEffect(() => {
    if (!imageUrl && !imageFile) return;

    const fetchData = async () => {
      try {
        const response = await fetchSegmentation({
          algorithm,
          n_components: nComponents,
          resize_shape: resizeShape,
          imageUrl: imageFile ? undefined : imageUrl || '',
          imageFile: imageFile || undefined,
        });

        if (response.data) {
          const clusterCounts: { [key: number]: number } = response.data.labels
            .flat()
            .reduce((acc: { [key: number]: number }, label: number) => {
              acc[label] = (acc[label] || 0) + 1;
              return acc;
            }, {});

          const totalPixels = resizeShape[0] * resizeShape[1];
          const clusterMetrics = Object.entries(clusterCounts)
            .map(
              ([label, count]) =>
                `Cluster ${label}: ${((count / totalPixels) * 100).toFixed(2)}%`
            )
            .join(' | ');

          setMetrics(clusterMetrics);
        }
      } catch (error) {
        console.error('Error al realizar el fetch de segmentación:', error);
      }
    };

    fetchData();
  }, [
    algorithm,
    nComponents,
    resizeShape,
    imageUrl,
    imageFile,
    fetchSegmentation,
  ]);

  useEffect(() => {
    if (!data || !canvasRef.current || !clusterColors) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = resizeShape[0];
    const height = resizeShape[1];
    canvas.width = width;
    canvas.height = height;

    const imageData = ctx.createImageData(width, height);

    // Crear mapa de colores para cada píxel
    data.labels.forEach((row: number[], y: number) => {
      row.forEach((label: number, x: number) => {
        const index = (y * width + x) * 4;
        const [r, g, b] = clusterColors[label];
        imageData.data[index] = r; // R
        imageData.data[index + 1] = g; // G
        imageData.data[index + 2] = b; // B
        imageData.data[index + 3] = 255; // Alpha
      });
    });

    ctx.putImageData(imageData, 0, 0);
  }, [data, resizeShape, clusterColors]);

  const handleMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !data || !data.labels) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mouseX = Math.floor((event.clientX - rect.left) * scaleX);
    const mouseY = Math.floor((event.clientY - rect.top) * scaleY);

    if (
      mouseX >= 0 &&
      mouseX < resizeShape[0] &&
      mouseY >= 0 &&
      mouseY < resizeShape[1]
    ) {
      const label = data.labels[mouseY][mouseX];
      setHoverInfo(`Posición: (${mouseX}, ${mouseY}), Cluster: ${label}`);
    } else {
      setHoverInfo(null);
    }
  };

  // Obtener la URL de la imagen original
  useEffect(() => {
    if (imageFile) {
      const objectUrl = URL.createObjectURL(imageFile);
      setOriginalImageUrl(objectUrl);

      return () => {
        URL.revokeObjectURL(objectUrl);
        setOriginalImageUrl(null);
      };
    } else {
      setOriginalImageUrl(imageUrl);
    }
  }, [imageUrl, imageFile]);

  return (
    <div className="flex flex-col md:flex-row md:space-x-4 gap-4">
      {/* Imagen Segmentada */}
      <div className="w-full relative">
        <h3 className="text-lg font-semibold mb-2">Imagen Segmentada</h3>
        {isLoading ? (
          <p className="text-gray-500">Procesando...</p>
        ) : isError ? (
          <p className="text-red-500">Error al segmentar la imagen.</p>
        ) : (
          <>
            <canvas
              ref={canvasRef}
              className="border w-full h-auto"
              onMouseMove={handleMouseMove}
              width={resizeShape[0]}
              height={resizeShape[1]}
              style={{
                display: 'block',
                maxWidth: '100%',
                height: 'auto',
              }}
            />
            {hoverInfo && (
              <div className="absolute top-0 right-0 bg-black text-white text-sm p-2 rounded">
                {hoverInfo}
              </div>
            )}
            {metrics && (
              <div className="absolute -bottom-5 left-0 bg-gray-800 text-white text-sm p-2 rounded">
                <h4 className="font-bold">Distribución de Clusters:</h4>
                <p>{metrics}</p>
              </div>
            )}
          </>
        )}
      </div>

      {/* Imagen Original */}
      <div className="w-full">
        <h3 className="text-lg font-semibold mb-2">Imagen Original</h3>
        {originalImageUrl ? (
          <img
            src={originalImageUrl}
            alt="Original"
            className="border w-full h-auto"
            style={{
              display: 'block',
              maxWidth: '100%',
              height: 'auto',
            }}
          />
        ) : (
          <p className="text-gray-500">No hay imagen seleccionada.</p>
        )}
      </div>
    </div>
  );
};

export default SegmentationChart;
