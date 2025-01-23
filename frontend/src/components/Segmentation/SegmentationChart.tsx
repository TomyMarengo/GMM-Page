import React, { useEffect, useMemo, useRef, useState } from 'react';

import { useFetchSegmentationMutation } from '@/services/segmentationApi';

interface SegmentationChartProps {
  algorithm: 'GMM' | 'KMeans';
  nComponents: number;
  resizeShape: [number, number];
  imageUrl: string;
}

const SegmentationChart: React.FC<SegmentationChartProps> = ({
  algorithm,
  nComponents,
  resizeShape,
  imageUrl,
}) => {
  const [fetchSegmentation, { data, isLoading, isError }] =
    useFetchSegmentationMutation();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hoverInfo, setHoverInfo] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<string | null>(null);

  const clusterColors = useMemo(
    () =>
      data?.cluster_centers.map((_, index) => {
        const hue = (index * 360) / data.cluster_centers.length;
        return [
          Math.round(hue), // R
          Math.round((hue + 120) % 255), // G
          Math.round((hue + 240) % 255), // B
        ];
      }),
    [data]
  );

  useEffect(() => {
    if (!imageUrl) return;

    const fetchData = async () => {
      try {
        const response = await fetchSegmentation({
          algorithm,
          n_components: nComponents,
          resize_shape: resizeShape,
          image_url: imageUrl,
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
  }, [algorithm, nComponents, resizeShape, imageUrl, fetchSegmentation]);

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
    data.labels.forEach((row, y) => {
      row.forEach((label, x) => {
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

  return (
    <div className="grid grid-cols-2 gap-4 relative">
      <div>
        <h3 className="text-lg font-semibold mb-2">Imagen Segmentada</h3>
        {isLoading ? (
          <p className="text-gray-500">Procesando...</p>
        ) : isError ? (
          <p className="text-red-500">Error al segmentar la imagen.</p>
        ) : (
          <>
            <canvas
              ref={canvasRef}
              className="w-full border"
              onMouseMove={handleMouseMove}
            />
            {hoverInfo && (
              <div className="absolute top-2 left-2 bg-black text-white text-sm p-2 rounded">
                {hoverInfo}
              </div>
            )}
            {metrics && (
              <div className="absolute bottom-2 left-2 bg-gray-800 text-white text-sm p-2 rounded">
                <h4 className="font-bold">Distribución de Clusters:</h4>
                <p>{metrics}</p>
              </div>
            )}
          </>
        )}
      </div>
      <div>
        <h3 className="text-lg font-semibold mb-2">Imagen Original</h3>
        <img src={imageUrl} alt="Original" className="w-full border" />
      </div>
    </div>
  );
};

export default SegmentationChart;
