// src/components/Anomaly/AnomalyDetectionChart.tsx
import 'react-loading-skeleton/dist/skeleton.css';

import * as d3 from 'd3';
import React, { useEffect, useRef, useState } from 'react';
import Skeleton from 'react-loading-skeleton';

import PRCurve from '@/components/Anomaly/PRCurve';
import ROCCurve from '@/components/Anomaly/ROCCurve';
import { useFetchAnomalyDetectionMutation } from '@/services/anomalyApi';
import {
  calculatePRCurve,
  calculateROCCurve,
  PRPoint,
  ROCPoint,
} from '@/utils/metrics';

interface AnomalyDetectionChartProps {
  algorithm: 'GMM' | 'IsolationForest';
  contamination: number;
  nComponents?: number; // Solo para GMM
  randomState: number;
  nSamples: number;
  nFeatures: number;
  nEstimators?: number; // Solo para Isolation Forest
  maxSamples?: string | number; // Solo para Isolation Forest
}

const AnomalyDetectionChart: React.FC<AnomalyDetectionChartProps> = ({
  algorithm,
  contamination,
  nComponents,
  randomState,
  nSamples,
  nFeatures,
  nEstimators,
  maxSamples,
}) => {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [fetchAnomalyDetection, { data, isLoading, isError }] =
    useFetchAnomalyDetectionMutation();

  // Estados para las curvas ROC y PR
  const [rocData, setRocData] = useState<ROCPoint[]>([]);
  const [prData, setPrData] = useState<PRPoint[]>([]);

  useEffect(() => {
    fetchAnomalyDetection({
      algorithm,
      contamination,
      n_components: nComponents,
      random_state: randomState,
      n_samples: nSamples,
      n_features: nFeatures,
      ...(algorithm === 'IsolationForest' && {
        n_estimators: nEstimators,
        max_samples: maxSamples,
      }),
    });
  }, [
    fetchAnomalyDetection,
    algorithm,
    contamination,
    nComponents,
    randomState,
    nSamples,
    nFeatures,
    nEstimators,
    maxSamples,
  ]);

  useEffect(() => {
    if (data && svgRef.current) {
      const width = 800;
      const height = 600;
      const margin = { top: 50, right: 50, bottom: 70, left: 70 };

      const points = data.data.map((point, i) => ({
        x: point[0],
        y: point[1],
        isAnomaly: data.predictions[i],
        trueLabel: data.labels_true[i] === -1 ? 'Anomalía' : 'Normal',
      }));

      const svg = d3.select(svgRef.current);
      svg.selectAll('*').remove();
      svg
        .attr('viewBox', `0 0 ${width} ${height}`)
        .attr('preserveAspectRatio', 'xMidYMid meet');

      const xExtent = d3.extent(points, (d) => d.x) as [number, number];
      const yExtent = d3.extent(points, (d) => d.y) as [number, number];

      const xScale = d3
        .scaleLinear()
        .domain([xExtent[0], xExtent[1]])
        .range([margin.left, width - margin.right]);
      const yScale = d3
        .scaleLinear()
        .domain([yExtent[0], yExtent[1]])
        .range([height - margin.bottom, margin.top]);

      const tooltip = d3
        .select('body')
        .append('div')
        .style('position', 'absolute')
        .style('background', '#f4f4f4')
        .style('padding', '8px')
        .style('border', '1px solid #333')
        .style('border-radius', '4px')
        .style('pointer-events', 'none')
        .style('opacity', 0);

      // Puntos normales
      svg
        .selectAll('circle.normal')
        .data(points.filter((d) => !d.isAnomaly))
        .enter()
        .append('circle')
        .attr('cx', (d) => xScale(d.x))
        .attr('cy', (d) => yScale(d.y))
        .attr('r', 5)
        .attr('fill', 'blue')
        .on('mouseover', (event, d) => {
          tooltip.transition().duration(200).style('opacity', 0.9);
          tooltip
            .html(`Predicción: Normal<br/>Etiqueta Verdadera: ${d.trueLabel}`)
            .style('left', event.pageX + 10 + 'px')
            .style('top', event.pageY - 28 + 'px');
        })
        .on('mousemove', (event) => {
          tooltip
            .style('left', event.pageX + 10 + 'px')
            .style('top', event.pageY - 28 + 'px');
        })
        .on('mouseout', () => {
          tooltip.transition().duration(500).style('opacity', 0);
        });

      // Puntos anómalos
      svg
        .selectAll('circle.anomaly')
        .data(points.filter((d) => d.isAnomaly))
        .enter()
        .append('circle')
        .attr('cx', (d) => xScale(d.x))
        .attr('cy', (d) => yScale(d.y))
        .attr('r', 5)
        .attr('fill', 'red')
        .on('mouseover', (event, d) => {
          tooltip.transition().duration(200).style('opacity', 0.9);
          tooltip
            .html(`Predicción: Anomalía<br/>Etiqueta Verdadera: ${d.trueLabel}`)
            .style('left', event.pageX + 10 + 'px')
            .style('top', event.pageY - 28 + 'px');
        })
        .on('mousemove', (event) => {
          tooltip
            .style('left', event.pageX + 10 + 'px')
            .style('top', event.pageY - 28 + 'px');
        })
        .on('mouseout', () => {
          tooltip.transition().duration(500).style('opacity', 0);
        });

      const labelsBoolean = data.labels_true.map((label) => label === -1);
      // Calcular Curvas ROC y PR
      const rocPoints = calculateROCCurve(data.scores, labelsBoolean);
      const prPoints = calculatePRCurve(data.scores, labelsBoolean);

      setRocData(rocPoints);
      setPrData(prPoints);

      return () => {
        tooltip.remove();
      };
    }
  }, [data, nFeatures]);

  return (
    <div className="p-4 bg-white shadow rounded">
      <h2 className="text-xl font-bold mb-2">
        Detección de Anomalías con {algorithm}
      </h2>
      <div
        style={{
          height: '75px',
          columnWidth: '200px',
          columnGap: '1rem',
          overflowY: 'auto',
        }}
      >
        {isLoading ? (
          <>
            <Skeleton height={40} width={200} count={4} />
          </>
        ) : isError ? (
          <p className="text-red-500">Error al cargar los datos.</p>
        ) : data ? (
          <>
            <p>
              <strong>Precisión:</strong> {data.metrics.precision.toFixed(2)}
            </p>
            <p>
              <strong>Recall:</strong> {data.metrics.recall.toFixed(2)}
            </p>
            <p>
              <strong>F1 Score:</strong> {data.metrics.f1_score.toFixed(2)}
            </p>
            {data.metrics.roc_auc !== null && (
              <p>
                <strong>ROC AUC:</strong> {data.metrics.roc_auc.toFixed(2)}
              </p>
            )}
          </>
        ) : null}
      </div>
      {/* Sección del Gráfico */}
      <div className="w-full">
        {isLoading ? (
          <div className="w-full h-full flex items-center justify-center border border-gray-300">
            <Skeleton height={300} width="100%" />
          </div>
        ) : isError ? (
          <p className="text-red-500">Error al cargar el gráfico.</p>
        ) : (
          <svg
            ref={svgRef}
            className="w-full h-full border border-gray-300"
          ></svg>
        )}
      </div>
      {/* Gráficos de Curva ROC y PR */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
        {isLoading ? (
          <Skeleton height={150} width="100%" />
        ) : (
          <ROCCurve data={rocData} />
        )}
        {isLoading ? (
          <Skeleton height={150} width="100%" />
        ) : (
          <PRCurve data={prData} />
        )}
      </div>
    </div>
  );
};

export default AnomalyDetectionChart;
