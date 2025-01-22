// src/components/AnomalyDetectionChart.tsx
import * as d3 from 'd3';
import React, { useEffect, useRef } from 'react';

import { useFetchAnomalyDetectionMutation } from '@/services/anomalyApi';

interface AnomalyDetectionChartProps {
  algorithm: 'GMM' | 'IsolationForest';
  contamination: number;
  nComponents?: number; // Solo para GMM
  randomState: number;
  nSamples: number;
  nFeatures: number;
}

const AnomalyDetectionChart: React.FC<AnomalyDetectionChartProps> = ({
  algorithm,
  contamination,
  nComponents,
  randomState,
  nSamples,
  nFeatures,
}) => {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [fetchAnomalyDetection, { data, isLoading, isError }] =
    useFetchAnomalyDetectionMutation();

  useEffect(() => {
    fetchAnomalyDetection({
      algorithm,
      contamination,
      n_components: nComponents,
      random_state: randomState,
      n_samples: nSamples,
      n_features: nFeatures,
    });
  }, [
    fetchAnomalyDetection,
    algorithm,
    contamination,
    nComponents,
    randomState,
    nSamples,
    nFeatures,
  ]);

  useEffect(() => {
    if (data && svgRef.current && nFeatures === 2) {
      const width = 800;
      const height = 600;
      const margin = { top: 50, right: 50, bottom: 70, left: 70 };

      const points = data.data.map((point, i) => ({
        x: point[0],
        y: point[1],
        isAnomaly: data.predictions[i],
      }));

      const svg = d3.select(svgRef.current);
      svg.selectAll('*').remove();
      svg.attr('width', width).attr('height', height);

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
            .html(`Normal`)
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
            .html(`Anomalía`)
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
      {isLoading && (
        <p className="text-blue-500">Cargando datos del modelo...</p>
      )}
      {isError && <p className="text-red-500">Error al cargar los datos.</p>}
      {data && (
        <div
          className="mb-4"
          style={{
            height: '120px',
            columnCount: 2,
            columnGap: '1rem',
            overflowY: 'auto',
          }}
        >
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
        </div>
      )}
      <svg
        ref={svgRef}
        className="border border-gray-300 place-self-center"
      ></svg>
    </div>
  );
};

export default AnomalyDetectionChart;
