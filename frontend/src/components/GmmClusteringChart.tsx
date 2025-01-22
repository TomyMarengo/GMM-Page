// src/components/GmmClusteringChart.tsx
import * as d3 from 'd3';
import * as numeric from 'numeric';
import React, { useEffect, useRef } from 'react';

import { useFetchGmmClusteringMutation } from '@/services/api';

interface GmmClusteringChartProps {
  nComponents: number;
  covarianceType: string;
  randomState: number;
  nSamples: number;
  nFeatures: number;
}

const GmmClusteringChart: React.FC<GmmClusteringChartProps> = ({
  nComponents,
  covarianceType,
  randomState,
  nSamples,
  nFeatures,
}) => {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [fetchGmmClustering, { data, isLoading, isError }] =
    useFetchGmmClusteringMutation();

  // Llamar al backend cuando los parámetros cambien
  useEffect(() => {
    fetchGmmClustering({
      n_components: nComponents,
      covariance_type: covarianceType,
      random_state: randomState,
      n_samples: nSamples,
      n_features: nFeatures,
    });
  }, [
    fetchGmmClustering,
    nComponents,
    covarianceType,
    randomState,
    nSamples,
    nFeatures,
  ]);

  useEffect(() => {
    if (data && svgRef.current && nFeatures === 2) {
      // Dimensiones del gráfico
      const width = 800;
      const height = 600;
      const margin = { top: 50, right: 50, bottom: 70, left: 70 };

      // Mapear los datos
      const points = data.data.map((point, i) => ({
        x: point[0],
        y: point[1],
        cluster: data.clusters[i],
      }));

      // Crear o limpiar el SVG
      const svg = d3.select(svgRef.current);
      svg.selectAll('*').remove();
      svg.attr('width', width).attr('height', height);

      // Escalas
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

      // Crear tooltip
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

      // Dibujar puntos
      const colorScale = d3.scaleOrdinal(d3.schemeCategory10);
      svg
        .selectAll('circle')
        .data(points)
        .enter()
        .append('circle')
        .attr('cx', (d) => xScale(d.x))
        .attr('cy', (d) => yScale(d.y))
        .attr('r', 5)
        .attr('fill', (d) => colorScale(String(d.cluster)))
        .on('mouseover', (event, d) => {
          // Mostrar el tooltip con las probabilidades
          const probabilities = data.probabilities[points.indexOf(d)];
          const probText = probabilities
            .map(
              (p: number, idx: number) =>
                `Cluster ${idx}: ${(p * 100).toFixed(2)}%`
            )
            .join('<br>');

          tooltip.transition().duration(200).style('opacity', 0.9);
          tooltip
            .html(`<strong>Probabilidades:</strong><br>${probText}`)
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

      // Dibujar elipses para cada cluster
      data.means.forEach((mean, idx) => {
        const cov = data.covariances[idx];

        // (Opcional) Verificar que sea cuadrada
        if (cov.length !== cov[0].length) {
          console.error(`La matriz de covarianza no es cuadrada:`, cov);
          return; // Saltar esta matriz
        }

        const eig = numeric.eig(cov);
        const eigenValues = eig.lambda.x as number[];
        const eigenVectors = eig.E.x as number[][];

        const angle = Math.atan2(eigenVectors[0][1], eigenVectors[0][0]);
        const rx = Math.sqrt(eigenValues[0]);
        const ry = Math.sqrt(eigenValues[1]);

        svg
          .append('ellipse')
          .attr('cx', xScale(mean[0]))
          .attr('cy', yScale(mean[1]))
          .attr('rx', rx * xScale(1))
          .attr('ry', ry * yScale(1))
          .attr(
            'transform',
            `rotate(${(angle * 180) / Math.PI}, ${xScale(mean[0])}, ${yScale(mean[1])})`
          )
          .style('fill', colorScale(String(idx)))
          .style('fill-opacity', 0.2)
          .style('stroke', colorScale(String(idx)))
          .style('stroke-width', 1.5);
      });

      // Cleanup del tooltip al desmontar el componente
      return () => {
        tooltip.remove();
      };
    }
  }, [data]);

  return (
    <div className="p-4 bg-white shadow rounded">
      <h2 className="text-xl font-bold mb-2">
        Clustering con Gaussian Mixture Model
      </h2>
      {isLoading && (
        <p className="text-blue-500">Cargando datos del modelo...</p>
      )}
      {isError && <p className="text-red-500">Error al cargar los datos.</p>}

      {data && (
        <div className="mb-4">
          <p>
            <strong>Log-Likelihood:</strong> {data.score.toFixed(2)}
          </p>
          <p>
            <strong>BIC:</strong> {data.bic}
          </p>
          <p>
            <strong>AIC:</strong> {data.aic}
          </p>
          <p>
            <strong>Iteraciones:</strong> {data.n_iter}
          </p>
          <p>
            <strong>¿Convergió?</strong> {data.converged ? 'Sí' : 'No'}
          </p>
        </div>
      )}

      <svg ref={svgRef} className="border border-gray-300"></svg>
    </div>
  );
};

export default GmmClusteringChart;
