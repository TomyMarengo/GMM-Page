// src/components/KMeansClusteringChart.tsx
import * as d3 from 'd3';
import React, { useEffect, useRef } from 'react';

import { useFetchKMeansClusteringMutation } from '@/services/clusteringApi';

interface KMeansClusteringChartProps {
  nClusters: number;
  nCenters: number;
  randomState: number;
  nSamples: number;
  nFeatures: number;
}

const KMeansClusteringChart: React.FC<KMeansClusteringChartProps> = ({
  nClusters,
  nCenters,
  randomState,
  nSamples,
  nFeatures,
}) => {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [fetchKMeansClustering, { data, isLoading, isError }] =
    useFetchKMeansClusteringMutation();

  useEffect(() => {
    fetchKMeansClustering({
      n_clusters: nClusters,
      n_centers: nCenters,
      random_state: randomState,
      n_samples: nSamples,
      n_features: nFeatures,
    });
  }, [
    fetchKMeansClustering,
    nClusters,
    nCenters,
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
        cluster: data.clusters[i],
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
          tooltip.transition().duration(200).style('opacity', 0.9);
          tooltip
            .html(`Cluster: ${d.cluster}`)
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

      if (data.centers) {
        svg
          .selectAll('rect')
          .data(data.centers)
          .enter()
          .append('rect')
          .attr('x', (center: number[]) => xScale(center[0]) - 6)
          .attr('y', (center: number[]) => yScale(center[1]) - 6)
          .attr('width', 12)
          .attr('height', 12)
          .attr('fill', (d, i) => colorScale(String(i)))
          .attr('stroke', '#000')
          .attr('stroke-width', 1.5)
          .style('pointer-events', 'none');
      }

      return () => {
        tooltip.remove();
      };
    }
  }, [data, nFeatures]);

  return (
    <div className="p-4 bg-white shadow rounded">
      <h2 className="text-xl font-bold mb-2">Clustering con KMeans</h2>
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
            <strong>Inertia:</strong> {data.inertia.toFixed(2)}
          </p>
          <p>
            <strong>Silhouette Score:</strong>{' '}
            {data.silhouette_score?.toFixed(2)}
          </p>
          <p>
            <strong>Calinski-Harabasz:</strong>{' '}
            {data.calinski_harabasz_score?.toFixed(2)}
          </p>
          <p>
            <strong>Davies-Bouldin:</strong>{' '}
            {data.davies_bouldin_score?.toFixed(2)}
          </p>
          <p>
            <strong>Adjusted Rand Index:</strong>{' '}
            {data.adjusted_rand_index?.toFixed(2)}
          </p>
        </div>
      )}
      <svg
        ref={svgRef}
        className="border border-gray-300 place-self-center"
      ></svg>
    </div>
  );
};

export default KMeansClusteringChart;
