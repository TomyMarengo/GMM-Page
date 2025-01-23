import 'react-loading-skeleton/dist/skeleton.css';

import * as d3 from 'd3';
import React, { useEffect, useRef } from 'react';
import Skeleton from 'react-loading-skeleton';

import { useFetchDensityMutation } from '@/services/densityApi';

interface DensityChartProps {
  nComponents: number;
  covarianceType: string;
  randomState: number;
  dataset: string;
}

const DensityChart: React.FC<DensityChartProps> = ({
  nComponents,
  covarianceType,
  randomState,
  dataset,
}) => {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [fetchDensity, { data, isLoading, isError }] =
    useFetchDensityMutation();

  useEffect(() => {
    fetchDensity({
      n_components: nComponents,
      covariance_type: covarianceType,
      random_state: randomState,
      dataset,
    });
  }, [fetchDensity, nComponents, covarianceType, randomState, dataset]);

  useEffect(() => {
    if (data && svgRef.current) {
      const width = 800;
      const height = 600;
      const margin = { top: 50, right: 50, bottom: 70, left: 70 };

      const points = data.data.map((point: number[], i: number) => ({
        x: point[0],
        y: point[1],
        density: data.density[i],
        probabilities: data.probabilities[i],
        component: data.probabilities[i].indexOf(
          Math.max(...data.probabilities[i])
        ),
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

      const colorScale = d3
        .scaleSequential(d3.interpolateViridis)
        .domain(d3.extent(points, (d) => d.density) as [number, number]);

      // Formas distintas para cada componente
      const shapes = [
        d3.symbolCircle,
        d3.symbolSquare,
        d3.symbolTriangle,
        d3.symbolDiamond,
        d3.symbolCross,
        d3.symbolStar,
        d3.symbolWye,
      ];

      const symbol = d3.symbol();

      // Crear tooltip
      const tooltip = d3
        .select('body')
        .append('div')
        .style('position', 'absolute')
        .style('background', '#f9f9f9')
        .style('border', '1px solid #ccc')
        .style('border-radius', '4px')
        .style('padding', '10px')
        .style('pointer-events', 'none')
        .style('opacity', 0);

      // Dibujar puntos con formas correspondientes al componente y tooltips
      svg
        .selectAll('path')
        .data(points)
        .enter()
        .append('path')
        .attr('d', (d) => {
          const componentShape = shapes[d.component % shapes.length];
          symbol.type(componentShape);
          symbol.size(100);
          return symbol();
        })
        .attr('transform', (d) => `translate(${xScale(d.x)},${yScale(d.y)})`)
        .attr('fill', (d) => colorScale(d.density))
        .on('mouseover', (event, d) => {
          const probabilitiesText = d.probabilities
            .map(
              (prob: number, idx: number) =>
                `Componente ${idx + 1}: ${(prob * 100).toFixed(2)}%`
            )
            .join('<br>');

          tooltip
            .style('opacity', 1)
            .html(
              `<strong>Coordenadas:</strong> (${d.x.toFixed(2)}, ${d.y.toFixed(
                2
              )})<br>
               <strong>Densidad:</strong> ${d.density.toFixed(4)}<br>
               <strong>Probabilidades:</strong><br>${probabilitiesText}`
            )
            .style('left', event.pageX + 10 + 'px')
            .style('top', event.pageY - 28 + 'px');
        })
        .on('mousemove', (event) => {
          tooltip
            .style('left', event.pageX + 10 + 'px')
            .style('top', event.pageY - 28 + 'px');
        })
        .on('mouseout', () => {
          tooltip.style('opacity', 0);
        });

      svg
        .append('g')
        .attr('transform', `translate(0,${height - margin.bottom})`)
        .call(d3.axisBottom(xScale));

      svg
        .append('g')
        .attr('transform', `translate(${margin.left},0)`)
        .call(d3.axisLeft(yScale));
    }
  }, [data]);

  return (
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
  );
};

export default DensityChart;
