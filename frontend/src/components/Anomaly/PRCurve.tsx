// src/components/Anomaly/PRCurve.tsx
import * as d3 from 'd3';
import React, { useEffect, useRef } from 'react';

import { PRPoint } from '@/utils/metrics';

interface PRCurveProps {
  data: PRPoint[];
}

const PRCurve: React.FC<PRCurveProps> = ({ data }) => {
  const svgRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    if (data.length === 0 || !svgRef.current) return;

    const width = 400;
    const height = 300;
    const margin = { top: 20, right: 30, bottom: 40, left: 50 };

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    svg
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');

    const xScale = d3
      .scaleLinear()
      .domain([0, 1])
      .range([margin.left, width - margin.right]);
    const yScale = d3
      .scaleLinear()
      .domain([0, 1])
      .range([height - margin.bottom, margin.top]);

    // Ejes
    const xAxis = d3.axisBottom(xScale).ticks(5);
    const yAxis = d3.axisLeft(yScale).ticks(5);

    svg
      .append('g')
      .attr('transform', `translate(0, ${height - margin.bottom})`)
      .call(xAxis)
      .append('text')
      .attr('x', width - margin.right)
      .attr('y', -10)
      .attr('fill', 'black')
      .attr('text-anchor', 'end')
      .text('Recall');

    svg
      .append('g')
      .attr('transform', `translate(${margin.left}, 0)`)
      .call(yAxis)
      .append('text')
      .attr('x', 10)
      .attr('y', margin.top)
      .attr('fill', 'black')
      .attr('text-anchor', 'start')
      .text('Precision');

    // Línea de la Curva PR
    const line = d3
      .line<PRPoint>()
      .x((d) => xScale(d.recall))
      .y((d) => yScale(d.precision))
      .curve(d3.curveMonotoneX);

    svg
      .append('path')
      .datum(data)
      .attr('fill', 'none')
      .attr('stroke', 'green')
      .attr('stroke-width', 2)
      .attr('d', line);
  }, [data]);

  return (
    <div className="p-4 bg-white shadow rounded">
      <h3 className="text-lg font-bold mb-2">Curva Precisión-Recall</h3>
      <svg ref={svgRef} className="w-full h-auto"></svg>
    </div>
  );
};

export default PRCurve;
