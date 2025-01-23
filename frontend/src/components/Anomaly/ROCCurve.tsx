// src/components/Anomaly/ROCCurve.tsx

import * as d3 from 'd3';
import React, { useEffect, useRef } from 'react';

import { ROCPoint } from '@/utils/metrics';

interface ROCCurveProps {
  data: ROCPoint[];
}

const ROCCurve: React.FC<ROCCurveProps> = ({ data }) => {
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
      .text('False Positive Rate');

    svg
      .append('g')
      .attr('transform', `translate(${margin.left}, 0)`)
      .call(yAxis)
      .append('text')
      .attr('x', 10)
      .attr('y', margin.top)
      .attr('fill', 'black')
      .attr('text-anchor', 'start')
      .text('True Positive Rate');

    // Línea de la Curva ROC
    const line = d3
      .line<ROCPoint>()
      .x((d) => xScale(d.fpr))
      .y((d) => yScale(d.tpr))
      .curve(d3.curveMonotoneX);

    svg
      .append('path')
      .datum(data)
      .attr('fill', 'none')
      .attr('stroke', 'steelblue')
      .attr('stroke-width', 2)
      .attr('d', line);

    // Línea de referencia (aleatoria)
    svg
      .append('line')
      .attr('x1', margin.left)
      .attr('y1', yScale(0))
      .attr('x2', width - margin.right)
      .attr('y2', yScale(1))
      .attr('stroke', 'grey')
      .attr('stroke-dasharray', '4');
  }, [data]);

  return (
    <div className="p-4 bg-white shadow rounded mb-4">
      <h3 className="text-lg font-bold mb-2">Curva ROC</h3>
      <svg ref={svgRef} className="w-full h-auto"></svg>
    </div>
  );
};

export default ROCCurve;
