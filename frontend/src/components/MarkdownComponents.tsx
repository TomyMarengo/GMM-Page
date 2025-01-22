// src/components/MarkdownComponents.tsx

import React from 'react';
import { Components } from 'react-markdown';

const markdownComponents: Components = {
  h1: ({ children, ...props }) => (
    <h1 className="text-3xl font-bold mb-2" {...props}>
      {children}
    </h1>
  ),
  h2: ({ children, ...props }) => (
    <h2 className="text-2xl font-semibold mb-2" {...props}>
      {children}
    </h2>
  ),
  p: ({ children, ...props }) => (
    <p className="mb-2 text-gray-800" {...props}>
      {children}
    </p>
  ),
  ul: ({ children, ...props }) => (
    <ul className="list-disc list-inside mb-2" {...props}>
      {children}
    </ul>
  ),
  li: ({ children, ...props }) => (
    <li className="mb-1 text-gray-700" {...props}>
      {children}
    </li>
  ),
  strong: ({ children, ...props }) => (
    <strong className="font-bold" {...props}>
      {children}
    </strong>
  ),
  // Puedes añadir más componentes personalizados según sea necesario
};

export default markdownComponents;
