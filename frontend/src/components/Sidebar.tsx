// src/components/Sidebar.tsx
import React from 'react';

export interface SidebarItem {
  key: string;
  label: string;
}

export interface SidebarProps {
  items: SidebarItem[];
  activeKey: string;
  onSelect: (key: string) => void;
}

const Sidebar: React.FC<SidebarProps> = ({ items, activeKey, onSelect }) => {
  return (
    <div className="group fixed top-0 left-0 h-screen w-12 hover:w-64 bg-gray-100 overflow-hidden transition-all duration-300 shadow-md z-50">
      {items.map((item) => (
        <div
          key={item.key}
          onClick={() => onSelect(item.key)}
          className={`p-2 my-1 cursor-pointer rounded hover:bg-gray-300 transition-colors duration-300 ${activeKey === item.key ? 'bg-gray-400 font-bold' : ''}`}
        >
          <span className="text-gray-600 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
            {item.label}
          </span>
        </div>
      ))}
    </div>
  );
};

export default Sidebar;
