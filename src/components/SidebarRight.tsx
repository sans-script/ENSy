import React, { useEffect, useRef, useState } from 'react';
import { SemanticNode } from '../types';

interface SidebarRightProps {
  activeNode: SemanticNode | null;
}

export const SidebarRight: React.FC<SidebarRightProps> = ({ activeNode }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const matrixRef = useRef<HTMLDivElement>(null);
  const [isTypesetting, setIsTypesetting] = useState(false);

  useEffect(() => {
    setIsTypesetting(!!activeNode);
  }, [activeNode?.token_id]);

  // Re-typeset LaTeX equations when selected node updates
  useEffect(() => {
    if (activeNode && (window as any).MathJax && (window as any).MathJax.typesetPromise) {
      const timer = setTimeout(() => {
        if (containerRef.current) {
          // Passamos referências específicas para re-renderizar apenas o título e o container da matriz
          (window as any).MathJax.typesetPromise([containerRef.current]).then(() => {
            requestAnimationFrame(() => {
              requestAnimationFrame(() => {
                setIsTypesetting(false);
              });
            });
          }).catch((err: any) => {
            console.error('MathJax typeset error:', err);
            setIsTypesetting(false);
          });
        }
      }, 50);
      return () => clearTimeout(timer);
    }
  }, [activeNode]);

  return (
    <aside className="w-[280px] h-full border-l border-[#111111] bg-[#fcfcfc] flex flex-col justify-between p-[18px] overflow-y-auto shrink-0 select-none font-sans text-[#111111]">
      <div className="flex flex-col h-full overflow-hidden" ref={containerRef}>
        <span className="block font-serif text-[14px] font-bold uppercase tracking-[0.5px] border-b border-[#111111] pb-[4px] mb-[12px] text-black shrink-0">
          Manifold Vector Analysis
        </span>

        {/* Display Idle / Active Vector status */}
        {!activeNode ? (
          <div className="flex-1 flex items-center justify-center text-center px-2 py-8 text-gray-500 font-serif italic text-[14px] leading-relaxed">
            Align the central mathematical reticle with a vector point to map its local neighbors.
          </div>
        ) : (
          <div className="flex-1 flex flex-col gap-3.5 min-h-0">
            {/* Vector Equation and Matrix */}
            <div className={`matrix-math flex flex-col items-center justify-center text-[22px] text-black my-1 shrink-0 min-h-[110px] transition-opacity ${isTypesetting ? 'opacity-0 duration-0' : 'opacity-100 duration-500'}`}>
              <div ref={matrixRef} className="w-full text-center">
                {`$$\\vec{v} = \\begin{bmatrix} ${activeNode.coords.x3d.toFixed(4)} \\\\ ${activeNode.coords.y3d.toFixed(4)} \\\\ ${activeNode.coords.z3d.toFixed(4)} \\end{bmatrix}$$`}
              </div>
            </div>

            <div className="flex flex-col gap-2.5 shrink-0">
              <div className="text-[13px] flex flex-col gap-1 border-b border-dashed border-[#cccccc] pb-2">
                <span className="font-serif text-[#111111]">Vector Matrix Index</span>
                <span className="font-times-numbers text-[14px] font-semibold text-black">
                  {activeNode.token_id}
                </span>
              </div>

              <div className="text-[13px] flex flex-col gap-1 border-b border-dashed border-[#cccccc] pb-2">
                <span className="font-serif text-[#111111]">Sub-Manifold Cluster</span>
                <span className="font-mono text-[12.5px] font-semibold text-black">
                  {activeNode.cluster}
                </span>
              </div>
            </div>

            {/* Nearest Neighbors section */}
            <div className="mt-1 flex-1 flex flex-col min-h-0">
              <span className="block font-serif text-[13px] font-bold uppercase tracking-[0.3px] border-b border-dashed border-[#cccccc] pb-[4px] mb-[10px] text-black shrink-0">
                Local Neighbors
              </span>
              
              <div className="flex-1 overflow-y-auto mb-2 pr-1">
                <table className="w-full border-collapse">
                  <tbody>
                    {activeNode.top5_neighbors.map((neighbor, index) => (
                      <tr 
                        key={index} 
                        className="flex justify-between items-center py-2.5 border-b border-dashed border-[#cccccc] last:border-b-0"
                      >
                        <td className="text-left font-sans font-semibold text-[14px] text-[#111111] tracking-wide">
                          {neighbor.label}
                        </td>
                        <td className="text-right font-times-numbers text-[14px] font-semibold text-[#111111]">
                          {neighbor.cosine_similarity.toFixed(4)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <button
                onClick={() => {
                  const csvRows = ['neighbor,cosine_similarity'];
                  activeNode.top5_neighbors.forEach(n => {
                    csvRows.push(`${n.label},${n.cosine_similarity.toFixed(4)}`);
                  });
                  const blob = new Blob([csvRows.join('\n')], { type: 'text/csv' });
                  const url = window.URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.setAttribute('hidden', '');
                  a.setAttribute('href', url);
                  a.setAttribute('download', `vector_${activeNode.label}_neighbors.csv`);
                  document.body.appendChild(a);
                  a.click();
                  document.body.removeChild(a);
                }}
                className="shrink-0 w-full flex items-center justify-center gap-1.5 bg-white border border-[#111111] hover:bg-[#111111] hover:text-white rounded-sm py-2 px-1 text-center cursor-pointer font-medium transition-all duration-75 text-[10.5px] text-black group uppercase font-bold tracking-wider"
              >
                Export CSV Dump
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="text-[12px] font-serif italic text-slate-700 text-center border-t border-dashed border-[#cccccc] pt-[12px] mt-[14px] shrink-0">
        Topological Data Analysis v0.4
      </div>
    </aside>
  );
};

