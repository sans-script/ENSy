import React, { useRef } from 'react';
import { ModelKey, SemanticNode, probeSets } from '../types';
import { Tex } from './Tex';
import { fmt } from './MetricsPanel';

interface SidebarRightProps {
  activeNode: SemanticNode | null;
  nodes: SemanticNode[] | null;
  metricK: number;
  probeEnabled: boolean;
  modelKey: ModelKey;
}

const ProbeList: React.FC<{ title: string; color: string; ids: Set<number>; nodes: SemanticNode[] }> = ({ title, color, ids, nodes }) => (
  <div className="text-[12px]">
    <span className="font-semibold" style={{ color }}>{title} ({ids.size})</span>
    <div className="font-serif italic text-[12.5px] leading-snug text-[#111111] break-words">
      {[...ids].map((j) => nodes.find((n) => n.id === j)?.label).join(', ') || '—'}
    </div>
  </div>
);

export const SidebarRight: React.FC<SidebarRightProps> = ({ activeNode, nodes, metricK, probeEnabled, modelKey }) => {
  const probe = activeNode && probeEnabled ? probeSets(activeNode, metricK) : null;
  const localB = activeNode?.b_nx_local?.[String(metricK)];
  const containerRef = useRef<HTMLDivElement>(null);

  return (
    <aside className="w-[340px] h-full border-l border-[#111111] bg-[#fcfcfc] flex flex-col justify-between p-[18px] overflow-y-auto shrink-0 select-none font-sans text-[#111111]">
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
            <div className="matrix-math flex flex-col items-center justify-center text-[22px] text-black my-1 shrink-0 min-h-[110px]">
              <Tex
                display
                className="w-full text-center"
                tex={`\\mathbf{y}_i = \\begin{bmatrix} ${activeNode.coords.x3d.toFixed(4)} \\\\ ${activeNode.coords.y3d.toFixed(4)} \\\\ ${activeNode.coords.z3d.toFixed(4)} \\end{bmatrix}`}
              />
            </div>

            <div className="flex flex-col gap-2.5 shrink-0">
              <div className="text-[13px] flex flex-col gap-1 border-b border-dashed border-[#cccccc] pb-2">
                <span className="font-serif text-[#111111]">Token</span>
                <span className="font-serif italic text-[16px] font-semibold text-black">{activeNode.label}</span>
              </div>

              <div className="text-[13px] flex flex-col gap-1 border-b border-dashed border-[#cccccc] pb-2">
                <span className="font-serif text-[#111111]">Embedding row (token id)</span>
                <span className="font-times-numbers text-[14px] font-semibold text-black">
                  {activeNode.token_id}
                </span>
              </div>

              <div className="text-[13px] flex flex-col gap-1 border-b border-dashed border-[#cccccc] pb-2">
                <span className="font-serif text-[#111111]">Reference partition</span>
                <span className="font-mono text-[12.5px] font-semibold text-black">
                  {activeNode.cluster}
                </span>
              </div>

              {localB !== undefined && (
                <div className="text-[13px] flex justify-between border-b border-dashed border-[#cccccc] pb-2" title="Net contribution of this token to intrusive (+) / extrusive (−) behaviour via mild K-intrusions/extrusions — not 'local distortion'">
                  <span className="font-serif text-[#111111]"><Tex tex={`B_{NX,i}\\ (K = ${metricK})`} /></span>
                  <span className="font-times-numbers text-[14px] font-semibold text-black">{fmt(localB, true)}</span>
                </div>
              )}

              {probe && nodes && (
                <div className="flex flex-col gap-1.5 border-b border-dashed border-[#cccccc] pb-2">
                  <ProbeList title="True neighbours" color="#16a34a" ids={probe.true} nodes={nodes} />
                  <ProbeList title="False neighbours" color="#dc2626" ids={probe.false} nodes={nodes} />
                  <ProbeList title="Missing neighbours" color="#3b82f6" ids={probe.missing} nodes={nodes} />
                </div>
              )}
            </div>

            {/* Nearest Neighbors section */}
            <div className="mt-1 flex-1 flex flex-col min-h-0">
              <span className="block font-serif text-[13px] font-bold uppercase tracking-[0.3px] border-b border-dashed border-[#cccccc] pb-[4px] mb-[10px] text-black shrink-0">
                Nearest Neighbours (cosine, <Tex className="normal-case" tex="\mathbb{R}^n" />)
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

      {/* vectors, metadata and a bookmark per projection, for projector.tensorflow.org */}
      <a
        href={`/api/export/projector?model=${modelKey}`}
        download
        title="vectors.tsv, metadata.tsv, state.txt (a bookmark with ENSy's exact layout per projection) and projector_config.json"
        className="shrink-0 mt-[14px] w-full flex items-center justify-center bg-white border border-[#111111] hover:bg-[#111111] hover:text-white rounded-sm py-2 px-1 text-center cursor-pointer transition-all duration-75 text-[10.5px] text-black uppercase font-bold tracking-wider"
      >
        Export to TF Projector
      </a>

      <div className="text-[12px] font-serif italic text-slate-700 text-center border-t border-dashed border-[#cccccc] pt-[12px] mt-[14px] shrink-0">
        ENSy v2.1.2 — projected coordinates y<sub>i</sub> ∈ ℝ³
      </div>
    </aside>
  );
};

