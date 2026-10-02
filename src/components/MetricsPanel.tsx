import React from 'react';
import { ColorLayer, SemanticMeta } from '../types';
import { Tex } from './Tex';

interface MetricsPanelProps {
  meta: SemanticMeta | null;
  metricK: number;
  colorLayer: ColorLayer;
  probeEnabled: boolean;
  error: string | null;
  loading: boolean;
}

const CATEGORY_LABELS: Record<string, string> = {
  content_noun: 'content noun',
  content_verb: 'content verb',
  function_word: 'function word',
};

/** Fixed decimals; signed values get a spaced sign: "+ 0.022", "− 0.007". */
export const fmt = (v: number, signed = false) =>
  signed ? `${v < 0 ? '−' : '+'} ${Math.abs(v).toFixed(3)}` : v.toFixed(3);

// Same frame as the navigation HUD (BottomHUD): ivory card, black rule, 4px radius, hard shadow.
const CARD = 'absolute bottom-[16px] right-[16px] z-40 bg-white/95 border border-[#111111] rounded-[4px] shadow-[3px_3px_0px_rgba(0,0,0,1)] font-sans text-[#111111]';
const NOTE = 'mb-2 border border-[#b45309] bg-[#fffbeb] rounded-[3px] px-2 py-1 text-[11px] text-[#78350f]';

const Swatch: React.FC<{ color: string; label: string; opacity?: number }> = ({ color, label, opacity = 1 }) => (
  <span className="flex items-center gap-1.5">
    <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: color, opacity }} />
    {label}
  </span>
);

/** Co-ranking curves for the current (model, algorithm) and the active colour legend. */
export const MetricsPanel: React.FC<MetricsPanelProps> = ({ meta, metricK, colorLayer, probeEnabled, error, loading }) => {
  if (loading) {
    return (
      <div className={`${CARD} max-w-[360px] px-4 py-2.5 text-[12px] animate-pulse`}>
        Computing projection and co-ranking metrics…
      </div>
    );
  }
  if (error) {
    return (
      <div className={`${CARD} max-w-[360px] px-4 py-2.5 text-[12px] text-[#7f1d1d]`}>
        <b>No data.</b> {error}
      </div>
    );
  }
  if (!meta?.metrics) return null;

  const isomapComponents = meta.diagnostics?.knn_graph_components as number | undefined;

  return (
    <div className={`${CARD} w-[360px] px-4 py-2.5 text-[12px] pointer-events-auto`}>
      {meta.dataset_frozen === false && (
        <div className={NOTE}>
          Provisional dataset — {meta.dataset_note ?? 'not frozen'}.
        </div>
      )}
      {meta.reduced_sample && (
        <div className={NOTE}>
          Reduced sample (<Tex tex={`N = ${meta.total_tokens}`} />) — indicators may show greater variability.
        </div>
      )}

      <div className="flex justify-between font-bold uppercase tracking-wide text-[11px] border-b border-[#111111] pb-1 mb-1.5">
        <span>Co-ranking (Lee &amp; Verleysen, 2009)</span>
        <Tex className="normal-case font-normal" tex={`N = ${meta.total_tokens}`} />
      </div>
      <table className="w-full font-times-numbers text-[12.5px]">
        <thead>
          <tr className="text-left text-[#555]">
            <th className="font-normal"><Tex tex="K" /></th>
            <th className="font-normal text-right" title="Trustworthiness M_T(K)"><Tex tex="M_T" /></th>
            <th className="font-normal text-right" title="Continuity M_C(K)"><Tex tex="M_C" /></th>
            <th className="font-normal text-right" title="Overall quality Q_NX(K)"><Tex tex="Q_{NX}" /></th>
            <th className="font-normal text-right" title="Behaviour B_NX(K) = U_X − U_N; > 0 intrusive, < 0 extrusive"><Tex tex="B_{NX}" /></th>
          </tr>
        </thead>
        <tbody>
          {meta.metrics.map((m) => (
            <tr key={m.K} className={m.K === metricK ? 'font-bold bg-[#f4f4f5]' : ''}>
              <td>{m.K}</td>
              <td className="text-right">{fmt(m.trustworthiness)}</td>
              <td className="text-right">{fmt(m.continuity)}</td>
              <td className="text-right">{fmt(m.Q_NX)}</td>
              <td className="text-right">{fmt(m.B_NX, true)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {[
        ['3D projection', meta.silhouette_projected],
        ['2D projection', meta.silhouette_projected_2d],
      ].map(([label, value]) => (
        <div key={label as string} className="mt-1 flex justify-between text-[11px]">
          <span title="Complementary measure of class separation in the projected space (Rousseeuw, 1987)">
            Silhouette, reference partition — {label}
          </span>
          <span className="font-times-numbers">{typeof value === 'number' ? fmt(value, true) : '—'}</span>
        </div>
      ))}
      {isomapComponents !== undefined && isomapComponents > 1 && (
        <div className="mt-1 text-[11px] text-[#78350f]">
          Isomap k-NN graph has {isomapComponents} components; scikit-learn bridged them.
        </div>
      )}

      <div className="mt-2 pt-1.5 border-t border-dashed border-[#cccccc] text-[11px] flex flex-col gap-1">
        {colorLayer === 'heat' ? (
          <>
            <span className="font-bold">Thermal camera — local contribution to <Tex tex={`B_{NX}\\ (K = ${metricK})`} /></span>
            <div className="h-2 w-full" style={{ background: 'linear-gradient(to right, #0891b2, #d4d4d8, #ea580c)' }} />
            <div className="flex justify-between font-times-numbers text-[11px]">
              <span>− 1 extrusive</span><span>0</span><span>intrusive + 1</span>
            </div>
          </>
        ) : (
          <div className="flex gap-3 flex-wrap">
            {Object.entries(meta.category_colors ?? {}).map(([cat, color]) => (
              <Swatch key={cat} color={color} label={CATEGORY_LABELS[cat] ?? cat} />
            ))}
          </div>
        )}
        {probeEnabled && (
          <div className="flex gap-3 flex-wrap">
            <Swatch color="#16a34a" label="true" />
            <Swatch color="#dc2626" label="false" />
            <Swatch color="#3b82f6" label="missing" opacity={0.45} />
            <span className="text-[#555]"><Tex tex={`K = ${metricK}`} /> neighbours of target</span>
          </div>
        )}
      </div>
    </div>
  );
};
