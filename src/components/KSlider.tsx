import React from 'react';
import { Tex } from './Tex';

/** The values the paper reports; the slider snaps to them within ±1. */
export const K_REPORTED = [5, 10, 15, 20];
const SNAP = 1;

interface KSliderProps {
  value: number;
  max: number;
  onChange: (k: number) => void;
}

const snap = (k: number) => {
  const near = K_REPORTED.find((r) => Math.abs(r - k) <= SNAP);
  return near ?? k;
};

/**
 * Neighbourhood size K from 1 to max. Every quantity is precomputed for each K,
 * so the behaviour map, the probe and the metrics panel follow while dragging.
 */
export const KSlider: React.FC<KSliderProps> = ({ value, max, onChange }) => {
  const pct = (k: number) => ((k - 1) / (max - 1)) * 100;
  return (
    <div className="select-none">
      <div className="flex items-baseline justify-between mb-1">
        <span className="text-xs font-bold text-[#111111] tracking-wide">
          Neighbourhood <Tex tex="K" />:
        </span>
        <span className="text-[13px] text-[#111111]">
          <Tex tex={`K = ${value}`} />
          {value === 10 && <span className="text-[11px] text-[#555]"> (main)</span>}
        </span>
      </div>
      <input
        type="range"
        aria-label="Neighbourhood K"
        min={1}
        max={max}
        step={1}
        value={value}
        onChange={(e) => onChange(snap(Number(e.target.value)))}
        // same look as the field-of-view slider
        className="w-full appearance-none cursor-pointer custom-range"
        style={{
          background: `linear-gradient(to right, #111111 0%, #111111 ${pct(value)}%, transparent ${pct(value)}%, transparent 100%)`,
        }}
      />
      {/* ticks at the reported K values */}
      <div className="relative h-3.5 mt-0.5 mx-[6px]">
        {K_REPORTED.filter((k) => k <= max).map((k) => (
          <button
            key={k}
            type="button"
            onClick={() => onChange(k)}
            className={`absolute -translate-x-1/2 text-[10px] leading-none cursor-pointer hover:underline
              ${k === value ? 'font-bold text-[#111111]' : 'text-[#555]'}`}
            style={{ left: `${pct(k)}%` }}
          >
            {k}
          </button>
        ))}
      </div>
    </div>
  );
};
