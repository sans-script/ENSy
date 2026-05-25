import React from 'react';
import { Sliders } from 'lucide-react';

interface SidebarLeftProps {
  modelName: string;
  projectionType: 'umap' | 'tsne' | 'pca' | 'isomap';
  onProjectionTypeChange: (type: 'umap' | 'tsne' | 'pca' | 'isomap') => void;
  embeddingDim: number;
  totalTokens: number;
  fps: number;
  coords: { x: number; y: number; z: number };
  currentFilter: 'all' | 'positive' | 'negative';
  onFilterChange: (filter: 'all' | 'positive' | 'negative') => void;
  cameraMode: 'perspective' | 'parallel' | 'twopoint';
  onCameraModeChange: (mode: 'perspective' | 'parallel' | 'twopoint') => void;
  fov: number;
  onFovChange: (fov: number) => void;
  onApplyPreset: (preset: 'front' | 'top' | 'side' | 'iso') => void;
  showXAxis: boolean;
  onShowXAxisChange: (val: boolean) => void;
  showYAxis: boolean;
  onShowYAxisChange: (val: boolean) => void;
  showZAxis: boolean;
  onShowZAxisChange: (val: boolean) => void;
  showGridPlanes: boolean;
  onShowGridPlanesChange: (val: boolean) => void;
  showPlaneBackground: boolean;
  onShowPlaneBackgroundChange: (val: boolean) => void;
  mouseSensitivity: number;
  onMouseSensitivityChange: (val: number) => void;
  showBottomHud: boolean;
  onShowBottomHudChange: (val: boolean) => void;
  status?: "STABLE" | "COMPUTING" | "LOADING" | "OFFLINE";
}

export const SidebarLeft: React.FC<SidebarLeftProps> = ({
  modelName,
  projectionType,
  onProjectionTypeChange,
  embeddingDim,
  totalTokens,
  fps,
  coords,
  currentFilter,
  onFilterChange,
  cameraMode,
  onCameraModeChange,
  fov,
  onFovChange,
  onApplyPreset,
  showXAxis,
  onShowXAxisChange,
  showYAxis,
  onShowYAxisChange,
  showZAxis,
  onShowZAxisChange,
  showGridPlanes,
  onShowGridPlanesChange,
  showPlaneBackground,
  onShowPlaneBackgroundChange,
  mouseSensitivity,
  onMouseSensitivityChange,
  showBottomHud,
  onShowBottomHudChange,
  status,
}) => {
  return (
    <aside className="w-[280px] h-full border-r border-[#111111] bg-[#fcfcfc] flex flex-col p-[18px] overflow-y-auto shrink-0 select-none font-sans text-[#111111]">
      <div className="flex flex-col gap-[14px]">
        {/* Manifold Properties Section */}
        <div>
          <span className="block font-serif text-[14px] font-bold uppercase tracking-[0.5px] border-b border-[#111111] pb-[4px] mb-[10px] text-black">
            Manifold Properties
          </span>
          
          <div className="flex flex-col gap-2.5">
            <div className="text-[13px] flex justify-between items-center text-[#111111]">
              <span className="font-serif">Status</span>
              <span className={`font-times-numbers text-[13.5px] font-semibold ${
                status === "LOADING" ? "text-amber-600 animate-pulse" :
                status === "COMPUTING" ? "text-blue-600 animate-pulse" :
                status === "OFFLINE" ? "text-rose-600" : "text-emerald-700 font-bold"
              }`}>
                {status || "STABLE"}
              </span>
            </div>
            
            <div className="text-[13px] flex flex-col gap-1">
              <div className="flex items-center gap-1.5 font-serif">
                <span>{"$\\mathcal{M}$"}</span>
                <span className="text-[#111111]">(Model Architecture)</span>
              </div>
              <span className="font-times-numbers text-[14px] font-semibold text-black break-all">
                {modelName}
              </span>
            </div>

            <div className="text-[13px] flex justify-between items-center">
              <div className="flex items-center gap-1.5 font-serif">
                <span>{"$\\mathbb{R}^n$"}</span>
                <span className="text-[#111111]">(Dimensions)</span>
              </div>
              <span className="font-times-numbers text-[14px] font-semibold text-black">{embeddingDim}</span>
            </div>

            <div className="text-[13px] flex justify-between items-center">
              <div className="flex items-center gap-1.5 font-serif">
                <span>{"$\\mathcal{V}_c$"}</span>
                <span className="text-[#111111]">(Vocabulary)</span>
              </div>
              <span className="font-times-numbers text-[14px] font-semibold text-black">{totalTokens}</span>
            </div>

            <div className="text-[13px] flex justify-between items-center">
              <span className="font-serif">Refresh Rate</span>
              <span className="font-times-numbers text-[14px] font-semibold text-black">{fps > 0 ? fps.toFixed(1) : '—'}</span>
            </div>

            <div className="text-[13px] flex justify-between items-center">
              <div className="flex items-center gap-1.5 font-serif">
                <span>{"$x$"}</span>
                <span className="text-[#111111]">Coordinate</span>
              </div>
              <span className="font-times-numbers text-[14px] font-semibold text-black">{coords.x.toFixed(2)}</span>
            </div>

            <div className="text-[13px] flex justify-between items-center">
              <div className="flex items-center gap-1.5 font-serif">
                <span>{"$y$"}</span>
                <span className="text-[#111111]">Coordinate</span>
              </div>
              <span className="font-times-numbers text-[14px] font-semibold text-black">{coords.y.toFixed(2)}</span>
            </div>

            <div className="text-[13px] flex justify-between items-center">
              <div className="flex items-center gap-1.5 font-serif">
                <span>{"$z$"}</span>
                <span className="text-[#111111]">Coordinate</span>
              </div>
              <span className="font-times-numbers text-[14px] font-semibold text-black">{coords.z.toFixed(2)}</span>
            </div>
          </div>
        </div>

        {/* Engine and Filter Section */}
        <div className="pt-2 border-t border-dashed border-[#cccccc] flex gap-2 w-full">
          {/* Manifold Projection Engine */}
          <div className="flex-1 min-w-0">
            <span className="block text-xs font-bold text-[#111111] tracking-wide mb-1.5 truncate" title="Manifold Projection Engine">
              Projection Engine:
            </span>
            <div className="relative w-full">
              <select
                value={projectionType}
                onChange={(e) => onProjectionTypeChange(e.target.value as any)}
                className="w-full appearance-none bg-white border border-[#111111] font-serif text-[14px] font-semibold text-black pl-2 pr-8 py-1.5 outline-none cursor-pointer hover:bg-slate-50 rounded-sm transition-colors text-ellipsis"
              >
                <option value="umap">UMAP</option>
                <option value="tsne">t-SNE</option>
                <option value="pca">PCA</option>
                <option value="isomap">Isomap</option>
              </select>
              <div className="pointer-events-none absolute inset-y-0 right-2 flex items-center text-[#111111]">
                <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                  <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" />
                </svg>
              </div>
            </div>
          </div>

          {/* Manifold Filter Section */}
          <div className="flex-1 min-w-0">
            <span className="block text-xs font-bold text-[#111111] tracking-wide mb-1.5 truncate" title="Manifold Quadrant Filter">
              Quadrant Filter:
            </span>
            <div className="relative w-full">
              <select
                value={currentFilter}
                onChange={(e) => onFilterChange(e.target.value as any)}
                className="w-full appearance-none bg-white border border-[#111111] font-serif text-[14px] font-semibold text-black pl-2 pr-8 py-1.5 outline-none cursor-pointer hover:bg-slate-50 rounded-sm transition-colors text-ellipsis"
              >
                <option value="all">ℝ³</option>
                <option value="positive">ℝ³₊</option>
                <option value="negative">ℝ³₋</option>
              </select>
              <div className="pointer-events-none absolute inset-y-0 right-2 flex items-center text-[#111111]">
                <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                  <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" />
                </svg>
              </div>
            </div>
          </div>
        </div>

        {/* Camera & Views Section */}
        <div className="mt-2.5 pt-2 border-t border-dashed border-[#cccccc]">
          <span className="block font-serif text-[14px] font-bold uppercase tracking-[0.5px] border-b border-[#111111] pb-[4px] mb-[10px] text-black">
            Camera & Optics
          </span>

          <div className="flex gap-1.5 justify-between mb-3.5">
            <button
              type="button"
              onClick={() => onCameraModeChange('perspective')}
              className={`flex-1 flex flex-col items-center justify-center gap-1 bg-white border border-[#111111] rounded-sm py-2 px-1 text-center cursor-pointer font-medium transition-all duration-75 text-[9px] text-[#111111] leading-tight
                ${cameraMode === 'perspective' ? 'bg-[#eaeaea] shadow-none font-bold translate-y-[1.5px]' : 'hover:bg-slate-50 shadow-[0_1.5px_0_0_#111111] translate-y-0'}`}
            >
              <svg viewBox="0 0 24 24" fill="none" className="w-5 h-5 text-[#111111]">
                <path fillRule="evenodd" clipRule="evenodd" d="M13.9661 3.61881C14.0068 3.60986 14.0485 3.60666 14.0901 3.60929L21.571 4.08216C21.6978 4.09017 21.8153 4.15179 21.8943 4.25174C21.9733 4.35168 22.0065 4.4806 21.9856 4.60651L20.8093 11.6996C20.7974 11.7718 20.7681 11.8399 20.7241 11.8982L13.9018 20.93C13.766 21.1099 13.5187 21.1611 13.3232 21.0498L5.51288 16.6048C5.41765 16.5506 5.3454 16.4633 5.30973 16.3593L2.01622 6.76004C1.97444 6.63828 1.98645 6.50438 2.0492 6.39209C2.11195 6.2798 2.21948 6.19982 2.34467 6.17231L13.9661 3.61881ZM14.0964 4.51364L3.02932 6.94536L6.10298 15.9039L13.4241 20.0705L19.9421 11.4414L21.0184 4.95118L14.0964 4.51364Z" fill="currentColor"></path>
                <path fillRule="evenodd" clipRule="evenodd" d="M12.9687 8.98852C13.0925 8.9824 13.1978 9.07829 13.2039 9.20271L13.764 20.6475C13.7701 20.772 13.6747 20.8778 13.5509 20.8839C13.4271 20.89 13.3218 20.7941 13.3157 20.6697L12.7556 9.22487C12.7495 9.10045 12.8449 8.99464 12.9687 8.98852Z" fill="currentColor"></path>
                <path fillRule="evenodd" clipRule="evenodd" d="M21.709 4.42406C21.7679 4.53364 21.7273 4.67049 21.6183 4.72972L13.0865 9.36491C13.0378 9.39138 12.981 9.39881 12.9272 9.38577L2.41858 6.84114C2.29809 6.81197 2.22395 6.69015 2.25298 6.56905C2.28201 6.44795 2.40322 6.37343 2.52371 6.40261L12.9487 8.92698L21.4048 4.3329C21.5139 4.27367 21.65 4.31448 21.709 4.42406Z" fill="currentColor"></path>
              </svg>
              <span>Perspective</span>
            </button>
            <button
              type="button"
              onClick={() => onCameraModeChange('parallel')}
              className={`flex-1 flex flex-col items-center justify-center gap-1 bg-white border border-[#111111] rounded-sm py-2 px-1 text-center cursor-pointer font-medium transition-all duration-75 text-[9px] text-[#111111] leading-tight
                ${cameraMode === 'parallel' ? 'bg-[#eaeaea] shadow-none font-bold translate-y-[1.5px]' : 'hover:bg-slate-50 shadow-[0_1.5px_0_0_#111111] translate-y-0'}`}
            >
              <svg viewBox="0 0 24 24" fill="none" className="w-5 h-5 text-[#111111]">
                <path fillRule="evenodd" clipRule="evenodd" d="M14.1191 2.06575C14.2697 1.97756 14.456 1.97795 14.6063 2.06676L21.7622 6.29591C21.9095 6.38295 21.9999 6.5418 21.9999 6.71346V14.5609C21.9999 14.7335 21.9084 14.8932 21.7598 14.9798L9.83325 21.9344C9.68449 22.0212 9.50099 22.0217 9.3517 21.9359L2.24256 17.8477C2.09249 17.7614 1.99991 17.601 1.99991 17.4273V9.43891C1.99991 9.26653 2.09108 9.10713 2.23933 9.02035L14.1191 2.06575ZM2.96425 9.71731V17.1464L9.58954 20.9563L21.0356 14.2819V6.99053L14.3607 3.04566L2.96425 9.71731Z" fill="currentColor"></path>
                <path fillRule="evenodd" clipRule="evenodd" d="M9.49939 13.3403L21.3503 6.45678L21.5916 6.87617L9.49596 13.9018L2.45241 9.69409L2.69883 9.2777L9.49939 13.3403Z" fill="currentColor"></path>
                <path fillRule="evenodd" clipRule="evenodd" d="M9.25659 21.3275V13.621H9.73876V21.3275H9.25659Z" fill="currentColor"></path>
              </svg>
              <span>Parallel<br/>Projection</span>
            </button>
            <button
              type="button"
              onClick={() => onCameraModeChange('twopoint')}
              className={`flex-1 flex flex-col items-center justify-center gap-1 bg-white border border-[#111111] rounded-sm py-2 px-1 text-center cursor-pointer font-medium transition-all duration-75 text-[9px] text-[#111111] leading-tight
                ${cameraMode === 'twopoint' ? 'bg-[#eaeaea] shadow-none font-bold translate-y-[1.5px]' : 'hover:bg-slate-50 shadow-[0_1.5px_0_0_#111111] translate-y-0'}`}
            >
              <svg viewBox="0 0 24 24" fill="none" className="w-5 h-5 text-[#111111]">
                <path fillRule="evenodd" clipRule="evenodd" d="M8.52237 2.12056C8.67485 1.97679 8.89504 1.96013 9.06398 2.0796L21.7913 11.0796C21.9214 11.1716 22 11.3301 22 11.5002V20.5002C22 20.7625 21.8157 20.9802 21.5778 20.9989L8.85057 21.9989C8.81831 22.0014 8.78591 22.0002 8.7539 21.9952L2.39026 20.9952C2.16633 20.96 2 20.749 2 20.5002V8.50019C2 8.35422 2.05798 8.21555 2.15873 8.12056L8.52237 2.12056ZM8.85705 3.12208L2.90909 8.73015V20.0665L8.83431 20.9976L21.0909 20.0346V11.7732L8.85705 3.12208Z" fill="currentColor"></path>
                <path fillRule="evenodd" clipRule="evenodd" d="M8.59091 21.5002V2.50019H9.04546V21.5002H8.59091Z" fill="currentColor"></path>
              </svg>
              <span>Two-Point<br/>Perspective</span>
            </button>
          </div>

          {/* FOV Slider */}
          <div className="flex flex-col gap-1.5 mb-4">
            <div className="flex justify-between text-xs font-semibold text-[#111111]">
              <span className="flex items-center gap-1">
                <Sliders className="w-3.5 h-3.5 text-[#111111]" />
                Field of View (FOV)
              </span>
              <span className="font-times-numbers font-bold text-black text-[14px]">{fov}</span>
            </div>
            <input
              type="range"
              min="10"
              max="100"
              value={fov}
              onChange={(e) => onFovChange(parseFloat(e.target.value))}
              style={{
                background: `linear-gradient(to right, #111111 0%, #111111 ${((fov - 10) / 90) * 100}%, transparent ${((fov - 10) / 90) * 100}%, transparent 100%)`
              }}
              className="w-full appearance-none cursor-pointer custom-range"
            />
          </div>

          {/* Camera Presets Grid */}
          <div className="pt-2 border-t border-dashed border-[#cccccc]">
            <span className="block text-[11px] font-bold text-[#111111] uppercase tracking-wider mb-2">
              Viewing Align Preset angles:
            </span>
            <div className="grid grid-cols-2 gap-1.5">
              <button
                type="button"
                onClick={() => onApplyPreset('front')}
                className="flex items-center justify-center gap-1.5 bg-white border border-[#111111] rounded-sm py-2 px-1 text-center cursor-pointer font-medium transition-all duration-75 text-[10px] text-[#111111] hover:bg-slate-50 shadow-[0_1.5px_0_0_#111111] active:translate-y-[1.5px] active:shadow-none"
              >
                Front (XY)
              </button>
              <button
                type="button"
                onClick={() => onApplyPreset('top')}
                className="flex items-center justify-center gap-1.5 bg-white border border-[#111111] rounded-sm py-2 px-1 text-center cursor-pointer font-medium transition-all duration-75 text-[10px] text-[#111111] hover:bg-slate-50 shadow-[0_1.5px_0_0_#111111] active:translate-y-[1.5px] active:shadow-none"
              >
                Top (XZ)
              </button>
              <button
                type="button"
                onClick={() => onApplyPreset('side')}
                className="flex items-center justify-center gap-1.5 bg-white border border-[#111111] rounded-sm py-2 px-1 text-center cursor-pointer font-medium transition-all duration-75 text-[10px] text-[#111111] hover:bg-slate-50 shadow-[0_1.5px_0_0_#111111] active:translate-y-[1.5px] active:shadow-none"
              >
                Side (YZ)
              </button>
              <button
                type="button"
                onClick={() => onApplyPreset('iso')}
                className="flex items-center justify-center gap-1.5 bg-white border border-[#111111] rounded-sm py-2 px-1 text-center cursor-pointer font-medium transition-all duration-75 text-[10px] text-[#111111] hover:bg-slate-50 shadow-[0_1.5px_0_0_#111111] active:translate-y-[1.5px] active:shadow-none"
              >
                Isometric
              </button>
            </div>
          </div>
        </div>

        {/* Visibility Toggles Section */}
        <div className="mt-2.5 pt-2 border-t border-dashed border-[#cccccc]">
          <span className="block font-serif text-[14px] font-bold uppercase tracking-[0.5px] border-b border-[#111111] pb-[4px] mb-[10px] text-black">
            Axes & Grid Visibility
          </span>

          <div className="flex flex-col gap-2">
            {/* Toggle Buttons Grid */}
            <div className="grid grid-cols-2 gap-1.5 mb-1.5">
              <button
                type="button"
                onClick={() => onShowXAxisChange(!showXAxis)}
                className={`flex items-center justify-center gap-1.5 bg-white border rounded-sm py-2 px-1 text-center cursor-pointer font-medium transition-all duration-75 text-[10px]
                  ${showXAxis ? 'bg-[#ff0000]/10 border-[#ff0000] text-[#cc0000] font-bold shadow-none translate-y-[1.5px]' : 'hover:bg-slate-50 shadow-[0_1.5px_0_0_#111111] translate-y-0 text-[#111111] border-[#111111]'}`}
              >
                <span className={`w-2 h-2 rounded-full ${showXAxis ? 'bg-[#ff0000]' : 'bg-slate-300'}`} />
                X Axis
              </button>

              <button
                type="button"
                onClick={() => onShowYAxisChange(!showYAxis)}
                className={`flex items-center justify-center gap-1.5 bg-white border rounded-sm py-2 px-1 text-center cursor-pointer font-medium transition-all duration-75 text-[10px]
                  ${showYAxis ? 'bg-[#00ff00]/10 border-[#00bb00] text-[#008800] font-bold shadow-none translate-y-[1.5px]' : 'hover:bg-slate-50 shadow-[0_1.5px_0_0_#111111] translate-y-0 text-[#111111] border-[#111111]'}`}
              >
                <span className={`w-2 h-2 rounded-full ${showYAxis ? 'bg-[#00bb00]' : 'bg-slate-300'}`} />
                Y Axis
              </button>

              <button
                type="button"
                onClick={() => onShowZAxisChange(!showZAxis)}
                className={`flex items-center justify-center gap-1.5 bg-white border rounded-sm py-2 px-1 text-center cursor-pointer font-medium transition-all duration-75 text-[10px]
                  ${showZAxis ? 'bg-[#0000ff]/10 border-[#0000ff] text-[#0000cc] font-bold shadow-none translate-y-[1.5px]' : 'hover:bg-slate-50 shadow-[0_1.5px_0_0_#111111] translate-y-0 text-[#111111] border-[#111111]'}`}
              >
                <span className={`w-2 h-2 rounded-full ${showZAxis ? 'bg-[#0000ff]' : 'bg-slate-300'}`} />
                Z Axis
              </button>

              <button
                type="button"
                onClick={() => onShowGridPlanesChange(!showGridPlanes)}
                className={`flex items-center justify-center gap-1.5 bg-white border rounded-sm py-2 px-1 text-center cursor-pointer font-medium transition-all duration-75 text-[10px]
                  ${showGridPlanes ? 'bg-[#111111]/15 border-[#111111] text-[#111111] font-bold shadow-none translate-y-[1.5px]' : 'hover:bg-slate-50 shadow-[0_1.5px_0_0_#111111] translate-y-0 text-[#111111] border-[#111111]'}`}
              >
                <span className={`w-2 h-2 rounded-full ${showGridPlanes ? 'bg-[#111111]' : 'bg-slate-300'}`} />
                Planes Grid
              </button>
            </div>

            <button
                type="button"
                onClick={() => onShowPlaneBackgroundChange(!showPlaneBackground)}
                className={`flex items-center justify-center gap-1.5 bg-white border rounded-sm py-2 px-1 text-center cursor-pointer font-medium transition-all duration-75 text-[10px] w-full
                  ${showPlaneBackground ? 'bg-[#111111]/15 border-[#111111] text-[#111111] font-bold shadow-none translate-y-[1.5px]' : 'hover:bg-slate-50 shadow-[0_1.5px_0_0_#111111] translate-y-0 text-[#111111] border-[#111111]'}`}
              >
                <span className={`w-2 h-2 rounded-full ${showPlaneBackground ? 'bg-[#111111]' : 'bg-slate-300'}`} />
                Planes Background
            </button>

          <div className="pt-2 border-t border-dashed border-[#cccccc]">
            <span className="block font-serif text-[12px] font-bold uppercase tracking-[0.5px] border-b border-[#111111] pb-[4px] mb-[6px] text-black">
              Sensitivity Aim
            </span>
            <div className="flex flex-col gap-1.5 mb-4 mt-2">
              <div className="flex justify-between text-xs font-semibold text-[#111111]">
                <span className="flex items-center gap-1">
                  Mouse Sensitivity
                </span>
                <span className="font-times-numbers font-bold text-black text-[14px]">{(mouseSensitivity * 1000).toFixed(1)}</span>
              </div>
              <input
                type="range"
                min="0.0005"
                max="0.01"
                step="0.0005"
                value={mouseSensitivity}
                onChange={(e) => onMouseSensitivityChange(parseFloat(e.target.value))}
                style={{
                  background: `linear-gradient(to right, #111111 0%, #111111 ${((mouseSensitivity - 0.0005) / 0.0095) * 100}%, transparent ${((mouseSensitivity - 0.0005) / 0.0095) * 100}%, transparent 100%)`
                }}
                className="w-full appearance-none cursor-pointer custom-range"
              />
            </div>
            
            <span className="block font-serif text-[12px] font-bold uppercase tracking-[0.5px] border-b border-[#111111] pb-[4px] mb-[6px] text-black pt-2">
              HUD Shortcuts
            </span>
            <button
                type="button"
                onClick={() => onShowBottomHudChange(!showBottomHud)}
                className={`flex items-center justify-center gap-1.5 bg-white border rounded-sm py-2 px-1 text-center cursor-pointer font-medium transition-all duration-75 text-[10px] w-full
                  ${showBottomHud ? 'bg-[#111111]/15 border-[#111111] text-[#111111] font-bold shadow-none translate-y-[1.5px]' : 'hover:bg-slate-50 shadow-[0_1.5px_0_0_#111111] translate-y-0 text-[#111111] border-[#111111]'}`}
              >
                <span className={`w-2 h-2 rounded-full ${showBottomHud ? 'bg-[#111111]' : 'bg-slate-300'}`} />
                Bottom HUD (Press H)
            </button>
          </div>
          </div>
        </div>
      </div>
    </aside>
  );
};
