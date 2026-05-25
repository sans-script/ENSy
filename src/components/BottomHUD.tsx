import React, { useEffect, useState } from 'react';

export const BottomHUD: React.FC = () => {
  const [pressedKeys, setPressedKeys] = useState<Set<string>>(new Set());

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Direct mapping to standard WASDRF controls and Escape
      if (['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyR', 'KeyF', 'Escape'].includes(e.code)) {
        setPressedKeys((prev) => {
          const next = new Set(prev);
          next.add(e.code);
          return next;
        });
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyR', 'KeyF', 'Escape'].includes(e.code)) {
        setPressedKeys((prev) => {
          const next = new Set(prev);
          next.delete(e.code);
          return next;
        });
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, []);

  return (
    <div className="absolute bottom-[16px] left-[16px] z-20 flex pointer-events-none w-max max-w-[90%]">
      <div className="bg-white/95 border border-[#111111] px-4 py-2.5 shadow-[3px_3px_0px_rgba(0,0,0,1)] rounded-[4px] flex items-stretch gap-[20px] pointer-events-auto text-[#111111] font-sans">
        
        {/* Navigation Indicator Group */}
        <div className="flex flex-col items-start gap-1.5">
          <span className="font-bold uppercase text-[10px] tracking-[0.5px] text-[#111111]">
            Navigation
          </span>
          <div className="inline-flex gap-2 items-center">
            <span className={`key ${pressedKeys.has('KeyW') ? 'active' : ''}`}>W</span>
            <span className={`key ${pressedKeys.has('KeyA') ? 'active' : ''}`}>A</span>
            <span className={`key ${pressedKeys.has('KeyS') ? 'active' : ''}`}>S</span>
            <span className={`key ${pressedKeys.has('KeyD') ? 'active' : ''}`}>D</span>
          </div>
        </div>

        {/* Divider 1 */}
        <div className="border-l border-dashed border-[#cccccc] self-stretch my-0.5" />

        {/* Elevation Group */}
        <div className="flex flex-col items-start gap-1.5">
          <span className="font-bold uppercase text-[10px] tracking-[0.5px] text-[#111111]">
            Elevation
          </span>
          <div className="inline-flex gap-2 items-center">
            <span className={`key ${pressedKeys.has('KeyR') ? 'active' : ''}`}>R</span>
            <span className={`key ${pressedKeys.has('KeyF') ? 'active' : ''}`}>F</span>
          </div>
        </div>

        {/* Divider 2 */}
        <div className="border-l border-dashed border-[#cccccc] self-stretch my-0.5" />

        {/* Mouse controls */}
        <div className="flex flex-col items-start gap-1.5">
          <span className="font-bold uppercase text-[10px] tracking-[0.5px] text-[#111111]">
            Mouse
          </span>
          <div className="inline-flex gap-2 items-center">
            <span className={`key min-w-[32px] ${pressedKeys.has('Escape') ? 'active' : ''}`}>ESC</span>
          </div>
        </div>
      </div>
    </div>
  );
};
