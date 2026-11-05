import React, { useCallback, useEffect, useRef, useState } from 'react';
import { PresentationStep, SemanticNode, SemanticSpace } from '../types';

const STORAGE_KEY = 'ensyPresentationSteps';

function loadSteps(): PresentationStep[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as PresentationStep[]) : [];
  } catch {
    return [];
  }
}

function saveSteps(steps: PresentationStep[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(steps));
  } catch {
    // best-effort only; a failed save just means the list resets on reload
  }
}

interface PresentationRemoteProps {
  /** The token currently centred/pinned in the viewport (right sidebar's "Token"),
   *  used as the source for "Salvar vista atual". */
  activeNode: SemanticNode | null;
  /** Default K for a newly saved step, when the step doesn't carry its own. */
  metricK: number;
  semanticSpace: SemanticSpace | null;
  presentationMode: boolean;
  onPresentationModeChange: (on: boolean) => void;
}

/**
 * Floating remote-control panel for driving the camera with a gamepad instead of
 * keyboard/mouse -- built so a TCC defense (or any screen-facing demo) doesn't
 * depend on Pointer Lock, which browser automation can't engage but a presenter's
 * own hands can use fine; this panel adds a *scripted* second mode on top of that.
 *
 * Two modes, both gamepad-driven (Y/Triangle, gamepad button index 3, toggles
 * between them):
 *  - Free flight (default): left stick = move, right stick = look, triggers =
 *    up/down. Handled inside SpaceVisualizer's own render loop
 *    (applyGamepadInputPrivate), gated off while presentationMode is on.
 *  - Presentation mode: L1/R1 (buttons 4/5) step backward/forward through a saved
 *    list of views, each driving __focusCamera exactly like the ?focus= URL
 *    parameter. Build the list live with "+ Salvar vista atual" while exploring
 *    normally, then step through it during the actual talk.
 *
 * Steps persist in localStorage (ensyPresentationSteps) and are scoped to one
 * model/algorithm at a time: jumping to a step whose token isn't in the
 * currently-loaded dataset is a no-op (see jumpTo's hasToken check) rather than a
 * silent wrong jump -- switch model/dataset by hand first if a walkthrough spans
 * more than one.
 */
export const PresentationRemote: React.FC<PresentationRemoteProps> = ({
  activeNode,
  metricK,
  semanticSpace,
  presentationMode,
  onPresentationModeChange,
}) => {
  const [steps, setSteps] = useState<PresentationStep[]>(() => loadSteps());
  const [index, setIndex] = useState(0);
  const [expanded, setExpanded] = useState(false);
  const [gamepadConnected, setGamepadConnected] = useState(false);

  const stepsRef = useRef(steps);
  const indexRef = useRef(index);
  const presentationModeRef = useRef(presentationMode);
  const semanticSpaceRef = useRef(semanticSpace);
  const metricKRef = useRef(metricK);

  useEffect(() => {
    stepsRef.current = steps;
  }, [steps]);
  useEffect(() => {
    indexRef.current = index;
  }, [index]);
  useEffect(() => {
    presentationModeRef.current = presentationMode;
  }, [presentationMode]);
  useEffect(() => {
    semanticSpaceRef.current = semanticSpace;
  }, [semanticSpace]);
  useEffect(() => {
    metricKRef.current = metricK;
  }, [metricK]);

  const jumpTo = useCallback((i: number) => {
    const list = stepsRef.current;
    if (i < 0 || i >= list.length) return;
    const step = list[i];
    const space = semanticSpaceRef.current;
    const target: string | number = step.tokenId ?? step.label;
    const hasToken = space?.nodes.some(
      (n) => n.label === target || n.token_id === target || n.id === target,
    );
    if (!hasToken) return; // step belongs to a dataset that isn't the one loaded right now
    const focusCamera = (window as any).__focusCamera;
    if (typeof focusCamera !== 'function') return;
    focusCamera(target, step.k ?? metricKRef.current, true, step.zoom);
    setIndex(i);
  }, []);

  const next = useCallback(() => {
    const list = stepsRef.current;
    if (list.length === 0) return;
    jumpTo(Math.min(indexRef.current + 1, list.length - 1));
  }, [jumpTo]);

  const prev = useCallback(() => {
    const list = stepsRef.current;
    if (list.length === 0) return;
    jumpTo(Math.max(indexRef.current - 1, 0));
  }, [jumpTo]);

  const addCurrentView = useCallback(() => {
    if (!activeNode) return;
    const step: PresentationStep = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      tokenId: activeNode.id,
      label: activeNode.label,
      k: metricKRef.current,
    };
    setSteps((prev) => {
      const next = [...prev, step];
      saveSteps(next);
      return next;
    });
  }, [activeNode]);

  const removeStep = useCallback((id: string) => {
    setSteps((prev) => {
      const next = prev.filter((s) => s.id !== id);
      saveSteps(next);
      return next;
    });
  }, []);

  const moveStep = useCallback((i: number, dir: -1 | 1) => {
    setSteps((prev) => {
      const j = i + dir;
      if (j < 0 || j >= prev.length) return prev;
      const next = prev.slice();
      [next[i], next[j]] = [next[j], next[i]];
      saveSteps(next);
      return next;
    });
  }, []);

  // Gamepad: connection status + edge-detected shoulder buttons (prev/next) and the
  // Y/Triangle button (toggle presentation mode). Left/right-stick flight itself is
  // polled separately inside SpaceVisualizer's own render loop, not here -- this
  // panel only needs discrete button *presses* (edge, not held state), which is
  // cheap enough to poll on its own requestAnimationFrame loop independent of the
  // 3D scene's.
  useEffect(() => {
    let raf = 0;
    const prevPressed = { l1: false, r1: false, y: false };

    const refreshConnected = () => {
      const pads = navigator.getGamepads ? navigator.getGamepads() : [];
      setGamepadConnected(Array.from(pads ?? []).some((p) => p && p.connected));
    };
    window.addEventListener('gamepadconnected', refreshConnected);
    window.addEventListener('gamepaddisconnected', refreshConnected);
    refreshConnected();

    const poll = () => {
      raf = requestAnimationFrame(poll);
      const pads = navigator.getGamepads ? navigator.getGamepads() : [];
      const pad = Array.from(pads ?? []).find((p) => p && p.connected);
      if (!pad) return;

      const l1 = pad.buttons[4]?.pressed ?? false;
      const r1 = pad.buttons[5]?.pressed ?? false;
      const y = pad.buttons[3]?.pressed ?? false;

      if (y && !prevPressed.y) {
        onPresentationModeChange(!presentationModeRef.current);
      }
      if (presentationModeRef.current) {
        if (r1 && !prevPressed.r1) next();
        if (l1 && !prevPressed.l1) prev();
      }

      prevPressed.l1 = l1;
      prevPressed.r1 = r1;
      prevPressed.y = y;
    };
    raf = requestAnimationFrame(poll);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('gamepadconnected', refreshConnected);
      window.removeEventListener('gamepaddisconnected', refreshConnected);
    };
  }, [next, prev, onPresentationModeChange]);

  const currentStep = steps[index];

  return (
    <div
      className="absolute bottom-4 right-4 z-40 font-sans text-xs select-none"
      data-hud="presentation-remote"
    >
      <div className="bg-white/95 border border-[#111111] px-3 py-2 shadow-[3px_3px_0px_rgba(0,0,0,1)] rounded-[4px] w-64">
        <div className="flex items-center justify-between mb-1.5">
          <span className="font-bold">Modo Apresentação</span>
          <span className={gamepadConnected ? 'text-green-700' : 'text-gray-400'}>
            {gamepadConnected ? 'controle conectado' : 'sem controle'}
          </span>
        </div>

        <div className="flex gap-1.5 mb-1.5">
          <button
            onClick={() => onPresentationModeChange(!presentationMode)}
            className={`flex-1 border border-[#111111] px-2 py-1 rounded-[3px] ${
              presentationMode ? 'bg-black text-white' : 'bg-white'
            }`}
          >
            {presentationMode ? 'Sair (Y)' : 'Entrar (Y)'}
          </button>
          <button
            onClick={() => setExpanded((e) => !e)}
            className="border border-[#111111] px-2 py-1 rounded-[3px] bg-white"
            aria-label={expanded ? 'Recolher lista de passos' : 'Expandir lista de passos'}
          >
            {expanded ? '▲' : '▼'}
          </button>
        </div>

        {presentationMode && steps.length > 0 && (
          <div className="mb-1.5 text-center truncate">
            {index + 1} / {steps.length} {'—'} <span className="font-bold">{currentStep?.label}</span>
          </div>
        )}

        {presentationMode && (
          <div className="flex gap-1.5 mb-1.5">
            <button
              onClick={prev}
              className="flex-1 border border-[#111111] px-2 py-1 rounded-[3px] bg-white"
            >
              ◀ L1
            </button>
            <button
              onClick={next}
              className="flex-1 border border-[#111111] px-2 py-1 rounded-[3px] bg-white"
            >
              R1 {'▶'}
            </button>
          </div>
        )}

        {expanded && (
          <div className="border-t border-black/20 pt-1.5 mt-1.5">
            <button
              onClick={addCurrentView}
              disabled={!activeNode}
              className="w-full border border-[#111111] px-2 py-1 rounded-[3px] bg-white disabled:opacity-40 mb-1.5"
            >
              {activeNode ? `+ Salvar vista atual: ${activeNode.label}` : '+ Salvar vista atual'}
            </button>
            <div className="max-h-48 overflow-y-auto">
              {steps.length === 0 && (
                <div className="text-gray-400 italic py-1">Nenhum passo salvo ainda.</div>
              )}
              {steps.map((s, i) => (
                <div
                  key={s.id}
                  className={`flex items-center gap-1 py-0.5 ${
                    presentationMode && i === index ? 'bg-black/5' : ''
                  }`}
                >
                  <span className="w-4 text-right text-gray-400">{i + 1}</span>
                  <button
                    onClick={() => jumpTo(i)}
                    className="flex-1 text-left truncate hover:underline"
                  >
                    {s.label}
                    {s.k ? ` (K=${s.k})` : ''}
                  </button>
                  <button
                    onClick={() => moveStep(i, -1)}
                    disabled={i === 0}
                    className="px-1 disabled:opacity-30"
                    aria-label="Mover para cima"
                  >
                    {'↑'}
                  </button>
                  <button
                    onClick={() => moveStep(i, 1)}
                    disabled={i === steps.length - 1}
                    className="px-1 disabled:opacity-30"
                    aria-label="Mover para baixo"
                  >
                    {'↓'}
                  </button>
                  <button
                    onClick={() => removeStep(s.id)}
                    className="px-1 text-red-700"
                    aria-label="Remover passo"
                  >
                    {'✕'}
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
