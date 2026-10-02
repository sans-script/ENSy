import React, { useEffect, useRef } from 'react';

/**
 * Inline LaTeX rendered by the page's MathJax. React never owns the children of
 * the span (MathJax replaces them), so dynamic expressions update safely.
 */
export const Tex: React.FC<{ tex: string; className?: string }> = ({ tex, className }) => {
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.textContent = `$${tex}$`;
    const mj = (window as any).MathJax;
    if (mj?.typesetPromise) mj.typesetPromise([el]).catch(() => {});
  }, [tex]);

  return <span ref={ref} className={className} />;
};
