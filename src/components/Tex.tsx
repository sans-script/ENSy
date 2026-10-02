import React, { useEffect, useRef } from 'react';

/**
 * LaTeX rendered by the page's MathJax without ever showing raw source: the
 * formula is converted off-screen (tex2chtmlPromise) and swapped in at once,
 * so the previous rendering stays visible until the new one is ready. React
 * never owns the children of the span.
 */
export const Tex: React.FC<{ tex: string; className?: string; display?: boolean }> = ({ tex, className, display = false }) => {
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    let cancelled = false;
    const render = () => {
      const mj = (window as any).MathJax;
      const el = ref.current;
      if (!el) return;
      if (!mj?.tex2chtmlPromise || !mj.startup?.document) {
        setTimeout(render, 100); // MathJax still loading: keep the old content
        return;
      }
      mj.tex2chtmlPromise(tex, { display })
        .then((node: HTMLElement) => {
          if (cancelled || !ref.current) return;
          ref.current.replaceChildren(node);
          mj.startup.document.clear();
          mj.startup.document.updateDocument(); // inject any CSS the new glyphs need
        })
        .catch(() => {});
    };
    render();
    return () => {
      cancelled = true;
    };
  }, [tex, display]);

  return <span ref={ref} className={className} />;
};
