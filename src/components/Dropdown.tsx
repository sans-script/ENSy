import React, { useEffect, useId, useRef, useState } from 'react';

export interface DropdownOption<T extends string | number> {
  value: T;
  /** Rendered content (may contain <Tex/>). */
  label: React.ReactNode;
  /** Plain-text version, used for the tooltip of truncated labels. */
  text: string;
}

interface DropdownProps<T extends string | number> {
  value: T;
  options: DropdownOption<T>[];
  onChange: (value: T) => void;
  ariaLabel: string;
  /** Which trigger edge the open menu lines up with (use 'right' in the right-hand column). */
  align?: 'left' | 'right';
}

const Chevron = () => (
  <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
    <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" />
  </svg>
);

/** Select replacement styled like the rest of the dashboard (serif, black rules, hard shadow). */
export function Dropdown<T extends string | number>({ value, options, onChange, ariaLabel, align = 'left' }: DropdownProps<T>) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const listId = useId();
  const selected = options.find((o) => o.value === value) ?? options[0];

  useEffect(() => {
    if (!open) return;
    setActive(Math.max(0, options.findIndex((o) => o.value === value)));
    const onDown = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [open]);

  const choose = (v: T) => {
    onChange(v);
    setOpen(false);
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      setOpen(false);
    } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      if (!open) return setOpen(true);
      const step = e.key === 'ArrowDown' ? 1 : -1;
      setActive((i) => (i + step + options.length) % options.length);
    } else if ((e.key === 'Enter' || e.key === ' ') && open) {
      e.preventDefault();
      choose(options[active].value);
    }
  };

  return (
    <div ref={rootRef} className="relative w-full" onKeyDown={onKeyDown}>
      <button
        type="button"
        aria-label={ariaLabel}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={listId}
        title={selected?.text}
        onClick={() => setOpen((o) => !o)}
        className={`relative w-full flex items-center bg-white border border-[#111111] font-serif text-[14px] font-semibold text-black pl-2 pr-7 py-1.5 outline-none cursor-pointer rounded-sm transition-colors text-left ${open ? 'bg-[#eaeaea]' : 'hover:bg-slate-50'}`}
      >
        <span className="block min-w-0 truncate [&_mjx-container]:pointer-events-none [&_mjx-container]:outline-none">{selected?.label}</span>
        <span className="pointer-events-none absolute inset-y-0 right-2 flex items-center text-[#111111]">
          <Chevron />
        </span>
      </button>

      {open && (
        <ul
          id={listId}
          role="listbox"
          aria-label={ariaLabel}
          className={`absolute ${align === 'right' ? 'right-0' : 'left-0'} top-full mt-1 min-w-full w-max max-w-[284px] z-50 bg-white border border-[#111111] rounded-[4px] shadow-[3px_3px_0px_rgba(0,0,0,1)] py-1`}
        >
          {options.map((o, i) => {
            const isSelected = o.value === value;
            return (
              <li
                key={String(o.value)}
                role="option"
                aria-selected={isSelected}
                data-value={String(o.value)}
                title={o.text}
                onMouseEnter={() => setActive(i)}
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => choose(o.value)}
                className={`flex items-center gap-1.5 px-2 py-1.5 cursor-pointer font-serif text-[13.5px] text-black
                  ${i === active ? 'bg-[#eaeaea]' : ''} ${isSelected ? 'font-bold' : 'font-normal'}`}
              >
                <span className={`w-1.5 h-1.5 shrink-0 rounded-full ${isSelected ? 'bg-[#111111]' : 'bg-transparent'}`} />
                <span className="block min-w-0 truncate [&_mjx-container]:pointer-events-none [&_mjx-container]:outline-none">{o.label}</span>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
