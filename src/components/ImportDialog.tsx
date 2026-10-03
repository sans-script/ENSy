import React, { useState } from 'react';
import { ExternalDataset } from '../types';

interface ImportDialogProps {
  onClose: () => void;
  onImported: (dataset: ExternalDataset) => void;
}

const readText = (file: File | null) => (file ? file.text() : Promise.resolve(null));

/** Header of a metadata TSV, with the Projector's rule: one column has no header. */
const metadataColumns = (text: string): string[] => {
  const first = text.split(/\r?\n/).find((l) => l.trim() !== '') ?? '';
  return first.includes('\t') ? first.split('\t') : [];
};

const FILE_INPUT =
  'block w-full text-[12px] file:mr-2 file:py-1 file:px-2 file:border file:border-[#111111] file:rounded-sm file:bg-white file:text-[11px] file:cursor-pointer';
const SELECT = 'w-full border border-[#111111] rounded-sm bg-white text-[12px] px-1.5 py-1';

/**
 * Import files from the TensorFlow Embedding Projector: vectors and metadata TSV and,
 * optionally, a bookmark file. ENSy then projects the vectors with its own algorithms
 * and measures each bookmarked layout as given.
 */
export const ImportDialog: React.FC<ImportDialogProps> = ({ onClose, onImported }) => {
  const [vectors, setVectors] = useState<File | null>(null);
  const [metadata, setMetadata] = useState<File | null>(null);
  const [state, setState] = useState<File | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [labelColumn, setLabelColumn] = useState('');
  const [partitionColumn, setPartitionColumn] = useState('');
  const [name, setName] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const chooseMetadata = async (file: File | null) => {
    setMetadata(file);
    const cols = file ? metadataColumns(await file.text()) : [];
    setColumns(cols);
    setLabelColumn('');
    setPartitionColumn('');
  };

  const submit = async () => {
    if (!vectors) return;
    setBusy(true);
    setError(null);
    try {
      const [v, m, s] = await Promise.all([readText(vectors), readText(metadata), readText(state)]);
      const res = await fetch('/api/import/projector', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: name.trim() || vectors.name.replace(/\.(tsv|txt)$/i, ''),
          vectors: v,
          metadata: m,
          state: s,
          label_column: labelColumn || null,
          partition_column: partitionColumn || null,
        }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(body.detail ?? res.statusText);
      onImported(body as ExternalDataset);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[100] bg-black/20 flex items-center justify-center" onClick={onClose}>
      <div
        role="dialog"
        aria-label="Import from the TF Embedding Projector"
        className="w-[440px] bg-white border border-[#111111] rounded-[4px] shadow-[3px_3px_0px_rgba(0,0,0,1)] p-4 font-sans text-[#111111]"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex justify-between items-baseline border-b border-[#111111] pb-1 mb-3">
          <span className="font-bold uppercase tracking-wide text-[12px]">Import from TF Embedding Projector</span>
          <button type="button" onClick={onClose} className="text-[12px] cursor-pointer hover:underline">close</button>
        </div>

        <div className="flex flex-col gap-2.5 text-[12px]">
          <label className="flex flex-col gap-1">
            <span className="font-bold">Vectors (TSV) — required</span>
            <input type="file" accept=".tsv,.txt" className={FILE_INPUT} onChange={(e) => setVectors(e.target.files?.[0] ?? null)} />
          </label>
          <label className="flex flex-col gap-1">
            <span className="font-bold">Metadata (TSV)</span>
            <input type="file" accept=".tsv,.txt" className={FILE_INPUT} onChange={(e) => chooseMetadata(e.target.files?.[0] ?? null)} />
          </label>
          {columns.length > 0 && (
            <div className="grid grid-cols-2 gap-2">
              <label className="flex flex-col gap-1">
                <span>Word column</span>
                <select className={SELECT} value={labelColumn} onChange={(e) => setLabelColumn(e.target.value)}>
                  <option value="">automatic</option>
                  {columns.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </label>
              <label className="flex flex-col gap-1">
                <span>Reference partition</span>
                <select className={SELECT} value={partitionColumn} onChange={(e) => setPartitionColumn(e.target.value)}>
                  <option value="">automatic</option>
                  {columns.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </label>
            </div>
          )}
          <label className="flex flex-col gap-1">
            <span className="font-bold">Bookmarks (the Projector's "state" file)</span>
            <input type="file" accept=".txt,.json" className={FILE_INPUT} onChange={(e) => setState(e.target.files?.[0] ?? null)} />
            <span className="text-[11px] text-[#555]">Each bookmark becomes a projection that ENSy measures as given.</span>
          </label>
          <label className="flex flex-col gap-1">
            <span className="font-bold">Name</span>
            <input
              type="text"
              value={name}
              placeholder={vectors?.name ?? 'Imported'}
              onChange={(e) => setName(e.target.value)}
              className="border border-[#111111] rounded-sm px-1.5 py-1 text-[12px]"
            />
          </label>

          <p className="text-[11px] text-[#555] leading-snug">
            Kept in memory until the server restarts, never mixed with the paper's frozen dataset.
            Vectors are L2-normalised; neighbours in the original space use cosine distance.
          </p>

          {error && (
            <div className="border border-[#b45309] bg-[#fffbeb] rounded-[3px] px-2 py-1 text-[11px] text-[#78350f]">{error}</div>
          )}

          <button
            type="button"
            disabled={!vectors || busy}
            onClick={submit}
            className="w-full bg-white border border-[#111111] rounded-sm py-2 font-bold uppercase tracking-wider text-[10.5px] cursor-pointer hover:bg-[#111111] hover:text-white disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-white disabled:hover:text-black"
          >
            {busy ? 'Importing…' : 'Import'}
          </button>
        </div>
      </div>
    </div>
  );
};
