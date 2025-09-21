import { FormEvent, useEffect, useState } from 'react';

import type {
  DatasetSummary,
  EmbeddingPoint,
  InferenceResult,
  MLMetadata,
} from '../types';

interface ControlPanelProps {
  datasets: DatasetSummary[];
  currentDatasetId?: string;
  onSelectDataset: (datasetId: string) => void;
  selectedCell: EmbeddingPoint | null;
  onAnnotate: (cellId: string, label: string) => Promise<void>;
  onInference: (cellId: string) => Promise<void>;
  inference?: InferenceResult | null;
  mlMetadata?: MLMetadata;
}

export const ControlPanel = ({
  datasets,
  currentDatasetId,
  onSelectDataset,
  selectedCell,
  onAnnotate,
  onInference,
  inference,
  mlMetadata,
}: ControlPanelProps) => {
  const [annotation, setAnnotation] = useState('');

  useEffect(() => {
    setAnnotation(selectedCell?.annotation ?? '');
  }, [selectedCell?.annotation]);

  const handleAnnotate = async (event: FormEvent) => {
    event.preventDefault();
    if (selectedCell && annotation.trim().length > 0) {
      await onAnnotate(selectedCell.cell_id, annotation.trim());
    }
  };

  return (
    <div className="control-panel">
      <section>
        <h2>Datasets</h2>
        <select
          aria-label="Select dataset"
          value={currentDatasetId ?? ''}
          onChange={(event) => onSelectDataset(event.target.value)}
        >
          {!currentDatasetId && <option value="">Select dataset…</option>}
          {datasets.map((dataset) => (
            <option key={dataset.dataset_id} value={dataset.dataset_id}>
              {dataset.dataset_id} ({dataset.cell_count} cells)
            </option>
          ))}
        </select>
      </section>

      <section>
        <h2>Cell Details</h2>
        {selectedCell ? (
          <div className="detail">
            <p>
              <strong>ID:</strong> {selectedCell.cell_id}
            </p>
            <p>
              <strong>Cluster:</strong> {selectedCell.cluster}
            </p>
            <p>
              <strong>Annotation:</strong> {selectedCell.annotation ?? '—'}
            </p>
            <form onSubmit={handleAnnotate} className="annotation-form">
              <label htmlFor="annotation-input">Assign label</label>
              <input
                id="annotation-input"
                type="text"
                value={annotation}
                onChange={(event) => setAnnotation(event.target.value)}
                placeholder="e.g. Neuron"
              />
              <button type="submit" disabled={annotation.trim().length === 0}>
                Save annotation
              </button>
            </form>
            <button type="button" onClick={() => onInference(selectedCell.cell_id)}>
              Run inference
            </button>
          </div>
        ) : (
          <p>Select a cell in the embedding to annotate or run inference.</p>
        )}
      </section>

      <section>
        <h2>ML Inference</h2>
        {inference ? (
          <div className="detail">
            <p>
              <strong>Predicted type:</strong> {inference.predicted_type}
            </p>
            <p>
              <strong>Latent vector:</strong> {inference.latent_vector.map((value) => value.toFixed(2)).join(', ')}
            </p>
          </div>
        ) : (
          <p>Run inference on a selected cell to view predictions.</p>
        )}
      </section>

      <section>
        <h2>Model metadata</h2>
        {mlMetadata ? (
          <ul>
            <li>
              Latent dimension: <strong>{mlMetadata.latent_dim}</strong>
            </li>
            <li>
              Input dimension: <strong>{mlMetadata.input_dim}</strong>
            </li>
            <li>
              Classes: <strong>{mlMetadata.classes.join(', ')}</strong>
            </li>
          </ul>
        ) : (
          <p>Loading model information…</p>
        )}
      </section>
    </div>
  );
};
