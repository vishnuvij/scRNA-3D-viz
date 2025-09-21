import { useCallback, useEffect, useMemo, useState } from 'react';

import type {
  AnnotationPayload,
  DatasetSummary,
  EmbeddingPoint,
  InferenceResult,
  MetadataPayload,
  MLMetadata,
} from '../types';

const API_BASE = (import.meta.env.VITE_BACKEND_URL as string | undefined) ?? 'http://localhost:8000';

const createUrl = (path: string) => {
  const base = API_BASE.endsWith('/') ? API_BASE.slice(0, -1) : API_BASE;
  return `${base}${path}`;
};

const createWebSocketUrl = (path: string) => {
  const url = new URL(path, createUrl('/'));
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  return url.toString();
};

interface BackendState {
  datasets: DatasetSummary[];
  currentDatasetId?: string;
  embedding: EmbeddingPoint[];
  metadata?: MetadataPayload;
  mlMetadata?: MLMetadata;
  inference?: InferenceResult | null;
  loading: boolean;
  selectedCell?: EmbeddingPoint | null;
  selectDataset: (datasetId: string) => void;
  selectCell: (cellId: string | null) => void;
  annotateCell: (cellId: string, label: string) => Promise<void>;
  runInference: (cellId: string) => Promise<void>;
}

export const useBackendData = (): BackendState => {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [currentDatasetId, setCurrentDatasetId] = useState<string>();
  const [embedding, setEmbedding] = useState<EmbeddingPoint[]>([]);
  const [metadata, setMetadata] = useState<MetadataPayload>();
  const [mlMetadata, setMlMetadata] = useState<MLMetadata>();
  const [inference, setInference] = useState<InferenceResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [selectedCellId, setSelectedCellId] = useState<string | null>(null);

  const fetchDatasets = useCallback(async () => {
    const response = await fetch(createUrl('/datasets'));
    if (!response.ok) {
      throw new Error('Failed to load datasets');
    }
    const data: DatasetSummary[] = await response.json();
    setDatasets(data);
    if (!currentDatasetId && data.length > 0) {
      setCurrentDatasetId(data[0].dataset_id);
    }
  }, [currentDatasetId]);

  const fetchMetadata = useCallback(async (datasetId: string) => {
    const response = await fetch(createUrl(`/datasets/${datasetId}/metadata`));
    if (response.ok) {
      const payload: MetadataPayload = await response.json();
      setMetadata(payload);
    }
  }, []);

  const fetchEmbedding = useCallback(async (datasetId: string) => {
    setLoading(true);
    try {
      const response = await fetch(createUrl(`/datasets/${datasetId}/embedding`));
      if (!response.ok) {
        throw new Error('Failed to load embedding');
      }
      const payload = await response.json();
      setEmbedding(payload.embedding as EmbeddingPoint[]);
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchMlMetadata = useCallback(async () => {
    const response = await fetch(createUrl('/ml/metadata'));
    if (response.ok) {
      const payload: MLMetadata = await response.json();
      setMlMetadata(payload);
    }
  }, []);

  useEffect(() => {
    fetchDatasets().catch((error) => console.error(error));
    fetchMlMetadata().catch((error) => console.error(error));
  }, [fetchDatasets, fetchMlMetadata]);

  useEffect(() => {
    if (!currentDatasetId) {
      return;
    }
    fetchMetadata(currentDatasetId).catch((error) => console.error(error));
    fetchEmbedding(currentDatasetId).catch((error) => console.error(error));
    setSelectedCellId(null);
    setInference(null);
  }, [currentDatasetId, fetchEmbedding, fetchMetadata]);

  useEffect(() => {
    const socket = new WebSocket(createWebSocketUrl('/ws/annotations'));
    socket.onmessage = (event) => {
      const payload: AnnotationPayload = JSON.parse(event.data);
      if (payload.dataset_id !== currentDatasetId) {
        return;
      }
      setEmbedding((prev) =>
        prev.map((point) =>
          point.cell_id === payload.cell_id ? { ...point, annotation: payload.label } : point,
        ),
      );
    };
    return () => {
      socket.close();
    };
  }, [currentDatasetId]);

  const annotateCell = useCallback(
    async (cellId: string, label: string) => {
      if (!currentDatasetId) {
        return;
      }
      const response = await fetch(createUrl('/annotations'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ dataset_id: currentDatasetId, cell_id: cellId, label }),
      });
      if (response.ok) {
        setEmbedding((prev) =>
          prev.map((point) => (point.cell_id === cellId ? { ...point, annotation: label } : point)),
        );
      }
    },
    [currentDatasetId],
  );

  const runInference = useCallback(
    async (cellId: string) => {
      if (!currentDatasetId) {
        return;
      }
      const response = await fetch(createUrl('/inference'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ dataset_id: currentDatasetId, cell_id: cellId }),
      });
      if (response.ok) {
        const payload: InferenceResult = await response.json();
        setInference(payload);
      }
    },
    [currentDatasetId],
  );

  const selectDataset = useCallback((datasetId: string) => {
    setCurrentDatasetId(datasetId);
  }, []);

  const selectCell = useCallback((cellId: string | null) => {
    setSelectedCellId(cellId);
  }, []);

  const selectedCell = useMemo(
    () => embedding.find((point) => point.cell_id === selectedCellId) ?? null,
    [embedding, selectedCellId],
  );

  return {
    datasets,
    currentDatasetId,
    embedding,
    metadata,
    mlMetadata,
    inference,
    loading,
    selectedCell,
    selectDataset,
    selectCell,
    annotateCell,
    runInference,
  };
};

export type { BackendState };
