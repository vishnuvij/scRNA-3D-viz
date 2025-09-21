import './App.css';

import { EmbeddingViewer } from './components/EmbeddingViewer';
import { ControlPanel } from './components/ControlPanel';
import { MetadataPanel } from './components/MetadataPanel';
import { useBackendData } from './hooks/useBackendData';

export const App = () => {
  const {
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
  } = useBackendData();

  const activeDataset = datasets.find((dataset) => dataset.dataset_id === currentDatasetId);

  return (
    <div className="app-container">
      <header>
        <h1>scRNA 3D Visualization</h1>
        <p>Interactively explore embeddings, annotate cells, and run ML inference.</p>
      </header>
      <main>
        <section className="visualization">
          <EmbeddingViewer
            points={embedding}
            loading={loading}
            onSelect={selectCell}
            selectedCellId={selectedCell?.cell_id}
          />
        </section>
        <aside className="sidebar">
          <ControlPanel
            datasets={datasets}
            currentDatasetId={currentDatasetId}
            onSelectDataset={selectDataset}
            selectedCell={selectedCell}
            onAnnotate={annotateCell}
            onInference={runInference}
            inference={inference}
            mlMetadata={mlMetadata}
          />
          <MetadataPanel dataset={activeDataset} metadata={metadata} />
        </aside>
      </main>
    </div>
  );
};

export default App;
