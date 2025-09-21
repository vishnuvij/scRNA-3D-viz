import { useMemo } from 'react';

import type { DatasetSummary, MetadataPayload } from '../types';

interface MetadataPanelProps {
  dataset?: DatasetSummary;
  metadata?: MetadataPayload;
}

export const MetadataPanel = ({ dataset, metadata }: MetadataPanelProps) => {
  const clusterCounts = useMemo(() => {
    if (!metadata?.original_clusters) {
      return [];
    }
    const counts = new Map<number, number>();
    metadata.original_clusters.forEach((cluster) => {
      counts.set(cluster, (counts.get(cluster) ?? 0) + 1);
    });
    return Array.from(counts.entries()).sort((a, b) => a[0] - b[0]);
  }, [metadata?.original_clusters]);

  const topGenes = metadata?.gene_columns.slice(0, 12) ?? [];

  return (
    <div className="metadata-panel">
      <section>
        <h2>Dataset summary</h2>
        {dataset ? (
          <ul>
            <li>
              Cells: <strong>{dataset.cell_count}</strong>
            </li>
            <li>
              Genes: <strong>{dataset.gene_count}</strong>
            </li>
          </ul>
        ) : (
          <p>Select a dataset to view summary information.</p>
        )}
      </section>

      <section>
        <h2>Cluster distribution</h2>
        {clusterCounts.length > 0 ? (
          <ul>
            {clusterCounts.map(([cluster, count]) => (
              <li key={cluster}>
                Cluster {cluster}: <strong>{count}</strong> cells
              </li>
            ))}
          </ul>
        ) : (
          <p>Clustering metadata is unavailable for this dataset.</p>
        )}
      </section>

      <section>
        <h2>Gene features</h2>
        {topGenes.length > 0 ? (
          <p>{topGenes.join(', ')}{metadata && metadata.gene_columns.length > topGenes.length ? '…' : ''}</p>
        ) : (
          <p>Gene metadata unavailable.</p>
        )}
      </section>
    </div>
  );
};
