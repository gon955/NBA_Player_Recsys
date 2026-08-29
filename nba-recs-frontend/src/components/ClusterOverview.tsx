import { clusterPcaUrl, type ClusterKind, type ClusterSummaryRecord } from "@/lib/api";

type ClusterSummaryCardProps = {
  title: string;
  kind: ClusterKind;
  records?: ClusterSummaryRecord[];
};

function ClusterSummaryCard({ title, kind, records }: ClusterSummaryCardProps) {
  if (!records || records.length === 0) {
    return (
      <div className="rounded-2xl border border-white/5 bg-slate-900/70 px-4 py-4 shadow-lg">
        <p className="text-slate-300 text-sm">{title}: no data</p>
      </div>
    );
  }

  const maxCount = Math.max(...records.map((r) => r.count));
  const eras = Array.from(new Set(records.map((r) => r.era)));

  return (
    <div className="rounded-2xl border border-white/5 bg-slate-900/70 px-4 py-4 shadow-lg space-y-6">
      <h3 className="text-lg font-semibold text-slate-100">{title}</h3>
      {eras.map((era) => (
        <div key={era} className="space-y-3">
          <div className="flex items-center justify-between">
            <p className="font-semibold text-slate-200">{era}</p>
            <a
              href={clusterPcaUrl(kind, era)}
              target="_blank"
              rel="noreferrer"
              className="text-xs text-indigo-300 hover:text-indigo-200"
            >
              View PCA
            </a>
          </div>
          <div className="space-y-2">
            {records
              .filter((r) => r.era === era)
              .map((cluster) => (
                <div key={cluster.cluster_label} className="space-y-1">
                  <div className="flex items-center justify-between text-xs text-slate-300">
                    <span>{cluster.cluster_label}</span>
                    <span>{cluster.count}</span>
                  </div>
                  <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-indigo-500"
                      style={{ width: `${(cluster.count / maxCount) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
          </div>
        </div>
      ))}
    </div>
  );
}

type ClusterOverviewProps = {
  loading: boolean;
  error: string | null;
  data: { player?: ClusterSummaryRecord[]; team?: ClusterSummaryRecord[] };
};

export default function ClusterOverview({ loading, error, data }: ClusterOverviewProps) {
  return (
    <section className="mt-8 space-y-8">
      <div>
        <h2 className="text-xl font-semibold text-slate-100">Cluster Overview</h2>
        <p className="text-slate-400 text-sm mt-1">
          Dominant clusters per era with population counts.
        </p>
      </div>
      {loading && <p className="text-slate-400">Loading cluster data…</p>}
      {error && <p className="text-rose-400">{error}</p>}
      {!loading && !error && (
        <div className="space-y-6">
          <ClusterSummaryCard title="Player clusters" kind="player" records={data.player} />
          <ClusterSummaryCard title="Team clusters" kind="team" records={data.team} />
        </div>
      )}
    </section>
  );
}
