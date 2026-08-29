import type { FeatureContribution, Recommendation } from "@/lib/api";
import { describeFeature } from "@/lib/features";
import ClusterImage from "./ClusterImage";

type FeatureListProps = {
  title: string;
  items?: FeatureContribution[];
  variant: "user" | "item";
};

function FeatureList({ title, items, variant }: FeatureListProps) {
  if (!items || items.length === 0) return null;
  return (
    <div>
      <p className="text-sm font-semibold text-slate-200">{title}</p>
      <ul className="mt-1 flex flex-col gap-2 text-sm text-slate-200">
        {items.map((feat) => (
          <li
            key={`${title}-${feat.feature}`}
            className="rounded-lg bg-slate-900/80 px-3 py-2 ring-1 ring-white/10"
          >
            <p className="font-medium text-slate-100">
              {describeFeature(feat.feature, feat.weight, variant)}
            </p>
            <p className="text-xs text-slate-400">
              Impact: {feat.weight >= 0 ? "+" : ""}
              {feat.weight.toFixed(2)}
            </p>
          </li>
        ))}
      </ul>
    </div>
  );
}

type ExplanationCardProps = {
  explanation: Recommendation["explanation"];
};

export default function ExplanationCard({ explanation }: ExplanationCardProps) {
  if (!explanation) return null;
  const {
    error,
    top_user_features,
    top_item_features,
    team_cluster_image,
    player_cluster_image,
  } = explanation;

  return (
    <div className="mt-3 rounded-xl bg-slate-900/80 px-4 py-3 text-sm text-slate-200 ring-1 ring-white/10">
      {error ? (
        <p className="text-rose-400">Explanation unavailable: {error}</p>
      ) : (
        <div className="grid gap-6 sm:grid-cols-2">
          <div className="space-y-3">
            <FeatureList title="Team fit drivers" items={top_user_features} variant="user" />
            {team_cluster_image && <ClusterImage label="Team cluster" src={team_cluster_image} />}
          </div>
          <div className="space-y-3">
            <FeatureList title="Player traits" items={top_item_features} variant="item" />
            {player_cluster_image && (
              <ClusterImage label="Player cluster" src={player_cluster_image} />
            )}
          </div>
        </div>
      )}
    </div>
  );
}
