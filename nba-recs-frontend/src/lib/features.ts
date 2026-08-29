/** Translates raw LightFM feature strings ("pace=med+", "age=U22") into the
 *  plain-English cause the results page shows under each recommendation. */

const USER_DESCRIPTIONS: Record<string, (value: string) => string> = {
  tcluster: (v) => `Team identity is ${v}`,
  pace: (v) => `Team pace target is ${v}`,
  ortg: (v) => `Offensive profile leans ${v}`,
  drtg: (v) => `Defensive profile trends ${v}`,
  era: (v) => `Team era tag is ${v}`,
};

const ITEM_DESCRIPTIONS: Record<string, (value: string) => string> = {
  pcluster: (v) => `Player archetype: ${v}`,
  pos: (v) => `Primary position: ${v}`,
  age: (v) => `Age band: ${v}`,
  era: (v) => `Season belongs to ${v}`,
};

const VALUE_MAPS: Record<string, Record<string, string>> = {
  pace: {
    slow: "Slow Pace",
    "med-": "Moderately Slow Pace",
    "med+": "Moderately Fast Pace",
    fast: "Fast Pace",
  },
  ortg: {
    o_low: "Low Offense",
    "o_mid-": "Below-average Offense",
    "o_mid+": "Above-average Offense",
    o_high: "High Offense",
  },
  drtg: {
    d_best: "Elite Defense",
    d_good: "Good Defense",
    d_ok: "Average Defense",
    d_poor: "Poor Defense",
  },
  age: {
    U22: "22 or younger",
    "23-26": "Ages 23-26",
    "27-30": "Ages 27-30",
    "31-34": "Ages 31-34",
    "35-39": "Ages 35-39",
    "40+": "40+",
  },
};

export function formatValue(value: string) {
  if (!value) return "unknown";
  return value.replace(/_/g, " ");
}

export function formatValueByKey(key: string, rawValue: string) {
  const map = VALUE_MAPS[key];
  if (map && rawValue in map) return map[rawValue];
  return formatValue(rawValue);
}

export function describeFeature(feature: string, weight: number, variant: "user" | "item") {
  const [rawKey, rawValue = ""] = feature.split("=");
  const key = rawKey?.trim() ?? "";
  const value = formatValueByKey(key, rawValue);
  const sentiment = weight >= 0 ? "Boosts fit" : "Adds risk";

  const describe = variant === "user" ? USER_DESCRIPTIONS[key] : ITEM_DESCRIPTIONS[key];
  if (describe) {
    const connector = variant === "user" ? "because" : "thanks to";
    return `${sentiment} ${connector} ${describe(value)}`;
  }

  const fallback = value || key || feature;
  return `${sentiment} via ${formatValue(fallback)}`;
}
