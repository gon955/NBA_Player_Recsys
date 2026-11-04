/** The three eras the Ask tab's filter (and the index behind it) is split by. */
export const ERAS = ["1999-2007", "2008-2015", "2016-present"] as const;

export type Era = (typeof ERAS)[number];

export const DEFAULT_ERA: Era = "2016-present";

export function getEraFromYear(year: number): Era {
  if (year >= 2016) return "2016-present";
  if (year >= 2008) return "2008-2015";
  return "1999-2007";
}
