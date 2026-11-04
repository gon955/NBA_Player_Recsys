/** Each team's color: the page's one accent once that team is selected. They are
 *  team identity colors, darkened where needed so that each one stays readable as
 *  text on the paper background and under white text. */
const TEAM_COLORS: Record<string, string> = {
  "Atlanta Hawks": "#c8102e",
  "Boston Celtics": "#007a33",
  "Brooklyn Nets": "#111111",
  "Charlotte Bobcats": "#0c2340",
  "Charlotte Hornets": "#00707f",
  "Chicago Bulls": "#ce1141",
  "Cleveland Cavaliers": "#860038",
  "Dallas Mavericks": "#00538c",
  "Denver Nuggets": "#0e2240",
  "Detroit Pistons": "#1d42ba",
  "Golden State Warriors": "#1d428a",
  "Houston Rockets": "#ce1141",
  "Indiana Pacers": "#002d62",
  "Los Angeles Clippers": "#c8102e",
  "Los Angeles Lakers": "#552583",
  "Memphis Grizzlies": "#12173f",
  "Miami Heat": "#98002e",
  "Milwaukee Bucks": "#00471b",
  "Minnesota Timberwolves": "#0c2340",
  "New Jersey Nets": "#002a60",
  "New Orleans Hornets": "#00707f",
  "New Orleans Pelicans": "#0c2340",
  "New Orleans/Oklahoma City Hornets": "#00707f",
  "New York Knicks": "#0060a8",
  "Oklahoma City Thunder": "#0064a4",
  "Orlando Magic": "#0064a4",
  "Philadelphia 76ers": "#0060a8",
  "Phoenix Suns": "#1d1160",
  "Portland Trail Blazers": "#c2252b",
  "Sacramento Kings": "#5a2d81",
  "San Antonio Spurs": "#1a1a1a",
  "Seattle SuperSonics": "#00653a",
  "Toronto Raptors": "#ce1141",
  "Utah Jazz": "#002b5c",
  "Vancouver Grizzlies": "#00706b",
  "Washington Wizards": "#002b5c",
};

/** The page's accent before a team is chosen, and for any team not listed. */
export const INK = "#121210";

export function teamColor(team: string) {
  return TEAM_COLORS[team] ?? INK;
}

/** Franchises that no longer exist under that name; the index lists them apart. */
export const FORMER_TEAMS = new Set([
  "Charlotte Bobcats",
  "New Jersey Nets",
  "New Orleans Hornets",
  "New Orleans/Oklahoma City Hornets",
  "Seattle SuperSonics",
  "Vancouver Grizzlies",
]);
