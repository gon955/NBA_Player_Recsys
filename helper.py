from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd

TOTAL_MARKERS = {"2TM","3TM","4TM","5TM","6TM","TOT"}

def season_totals(df):
    keys = ['player_id', 'season']
    df = df.copy()
    
    has_tot = df.groupby(keys)['team'].transform(lambda s: s.isin(TOTAL_MARKERS).any())
    
    keep_totals = df[has_tot & df['team'].isin(TOTAL_MARKERS)]
    keep_others = df[~has_tot & ~df['team'].isin(TOTAL_MARKERS)]
    
    out = pd.concat([keep_totals, keep_others], ignore_index=True)
    out = out.sort_values(keys).drop_duplicates(keys, keep="first").reset_index(drop=True)
    return out

def assign_era(season: int) -> str:
    if 2016 <= season:
        return "2016-present"          # Pace & Space
    if 2008 <= season <= 2015:
        return "2008-2015"             # Early Spacing Adaptation
    if 1999 <= season <= 2007:
        return "1999-2007"             # Post Hand-Check / Zone
    return "1990-1998"                  # Physical Defense


def save_models(models,path="models.joblib"):
    serial = {}
    for era, pack in models.items():
        serial[era] = {
            "model":pack.model,
            "ds":pack.ds,
            "ufeat":pack.ufeat,
            "ifeat":pack.ifeat,
            "items":np.array(pack.items),
            "display_map": getattr(pack,"display_map",None),
        }
    joblib.dump(serial,path)
    
def load_models(path="models.joblib"):
    serial = joblib.load(path)
    models = {}
    for era, d in serial.items():
        models[era] = SimpleNamespace(
            model=d["model"],
            ds=d["ds"],
            ufeat=d["ufeat"],
            ifeat=d["ifeat"],
            items=d["items"],
            display_map=d.get("display_map", {}),
        )
    return models


def era_of(season):
    if 1999 <= season <= 2007:
        return "1999-2007"
    if 2008 <= season <= 2015:
        return "2008-2015"
    if 2016 <= season:
        return "2016-present"
    return None



TEAM_CANON = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets", "BRK": "Brooklyn Nets", "NJN": "New Jersey Nets",
    "NYK": "New York Knicks", "PHI": "Philadelphia 76ers", "TOR": "Toronto Raptors", "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers", "DET": "Detroit Pistons", "IND": "Indiana Pacers", "MIL": "Milwaukee Bucks",
    "MIA": "Miami Heat", "ORL": "Orlando Magic", "WAS": "Washington Wizards", "WSH": "Washington Wizards",
    "DAL": "Dallas Mavericks", "HOU": "Houston Rockets", "SAS": "San Antonio Spurs", "LAL": "Los Angeles Lakers",
    "LAC": "Los Angeles Clippers", "PHX": "Phoenix Suns", "PHO": "Phoenix Suns", "SAC": "Sacramento Kings",
    "GSW": "Golden State Warriors", "POR": "Portland Trail Blazers", "UTA": "Utah Jazz", "DEN": "Denver Nuggets",
    "MIN": "Minnesota Timberwolves", "OKC": "Oklahoma City Thunder", "MEM": "Memphis Grizzlies",
    "NOP": "New Orleans Pelicans", "NOH": "New Orleans Hornets", "NOK": "New Orleans/Oklahoma City Hornets","SEA": "Seattle SuperSonics",
    "VAN": "Vancouver Grizzlies", "WSB": "Washington Bullets","CHH": "Charlotte Hornets", "CHO": "Charlotte Hornets",
    # multi-team markers handled separately
    "2TM": None, "3TM": None, "4TM": None, "5TM": None
}
def canonical_team(abbr: str, season: int):
    if pd.isna(abbr):
        return np.nan
    a = str(abbr).strip()
    if a in {"CHH", "CHO"}:
        return "Charlotte Hornets"
    if a in {"NOK", "NOH"}:
        if season in (2006, 2007):
            return "New Orleans/Oklahoma City Hornets"
        elif 2002 <= season <= 2013:
            return "New Orleans Hornets"
        return "New Orleans Hornets"
    if a == "CHA":
        return "Charlotte Bobcats"
    return TEAM_CANON.get(a, a)