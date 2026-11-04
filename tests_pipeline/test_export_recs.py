import pandas as pd

import export_recs


def test_floor_minutes_sums_teams_and_caps_at_a_full_game():
    member = pd.DataFrame({
        "player_id": ["a", "a", "b", "c"],
        "season": [2025, 2025, 2025, 2025],
        "TeamId": [1, 2, 1, 1],
        "share": [0.25, 0.25, 0.5, 1.2],
    })
    minutes = export_recs.floor_minutes(member)
    assert minutes[("a", 2025)] == 24.0
    assert minutes[("b", 2025)] == 24.0
    assert minutes[("c", 2025)] == 48.0
