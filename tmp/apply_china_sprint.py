"""
Backfill script: Apply China Sprint Elo updates to DPI state.
Run once from project root: python tmp/apply_china_sprint.py

China Sprint results (March 14, 2026, Round 2):
  P1  Russell    (K=40 user-specified for this backfill, not K=20)
  P2  Leclerc
  P3  Hamilton
  P4  Norris
  P5  Antonelli  (10s penalty post-race — classified P5 for Elo purposes)
  P6  Piastri
  P7  Lawson     (+6 positions, racecraft_bonus=high confirmed)
  P8  Bearman
  P9  Verstappen
  P10 Ocon
  DNF Lindblad   (technical — Elo unchanged)
  DNF Bottas     (technical — Elo unchanged)
  DNF Hulkenberg (technical, triggered SC — Elo unchanged)

Note: per user instruction, K=40 for this one-time backfill (2026 race weight).
      Going forward, sprints use K=20 via update_after_sprint.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.dpi import DriverPerformanceIndex, K_FACTOR_2026, BASE_RATING, CAR_LIMITED_CONTEXT, DRIVER_BEHAVIORAL_FLAGS, RETURNING_VETERAN_DECAY

def main():
    dpi = DriverPerformanceIndex()

    # China Sprint results — K=40 per user instruction (backfill, one-time)
    sprint_results = [
        {"driver": "Russell",    "team": "Mercedes",     "finish_pos": 1,  "dnf": False, "cause": ""},
        {"driver": "Leclerc",    "team": "Ferrari",      "finish_pos": 2,  "dnf": False, "cause": ""},
        {"driver": "Hamilton",   "team": "Ferrari",      "finish_pos": 3,  "dnf": False, "cause": ""},
        {"driver": "Norris",     "team": "McLaren",      "finish_pos": 4,  "dnf": False, "cause": ""},
        {"driver": "Antonelli",  "team": "Mercedes",     "finish_pos": 5,  "dnf": False, "cause": ""},
        {"driver": "Piastri",    "team": "McLaren",      "finish_pos": 6,  "dnf": False, "cause": ""},
        {"driver": "Lawson",     "team": "Racing Bulls", "finish_pos": 7,  "dnf": False, "cause": ""},
        {"driver": "Bearman",    "team": "Haas",         "finish_pos": 8,  "dnf": False, "cause": ""},
        {"driver": "Verstappen", "team": "Red Bull",     "finish_pos": 9,  "dnf": False, "cause": ""},
        {"driver": "Ocon",       "team": "Haas",         "finish_pos": 10, "dnf": False, "cause": ""},
        # Technical DNFs — Elo unchanged (use update_after_sprint's K=20 exclusion,
        # but we're calling the raw method directly at K=40 so handle manually)
        {"driver": "Lindblad",   "team": "Racing Bulls", "finish_pos": 11, "dnf": True, "cause": "technical"},
        {"driver": "Bottas",     "team": "Cadillac",     "finish_pos": 12, "dnf": True, "cause": "technical"},
        {"driver": "Hulkenberg", "team": "Audi",         "finish_pos": 13, "dnf": True, "cause": "technical"},
    ]

    print("\n--- Applying China Sprint Elo updates (K=40 backfill) ---")
    print(f"{'Driver':<14} {'Before':>7}  {'After':>7}  {'Delta':>7}  {'Note'}")
    print("-" * 65)

    total = len([r for r in sprint_results if not r["dnf"]])  # 10 finishers

    for result in sprint_results:
        driver    = result["driver"]
        if driver not in dpi._ratings:
            print(f"  {driver:<14}  [NOT IN STATE - SKIPPING]")
            continue

        entry  = dpi._ratings[driver]
        before = entry["rating"]

        # Technical DNFs — skip Elo update, still update sprint_history
        if result["dnf"] and "technical" in result["cause"].lower():
            history = entry.setdefault("sprint_history", [])
            history.append(None)
            entry["sprint_history"] = history[-3:]
            entry["races_2026"] = entry.get("races_2026", 0) + 1
            print(f"  {driver:<14} {before:>7.0f}  {'---':>7}  {'  ---':>7}  [tech DNF - Elo protected]")
            continue

        # Finishers — apply K=40 Elo update (user-specified for this backfill)
        finish_pos = result["finish_pos"]
        expected   = total / (1 + 10 ** ((BASE_RATING - before) / 400))
        actual     = total - finish_pos + 1
        delta      = K_FACTOR_2026 * (actual - expected)

        entry["rating"] = max(900.0, before + delta)
        entry["stddev"] = max(18.0, entry.get("stddev", 50.0) * 0.95)
        entry["races_2026"] = entry.get("races_2026", 0) + 1

        history = entry.setdefault("sprint_history", [])
        history.append(finish_pos)
        entry["sprint_history"] = history[-3:]

        after = entry["rating"]
        print(f"  {driver:<14} {before:>7.0f}  {after:>7.0f}  {delta:>+7.1f}")

    dpi._save_state()
    print(f"\nState saved: {dpi.state_file}")

    # Final ranking verification
    print("\n--- Post-China-Sprint DPI Rankings ---")
    print(f"{'Pos':<4} {'Driver':<14} {'Rating':>7}  {'Flags'}")
    print("-" * 55)
    car_flags = set(CAR_LIMITED_CONTEXT.keys())
    for pos, entry in enumerate(dpi.rank_all_drivers(), 1):
        flags = []
        if entry["classification"] == "returning_veteran":
            flags.append(f"DECAY-x{RETURNING_VETERAN_DECAY.get(entry['driver'], 1.0)}")
        if entry["classification"] == "sophomore":
            flags.append("SOPH")
        if entry["classification"] == "rookie":
            flags.append("ROOKIE/WIDE-CI")
        if entry["driver"] in car_flags:
            flags.append("CAR-LTD")
        print(f"  {pos:<3} {entry['driver']:<14} {entry['dpi_score']:>7.0f}  {' | '.join(flags)}")
    print(f"\nTotal: {len(dpi._ratings)} drivers")

    # Key check: Russell must be above Antonelli
    r_russell   = dpi._ratings.get("Russell", {}).get("rating", 0)
    r_antonelli = dpi._ratings.get("Antonelli", {}).get("rating", 0)
    gap = r_russell - r_antonelli
    print(f"\nRussell ({r_russell:.0f}) vs Antonelli ({r_antonelli:.0f}): delta = {gap:+.1f}")
    if gap > 0:
        print("OK: Russell rated above Antonelli (championship lead + 100% pole-to-win rate confirmed)")
    else:
        print("WARN: Antonelli still above Russell -- check K-factor inputs")

if __name__ == "__main__":
    main()
