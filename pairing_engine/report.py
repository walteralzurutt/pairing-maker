"""
Format-agnostic decision-report helpers: ranking a set of options by
expected value against an equilibrium-playing opponent, with a
near-tie/lower-spread tie-break, plus turning that ranking into
plain-language comparison sentences. Used by every pairing format --
none of this depends on how a format's stages/roles are structured,
only on an already-solved payoff matrix + equilibrium strategies.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

DEFAULT_TIE_TOLERANCE = 0.75  # points; see rank_options' tie-breaking rule below

# Spanish display labels for the report columns below -- shared by
# insight_sentences() and by the UI layer when rendering a report table,
# so the two stay consistent.
COLUMN_LABELS_ES = {
    "option": "Opción",
    "vs_equilibrium_opponent": "Valor esperado",
    "worst_case": "Peor caso",
    "best_case": "Mejor caso",
    "spread": "Volatilidad",
    "equilibrium_weight": "Peso de equilibrio",
    "gap_to_best": "Diferencia con la mejor",
}


def rank_options(payoff_for_me: np.ndarray, opp_equilibrium_strategy: np.ndarray,
                  my_equilibrium_strategy: np.ndarray, option_labels: Sequence,
                  tie_tolerance: float = DEFAULT_TIE_TOLERANCE, tie_break: bool = True) -> pd.DataFrame:
    """Given a payoff matrix already oriented so rows = my options, columns =
    opponent options, and values = MY score (higher is better for me),
    report the expected outcome for each option assuming the opponent
    sticks to their game-theoretically optimal mix, regardless of what I
    pick (`vs_equilibrium_opponent`). This is the most defensible
    "expected value" number -- it equals the overall game value for
    whichever option(s) are actually part of my own equilibrium mix.

    `worst_case`/`best_case` are the SPREAD around that average: the real
    opponent doesn't actually play a probabilistic blend in any one match,
    they commit to one specific option, so the real result of picking this
    option could land anywhere from `worst_case` (if they happen to pick
    their best counter to you) to `best_case` (if they happen to pick
    whatever helps you most) -- `vs_equilibrium_opponent` is the average
    across those outcomes if this exact situation repeated many times
    against an opponent playing their equilibrium mix, not a guarantee
    about this one match. `spread` = best_case - worst_case, a quick sense
    of how much a single match could bounce around that average.

    `my_equilibrium_strategy` is included so you can see which options the
    solver actually recommends mixing between (weight > 0) versus options
    that are strictly worse in equilibrium (weight == 0).

    Tie-breaking: with real data, two options are rarely EXACTLY equal on
    `vs_equilibrium_opponent` -- they're often off by a fraction of a
    point for reasons that don't really matter (rounding, near-symmetric
    matchups). Ranking purely by that value would recommend whichever one
    happens to be a hair ahead, ignoring that it might also be far riskier.
    So: any option within `tie_tolerance` points of the best value is
    treated as "effectively tied" with it, and among that near-best group,
    the recommendation (row 0) is whichever has the LOWEST spread rather
    than the highest raw value. Options clearly outside that tolerance are
    still ranked purely by `vs_equilibrium_opponent`, below the near-best
    group.

    That spread-based override is only valid for describing what WE
    should pick -- we're genuinely free to choose the safer of two
    near-tied options. It is NOT valid for predicting what an opponent
    will do: when two of their options tie on value, the specific
    equilibrium the solver returned already picked how to split weight
    between them (possibly 100/0), and that IS the actual prediction.
    Overriding it with our own risk preference can promote a 0%-weight
    option to row 0, contradicting the `equilibrium_weight` column right
    next to it. Pass `tie_break=False` for opponent-prediction reports to
    sort strictly by `equilibrium_weight` (then `vs_equilibrium_opponent`)
    instead.
    """
    vs_equilibrium_opp = payoff_for_me @ opp_equilibrium_strategy
    worst_case = payoff_for_me.min(axis=1)
    best_case = payoff_for_me.max(axis=1)

    df = pd.DataFrame({
        "option": list(option_labels),
        "vs_equilibrium_opponent": vs_equilibrium_opp,
        "worst_case": worst_case,
        "best_case": best_case,
        "spread": best_case - worst_case,
        "equilibrium_weight": np.round(my_equilibrium_strategy, 4),
    })
    df["gap_to_best"] = (df["vs_equilibrium_opponent"] - df["vs_equilibrium_opponent"].max()).round(2)

    if not tie_break:
        return df.sort_values(["equilibrium_weight", "vs_equilibrium_opponent"],
                               ascending=[False, False]).reset_index(drop=True).round(2)

    best_value = df["vs_equilibrium_opponent"].max()
    near_best_mask = (best_value - df["vs_equilibrium_opponent"]) <= tie_tolerance
    near_best = df[near_best_mask].sort_values(
        ["spread", "vs_equilibrium_opponent"], ascending=[True, False], kind="mergesort")
    rest = df[~near_best_mask].sort_values("vs_equilibrium_opponent", ascending=False)
    return pd.concat([near_best, rest]).reset_index(drop=True).round(2)


def format_option(option) -> str:
    """Display form of a report option: a plain player name, or two names
    joined with 'y' for the swords stage's 2-player tuples."""
    return " y ".join(option) if isinstance(option, tuple) else str(option)


def _options_match(a, b) -> bool:
    """Two options are the same pick regardless of tuple order (a 2-player
    swords choice doesn't care which name came first)."""
    if isinstance(a, tuple) or isinstance(b, tuple):
        return set(a) == set(b)
    return a == b


def explain_naive_comparison(report: pd.DataFrame, naive_option, naive_label: str,
                              option_col: str = "option", metric: str = "vs_equilibrium_opponent") -> str:
    """Always returns a plain-language sentence explaining the top-ranked
    (recommended) option relative to a simpler heuristic pick a human might
    make without game theory (`naive_option`, described by `naive_label`,
    e.g. "el mejor promedio en escudo"). No new solving happens here --
    `naive_option`'s value is just another row already present in `report`.

    - If the recommendation already matches the naive pick, returns a short
      reassurance -- there's nothing counterintuitive to explain.
    - If they differ, explains the recommendation is worth more on `metric`
      even though the naive pick looks better at first glance, framing the
      gap as coming from how the rest of the match plays out from here.
    """
    recommended = report.iloc[0][option_col]
    metric_label = COLUMN_LABELS_ES.get(metric, metric).lower()

    if _options_match(recommended, naive_option):
        return (f"Esta opción también es {naive_label}, así que coincide con lo que la intuición "
                f"sugeriría.")

    naive_rows = report[report[option_col].apply(lambda o: _options_match(o, naive_option))]
    if naive_rows.empty:
        return (f"El modelo recomienda {format_option(recommended)} tras considerar cómo conviene "
                f"jugar el resto de la partida a partir de acá, no solo este enfrentamiento.")

    rec_value = report.iloc[0][metric]
    naive_value = naive_rows.iloc[0][metric]
    return (f"Aunque {format_option(naive_option)} parece la opción más intuitiva ({naive_label}), "
            f"el modelo recomienda {format_option(recommended)} porque vale {rec_value:.2f} puntos en "
            f"{metric_label}, frente a {naive_value:.2f} si eligieras {format_option(naive_option)} -- "
            f"la diferencia viene de cómo conviene jugar el resto de la partida a partir de acá, no "
            f"solo de este enfrentamiento en particular.")


def insight_sentences(report: pd.DataFrame, metric: str = "vs_equilibrium_opponent",
                       option_col: str = "option") -> list:
    """Turn a decision report into plain-language (Spanish) comparison
    sentences, e.g. 'Elegir A1 en vez de A3 vale +10.40 puntos en Valor
    esperado.' Always compares every option to the top-ranked
    (recommended) one.

    Because the top row can be picked for having a lower `spread` rather
    than the strictly highest `vs_equilibrium_opponent` (see the
    tie-breaking rule in rank_options), the gap can occasionally come out
    negative -- that just means the recommended pick is marginally lower
    on this metric but was preferred for being less volatile.
    """
    metric_label = COLUMN_LABELS_ES.get(metric, metric)
    best_label = report.iloc[0][option_col]
    best_value = report.iloc[0][metric]
    sentences = []
    for _, row in report.iterrows():
        if row[option_col] == best_label:
            continue
        gap = best_value - row[metric]
        sentences.append(
            f"Elegir {best_label} en vez de {row[option_col]} vale "
            f"{gap:+.2f} puntos en {metric_label}."
        )
    return sentences
