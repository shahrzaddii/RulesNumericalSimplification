"""
Post-processing of one oblique proposition via MILP.

Goal
-----
Given an original proposition

    a^T x >= t

find simplified coefficients and threshold from finite candidate sets that
minimize numerical complexity while preserving the original proposition's
behavior on a reference dataset as much as possible.

This implementation uses:
- candidate values for each coefficient and the threshold
- a linear objective based on a user-defined simplicity cost
- binary mismatch variables with big-M constraints
- either:
    (a) hard limit on number of mismatches, or
    (b) penalty on mismatches in the objective

Dependencies
------------
pip install pulp numpy

Notes
-----
1) This code solves the simplification problem for a SINGLE proposition.
2) Fidelity is defined relative to the ORIGINAL proposition, not the ground truth.
3) Complexity is represented by a candidate-wise cost, which can approximate:
   - string length
   - number of significant digits
   - number of decimal places
   - any custom numerical simplicity score
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import math

import numpy as np
import pulp


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def proposition_labels(X: np.ndarray, a: np.ndarray, t: float) -> np.ndarray:
    """
    Original proposition labels in {-1, +1}.
    +1  if a^T x >= t
    -1  otherwise
    """
    scores = X @ a - t
    return np.where(scores >= 0.0, 1, -1)


def round_significant(x: float, sig: int) -> float:
    """
    Round x to 'sig' significant digits.
    """
    if x == 0.0:
        return 0.0
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)


def truncate_significant(x: float, sig: int) -> float:
    """
    Truncate x to 'sig' significant digits toward zero.
    """
    if x == 0.0:
        return 0.0
    order = int(math.floor(math.log10(abs(x))))
    factor = 10 ** (sig - order - 1)
    return math.trunc(x * factor) / factor


def count_significant_digits_str(x: float) -> int:
    """
    Approximate number of significant digits from a compact decimal string.
    This is a practical proxy, not a canonical mathematical notion.
    """
    s = format(float(x), ".15g")
    s = s.lstrip("-")
    if "e" in s or "E" in s:
        # fallback: exponent notation can complicate string counting
        # use mantissa part only as a rough proxy
        mantissa = s.split("e")[0].split("E")[0]
        s = mantissa
    s = s.replace(".", "")
    s = s.lstrip("0")
    return len(s) if s else 1


def has_decimal_part(x: float, tol: float = 1e-12) -> int:
    nn = int(abs(x - round(x)) > tol)
    return int(abs(x - round(x)) > tol)


def default_complexity_cost(x: float) -> float:
    """
    Example simplicity cost:
        cost = (# significant digits) + 0.5 * indicator(has decimal part)

    You should replace this with your preferred metric if needed.
    """
    decimal_part_cost = has_decimal_part(x)
    significant_digits_cost = count_significant_digits_str(x)
    return float(count_significant_digits_str(x) + 0.5 * has_decimal_part(x))


def build_candidate_values(
    value: float,
    significant_digits: Sequence[int],
    include_original: bool = True,
    use_round: bool = True,
    use_trunc: bool = True,
    use_integr: bool = True,
) -> List[float]:
    """
    Build a finite candidate set for one parameter by applying
    rounding and/or truncation to multiple significant-digit levels.

    Duplicate values are removed.
    """
    cand = set()
    for sig in significant_digits:
        if use_round:
            cand.add(float(round_significant(value, sig)))
        if use_trunc:
            cand.add(float(truncate_significant(value, sig)))
    if include_original:
        cand.add(float(value))
    if use_integr:
        cand.add(float(round(value, 0)))  # also include nearest integer
        cand.add(float(math.floor(value)))  # also include floor
        cand.add(float(math.ceil(value)))  # also include ceil
    return sorted(cand)


def safe_big_m(
    X: np.ndarray,
    coeff_candidates: Sequence[Sequence[float]],
    threshold_candidates: Sequence[float],
    margin: float = 1e-6,
) -> float:
    """
    Compute a valid big-M upper bound for all possible scores:
        s_i = a_tilde^T x_i - t_tilde

    We use:
        |s_i| <= sum_j max|a_j| * |x_ij| + max|t|
    and then take the maximum over samples.
    """
    max_abs_a = np.array([max(abs(v) for v in cand) for cand in coeff_candidates], dtype=float)

    max_abs_t = max(abs(v) for v in threshold_candidates)

    # bounds = np.abs(X) @ max_abs_a + max_abs_t
    # M = float(np.max(bounds) + margin)

    M = np.abs(X) @ max_abs_a + max_abs_t + margin
    return np.maximum(M, 1.0)


# ---------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------

@dataclass
class SimplificationResult:
    simplified_a: np.ndarray
    simplified_t: float
    objective_value: float
    mismatches: int
    original_labels: np.ndarray
    simplified_labels: np.ndarray
    mismatch_indices: np.ndarray
    selected_candidates: Dict[str, float]


# ---------------------------------------------------------------------
# MILP solver
# ---------------------------------------------------------------------

def simplify_proposition_mip(
    X: np.ndarray,
    original_a: np.ndarray,
    original_t: float,
    coeff_candidate_sets: Sequence[Sequence[float]],
    threshold_candidate_set: Sequence[float],
    cost_fn: Callable[[float], float] = default_complexity_cost,
    margin: float = 1e-6,
    max_mismatches: Optional[int] = None,
    mismatch_penalty: Optional[float] = None,
    solver: Optional[pulp.LpSolver] = None,
    verbose: bool = False,
) -> SimplificationResult:
    """
    Solve the simplification MILP for one proposition.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Reference dataset used to preserve the behavior of the original proposition.
    original_a : array, shape (n_features,)
        Original coefficients.
    original_t : float
        Original threshold.
    coeff_candidate_sets : list of candidate lists, length = n_features
        Candidate simplified values for each coefficient.
    threshold_candidate_set : list of floats
        Candidate simplified values for the threshold.
    cost_fn : callable
        Complexity cost per candidate value.
    margin : float
        Small classification margin epsilon.
    max_mismatches : int or None
        If not None, impose sum_i e_i <= max_mismatches.
    mismatch_penalty : float or None
        If not None, add mismatch_penalty * sum_i e_i to objective.
    solver : pulp solver or None
        If None, default CBC solver is used.
    verbose : bool
        Print solver messages if True.

    Returns
    -------
    SimplificationResult
    """
    X = np.asarray(X, dtype=float)
    original_a = np.asarray(original_a, dtype=float)

    n_samples, n_features = X.shape
    assert original_a.shape == (n_features,), "original_a must have shape (n_features,)"
    assert len(coeff_candidate_sets) == n_features, "Need one candidate set per coefficient."
    assert len(threshold_candidate_set) >= 1, "Threshold candidate set cannot be empty."

    # Original labels to preserve
    y = proposition_labels(X, original_a, original_t)  # in {-1, +1}

    # Big-M
    M = safe_big_m(X, coeff_candidate_sets, threshold_candidate_set, margin=margin)

    # Problem
    prob = pulp.LpProblem("Simplify_Oblique_Proposition", pulp.LpMinimize)

    # Selection variables for coefficients
    z_coeff = []
    for j in range(n_features):
        z_j = [
            pulp.LpVariable(f"z_a_{j}_{k}", cat="Binary")
            for k in range(len(coeff_candidate_sets[j]))
        ]
        z_coeff.append(z_j)

    # Selection variables for threshold
    z_t = [
        pulp.LpVariable(f"z_t_{k}", cat="Binary")
        for k in range(len(threshold_candidate_set))
    ]

    # Mismatch variables
    e = [pulp.LpVariable(f"e_{i}", cat="Binary") for i in range(n_samples)]

    # One candidate per parameter
    for j in range(n_features):
        prob += pulp.lpSum(z_coeff[j]) == 1, f"one_candidate_for_coeff_{j}"
    prob += pulp.lpSum(z_t) == 1, "one_candidate_for_threshold"

    # Linear expressions for simplified parameters
    a_tilde_expr = []
    for j in range(n_features):
        expr = pulp.lpSum(
            coeff_candidate_sets[j][k] * z_coeff[j][k]
            for k in range(len(coeff_candidate_sets[j]))
        )
        a_tilde_expr.append(expr)

    t_tilde_expr = pulp.lpSum(
        threshold_candidate_set[k] * z_t[k]
        for k in range(len(threshold_candidate_set))
    )

    # Fidelity constraints:
    # y_i * s_i >= margin - M e_i
    # where s_i = a_tilde^T x_i - t_tilde
    for i in range(n_samples):
        score_expr = pulp.lpSum(a_tilde_expr[j] * X[i, j] for j in range(n_features)) - t_tilde_expr
        # prob += y[i] * score_expr >= margin - M * e[i], f"fidelity_{i}"
        prob += y[i] * score_expr >= margin - M[i] * e[i], f"fidelity_with_bigM_{i}"

    # Optional hard mismatch budget
    if max_mismatches is not None:
        prob += pulp.lpSum(e) <= max_mismatches, "mismatch_budget"

    # Objective = complexity cost + optional mismatch penalty
    complexity_terms = []

    for j in range(n_features):
        complexity_terms.extend(
            cost_fn(coeff_candidate_sets[j][k]) * z_coeff[j][k]
            for k in range(len(coeff_candidate_sets[j]))
        )
    complexity_terms.extend(
        cost_fn(threshold_candidate_set[k]) * z_t[k]
        for k in range(len(threshold_candidate_set))
    )

    objective = pulp.lpSum(complexity_terms)

    if mismatch_penalty is not None:
        objective += mismatch_penalty * pulp.lpSum(e)

    prob += objective

    # Solver
    if solver is None:
        solver = pulp.PULP_CBC_CMD(msg=verbose)

    status = prob.solve(solver)
    if pulp.LpStatus[status] not in ("Optimal", "Feasible"):
        raise RuntimeError(f"MILP did not solve successfully. Status: {pulp.LpStatus[status]}")

    # Recover selected values
    simplified_a = np.zeros(n_features, dtype=float)
    selected_candidates: Dict[str, float] = {}

    for j in range(n_features):
        chosen_idx = None
        for k, var in enumerate(z_coeff[j]):
            if pulp.value(var) > 0.5:
                chosen_idx = k
                break
        if chosen_idx is None:
            raise RuntimeError(f"No candidate selected for coefficient {j}.")
        simplified_a[j] = coeff_candidate_sets[j][chosen_idx]
        selected_candidates[f"a_{j}"] = simplified_a[j]

    chosen_t_idx = None
    for k, var in enumerate(z_t):
        if pulp.value(var) > 0.5:
            chosen_t_idx = k
            break
    if chosen_t_idx is None:
        raise RuntimeError("No candidate selected for threshold.")
    simplified_t = float(threshold_candidate_set[chosen_t_idx])
    selected_candidates["t"] = simplified_t

    simplified_labels = proposition_labels(X, simplified_a, simplified_t)
    mismatch_mask = simplified_labels != y

    return SimplificationResult(
        simplified_a=simplified_a,
        simplified_t=simplified_t,
        objective_value=float(pulp.value(prob.objective)),
        mismatches=int(np.sum(mismatch_mask)),
        original_labels=y,
        simplified_labels=simplified_labels,
        mismatch_indices=np.where(mismatch_mask)[0],
        selected_candidates=selected_candidates,
    )


# ---------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------

# def demo():
#     # -------------------------------------------------------------
#     # Original proposition from your example:
#     #
#     #   -1.1982560070086268 x1 -2.2994697292338455 x2 >= -2.625262397819907
#     #
#     # -------------------------------------------------------------
#     original_a = np.array([-1.1982560070086268, -2.2994697292338455], dtype=float)
#     original_t = -2.625262397819907

#     # Artificial 2D reference samples
#     X = np.array([
#         [0.2, 0.5],
#         [0.5, 0.3],
#         [0.8, 0.1],
#         [0.6, 0.7],
#         [1.0, 0.4],
#         [1.2, 0.8],
#         [1.4, 0.2],
#         [1.5, 1.1],
#         [1.8, 0.6],
#         [2.0, 1.0],
#     ], dtype=float)

#     # Candidate sets built by rounding/truncating to 1..5 significant digits
#     sig_levels = [1, 2, 3, 4, 5]

#     coeff_candidate_sets = [
#         build_candidate_values(original_a[j], sig_levels, include_original=True, use_round=True, use_trunc=True)
#         for j in range(len(original_a))
#     ]
#     threshold_candidate_set = build_candidate_values(original_t, sig_levels, include_original=True, use_round=True, use_trunc=True)

#     print("Candidate sets:")
#     for j, cand in enumerate(coeff_candidate_sets):
#         print(f"  a_{j}: {cand}")
#     print(f"  t: {threshold_candidate_set}")

#     # Solve with mismatch penalty + optional hard cap
#     result = simplify_proposition_mip(
#         X=X,
#         original_a=original_a,
#         original_t=original_t,
#         coeff_candidate_sets=coeff_candidate_sets,
#         threshold_candidate_set=threshold_candidate_set,
#         cost_fn=default_complexity_cost,
#         margin=1e-6,
#         max_mismatches=0,         # exact matching on reference set
#         mismatch_penalty=None,    # not needed if max_mismatches=0
#         verbose=False,
#     )

#     print("\nOriginal proposition:")
#     print(f"  {original_a[0]:.16f} * x1 + {original_a[1]:.16f} * x2 >= {original_t:.16f}")

#     print("\nSimplified proposition:")
#     print(f"  {result.simplified_a[0]} * x1 + {result.simplified_a[1]} * x2 >= {result.simplified_t}")

#     print("\nObjective value:", result.objective_value)
#     print("Mismatches:", result.mismatches)
#     print("Mismatch indices:", result.mismatch_indices.tolist())

#     print("\nSelected candidates:")
#     for k, v in result.selected_candidates.items():
#         print(f"  {k}: {v}")


# if __name__ == "__main__":
#     demo()