import logging
import time
from functools import wraps

import mpmath
import numpy as np
from koala.graph_color import vertex_color
from koala.lattice import Lattice
from scipy import linalg as la

from dimer_models.kasteleyn import (
    fast_pfaffian_as_mpmath,
    find_omega,
    generate_kasteleyn_orientation,
    generate_torus_orientations,
    kasteleyn_matrix,
    torus_dimer_correlation,
    torus_dimer_probabilities,
    torus_monomer_count,
    torus_vison_correlation,
)

logger = logging.getLogger(__name__)


class StepTimer:
    def __init__(self, name=None):
        self.start = time.perf_counter()
        self.last = self.start
        self.name = name or "function"

    def mark(self, msg):
        now = time.perf_counter()
        logger.info(
            f"[{self.name}] +{now - self.last:.4f}s | total {now - self.start:.4f}s | {msg}"
        )
        self.last = now


def timed_steps(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        timer = StepTimer(func.__name__)
        return func(*args, timer=timer, **kwargs)

    return wrapper


def make_and_solve_kasteleyn_tilde(
    lattice: Lattice,
    chosen_plaquettes: np.ndarray,
    initial_ujk: np.ndarray,
):
    monomer_ujk = generate_kasteleyn_orientation(lattice, chosen_plaquettes, initial_ujk)
    monomer_orientations = generate_torus_orientations(lattice, monomer_ujk)
    kasteleyn_tilde_matrices = np.array(
        [kasteleyn_matrix(lattice, o) for o in monomer_orientations]
    )
    pfaffians_tilde = np.array([fast_pfaffian_as_mpmath(k) for k in kasteleyn_tilde_matrices])
    inverses_tilde = np.array(
        [
            np.zeros_like(k) if abs(pf) < 1e-10 else la.inv(k)
            for k, pf in zip(kasteleyn_tilde_matrices, pfaffians_tilde, strict=True)
        ]
    )
    return kasteleyn_tilde_matrices, pfaffians_tilde, inverses_tilde


@timed_steps
def solve_lattice_data(
    lattice: Lattice,
    edge_list: np.ndarray,
    plaquette_list: np.ndarray,
    timer=None,
):

    timer.mark("START")
    # calculate the non-reduced Kasteleyn data
    initial_ujk = generate_kasteleyn_orientation(lattice)
    orientations = generate_torus_orientations(lattice, initial_ujk)
    kasteleyn_matrices = np.array([kasteleyn_matrix(lattice, o) for o in orientations])
    pfaffians = np.array([fast_pfaffian_as_mpmath(k) for k in kasteleyn_matrices])
    inverses = np.array([la.inv(k) for k in kasteleyn_matrices])
    omega = find_omega(pfaffians)

    # dimerisations and probabilities
    n_dimerisations = int(mpmath.nint(abs(sum(omega * pfaffians) / 2)))
    edge_probabilities = torus_dimer_probabilities(
        lattice, kasteleyn_matrices, inverses, pfaffians, omega
    )
    timer.mark("Calculated dimerisations and probs")

    # dimer dimer probabilities
    def compute_probability(edges):
        return torus_dimer_correlation(
            lattice,
            kasteleyn_matrices,
            inverses,
            pfaffians,
            omega,
            edges,
        )

    multi_probabilities = np.fromiter(
        (compute_probability(edges) for edges in edge_list), dtype=float
    )

    timer.mark("Calculated dimer corrs")
    # find a bipartition of the vertices
    bipartition = vertex_color(lattice.edges.indices, 2)[1]

    visons = np.zeros(len(plaquette_list))
    monomers = {}
    # visons and monomers

    next_mark = 10
    for n, p in enumerate(plaquette_list):
        percent = 100 * (n + 1) / len(plaquette_list)
        if percent >= next_mark:
            timer.mark(f"{next_mark}% complete")
            next_mark += 10

        kasteleyn_tilde_matrices, pfaffians_tilde, inverses_tilde = make_and_solve_kasteleyn_tilde(
            lattice, p, initial_ujk
        )

        # visons
        visons[n] = torus_vison_correlation(pfaffians_tilde, omega)

        # monomers
        # first find a pair of vertices one from each plaquette
        vertices1 = lattice.plaquettes[p[0]].vertices
        vertices2 = lattice.plaquettes[p[1]].vertices
        all_pairs = np.array(np.meshgrid(vertices1, vertices2))
        all_pairs = all_pairs.reshape(2, -1)

        # the vertex bit is basically free so lets compute it for all the relevant vertex pairs
        for v in all_pairs.T:
            if np.sum(bipartition[v]) != 1:
                continue
            if tuple(v) in monomers:
                # uncomment this if you want to check that the monomers are correct by ensuring it
                # always gives the same value using any plaq
                # guess = torus_monomer_count(
                #     kasteleyn_tilde_matrices, inverses_tilde, pfaffians_tilde, v
                # )
                # assert guess == monomers[tuple(v)]

                continue
            monomers[tuple(v)] = torus_monomer_count(
                kasteleyn_tilde_matrices, inverses_tilde, pfaffians_tilde, v
            )

    timer.mark("Finished")

    results = {
        "n_dimerisations": n_dimerisations,
        "edge_probabilities": edge_probabilities,
        "multi_probabilities": multi_probabilities,
        "visons": visons,
        "monomers": monomers,
    }

    return results
