import os
import pickle

import mpmath
import numpy as np
import pytest
from koala.graph_utils import dimerise, remove_vertices
from koala.lattice import Lattice
from koala.pointsets import uniform
from koala.voronization import generate_lattice
from scipy import linalg as la

from dimer_models.dimerisation import dimer_height
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
from dimer_models.lattice_generation import (
    bipartite_squarefull,
    reduce_bipartite,
)

rng = np.random.default_rng(42)

# this text checks that all the kasteleyn stuff is working properly.
# It does this by generating a bunch of lattices and checking that
# the number of dimerisations is correct, and that all the
# dimer-dimer, vison-vison and monomer-monomer correlations are correct.


def _dimer_dimer_prob(dimerisations, chosen_edges):
    e = np.array([dimerisations[:, x] for x in chosen_edges])
    return np.average(e.prod(axis=0))


def _make_and_solve_kasteleyn_tilde(
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


def _kasteleyn_dimer_vals(lattice, edge_list, plaq_list):
    initial_ujk = generate_kasteleyn_orientation(lattice)
    orientations = generate_torus_orientations(lattice, initial_ujk)
    kasteleyn_matrices = np.array([kasteleyn_matrix(lattice, o) for o in orientations])
    pfaffians = np.array([fast_pfaffian_as_mpmath(k) for k in kasteleyn_matrices])
    inverses = np.array(
        [
            np.zeros_like(k) if abs(pf) < 1e-10 else la.inv(k)
            for k, pf in zip(kasteleyn_matrices, pfaffians, strict=True)
        ]
    )
    omega = find_omega(pfaffians)

    n_dimerisations = int(mpmath.nint(abs(sum(omega * pfaffians) / 2)))
    edge_probabilities = torus_dimer_probabilities(
        lattice, kasteleyn_matrices, inverses, pfaffians, omega
    )

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

    visons = np.zeros(len(plaq_list))
    monomers = {}

    # visons and monomers
    for n, p in enumerate(plaq_list):
        kasteleyn_tilde_matrices, pfaffians_tilde, inverses_tilde = _make_and_solve_kasteleyn_tilde(
            lattice, p, initial_ujk
        )

        # visons
        visons[n] = torus_vison_correlation(pfaffians_tilde, omega)

        # monomers
        vertices1 = lattice.plaquettes[p[0]].vertices
        vertices2 = lattice.plaquettes[p[1]].vertices
        all_pairs = np.array(np.meshgrid(vertices1, vertices2))
        all_pairs = all_pairs.reshape(2, -1)

        # the vertex bit is basically free so lets compute it for all the relevant vertex pairs
        for v in all_pairs.T:
            monomers[tuple(v)] = torus_monomer_count(
                kasteleyn_tilde_matrices, inverses_tilde, pfaffians_tilde, v
            )

    return n_dimerisations, edge_probabilities, multi_probabilities, visons, monomers


def _brute_force_dimer_vals(lattice, edge_list, plaq_list, monomers):

    # brute force dimerisations
    brute_dimerisations = dimerise(lattice, -1)
    brute_dimer_probabilities = np.average(brute_dimerisations, axis=0)
    brute_dimer_dimer_probabilities = np.array(
        [_dimer_dimer_prob(brute_dimerisations, c) for c in edge_list]
    )
    brute_n_dimerisations = brute_dimerisations.shape[0]

    # brute force visons
    brute_force_visons = []
    initial_ujk = generate_kasteleyn_orientation(lattice)
    for p in plaq_list:
        monomer_ujk = generate_kasteleyn_orientation(lattice, p, initial_ujk)
        flip_edges = np.where(monomer_ujk - initial_ujk)[0]
        sign = np.prod((-1) ** brute_dimerisations[:, flip_edges], axis=1)
        brute_force_visons.append(abs(np.sum(sign)))
    brute_force_visons = np.array(brute_force_visons)

    brute_force_monomers = {}
    for key in monomers:
        reduced_lattice = remove_vertices(lattice, np.array(key))
        nn = dimerise(reduced_lattice, -1).shape[0]
        brute_force_monomers[key] = nn

    return (
        brute_n_dimerisations,
        brute_dimer_probabilities,
        brute_dimer_dimer_probabilities,
        brute_force_visons,
        brute_force_monomers,
    )


def _check_kast(lattice):

    print(lattice)
    # compute the values with the kasteleyn stuff
    n_pairs = 5
    edge_list = np.array([rng.choice(lattice.n_edges, 2, replace=False) for _ in range(n_pairs)])
    plaq_list = np.array(
        [rng.choice(lattice.n_plaquettes, 2, replace=False) for _ in range(n_pairs)]
    )
    n_dimerisations, edge_probabilities, multi_probabilities, visons, monomers = (
        _kasteleyn_dimer_vals(lattice, edge_list, plaq_list)
    )

    # brute force the values
    (
        brute_n_dimerisations,
        brute_dimer_probabilities,
        brute_dimer_dimer_probabilities,
        brute_force_visons,
        brute_force_monomers,
    ) = _brute_force_dimer_vals(lattice, edge_list, plaq_list, monomers)

    assert n_dimerisations == brute_n_dimerisations
    assert np.allclose(edge_probabilities, brute_dimer_probabilities), (
        f"Edge probabilities diff: {edge_probabilities} vs {brute_dimer_probabilities}"
    )
    assert np.allclose(multi_probabilities, brute_dimer_dimer_probabilities), (
        f"Multi probabilities diff: {multi_probabilities} vs {brute_dimer_dimer_probabilities}"
    )
    assert np.allclose(visons, brute_force_visons), f"Visons diff: {visons} vs {brute_force_visons}"
    for key in monomers:
        assert monomers[key] == brute_force_monomers[key]


def load_all_lattices_from_disk():
    """Helper to un-nest all lattices from the pickle file into a flat list."""
    flat_lattices = []
    if not os.path.exists("tests/test_lattices"):
        create_lattices_for_testing()
    with open("tests/test_lattices", "rb") as f:
        while True:
            try:
                # This unpacks your tuples (l0, l1, l2) and flattens them into one list
                group = pickle.load(f)
                flat_lattices.extend(group)
            except EOFError:
                break
    return flat_lattices


# Load them once at collection time so pytest can dynamically parameterize them
ALL_LATTICES = load_all_lattices_from_disk()


@pytest.mark.parametrize("lattice", ALL_LATTICES)
def test_kasteleyn_number(lattice):
    """Runs as a standalone test for each individual lattice."""
    _check_kast(lattice)


@pytest.mark.parametrize("lattice", ALL_LATTICES)
def test_dimer_heights(lattice):
    """Runs as a standalone test for each individual lattice."""
    plaq_lengths = np.array([p.n_sides for p in lattice.plaquettes])
    regular = np.all(
        lattice.vertices.coordination_numbers == lattice.vertices.coordination_numbers[0]
    )
    if np.sum(plaq_lengths % 2) == 0 and regular:
        dimer = dimerise(lattice)
        dimer_height(lattice, dimer)


def create_lattices_for_testing():
    with open("tests/test_lattices", "wb") as f:
        n = 0
        while n < 3:
            try:
                # bipartite lattice, reduced squares, open boundaries
                # l3 = reduce_bipartite(bipartite_squarefull(80, ensure_true_bipartite=False))
                # l3 = remove_trailing_edges(cut_boundaries(l3))
                # assert l3.n_vertices % 2 == 0
                # _ = ujk_from_fluxes(l3, [-1] * l3.n_plaquettes)

                # general lattice non bipartite, periodic boundaries
                l0 = generate_lattice(uniform(20))
                l1 = bipartite_squarefull(35, ensure_true_bipartite=True)
                l2 = reduce_bipartite(bipartite_squarefull(60, ensure_true_bipartite=True))

                pickle.dump((l0, l1, l2), f)
                n += 1
            except AssertionError:
                pass


if __name__ == "__main__":
    create_lattices_for_testing()
