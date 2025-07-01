from koala.voronization import generate_lattice
from koala.pointsets import uniform
from dimer_models.kasteleyn import find_kasteleyn_number, find_local_dimer_probability
from koala import graph_utils as gu
from dimer_models.lattice_generation import (
    bipartite_squarefull,
    reduce_bipartite,
)
from dimer_models.dimerisation import dimer_height
from koala.flux_finder import ujk_from_fluxes
from koala.graph_utils import dimerise
import pickle
import numpy as np

def _check_probabilities_sum(lattice, probs):
    sums = np.array([np.sum(probs[v]) for v in lattice.vertices.adjacent_edges])
    assert np.allclose(sums, 1)

def _check_kast(lattice):

    all_dimers = gu.dimerise(lattice, -1)
    n_dimers = all_dimers.shape[0]

    n_kast = find_kasteleyn_number(lattice)
    n_kast_log = find_kasteleyn_number(lattice, True)

    if n_dimers > 0:
        dimer_probabilities = np.mean(all_dimers, axis=0)
        kast_probs = find_local_dimer_probability(lattice)
        _check_probabilities_sum(lattice, kast_probs)
        assert np.allclose(dimer_probabilities, kast_probs)

    assert n_dimers == n_kast
    assert round(n_kast_log[0] * 10 ** n_kast_log[1]) == n_kast


def test_kasteleyn_number():
    with open("tests/test_lattices", "rb") as f:
        while True:
            try:
                lattices = pickle.load(f)
                [_check_kast(l) for l in lattices]

            except EOFError:
                break


def test_dimer_heights():
    with open("tests/test_lattices", "rb") as f:
        while True:
            try:
                lattices = pickle.load(f)
                for lattice in lattices:

                    plaq_lengths = np.array([p.n_sides for p in lattice.plaquettes])
                    regular = np.all(lattice.vertices.coordination_numbers == lattice.vertices.coordination_numbers[0])
                    if np.sum(plaq_lengths%2) == 0 and regular:
                        dimer = dimerise(lattice)
                        dimer_height(lattice,dimer)

            except EOFError:
                break



def create_lattices_for_testing():
    with open("tests/test_lattices", "wb") as f:
        n = 0
        while n < 10:
            try:
                
                # bipartite lattice, reduced squares, open boundaries
                l3 = reduce_bipartite(
                    bipartite_squarefull(80, ensure_true_bipartite=False)
                )
                l3 = gu.remove_trailing_edges(gu.cut_boundaries(l3))
                assert l3.n_vertices % 2 == 0
                ujk_reduced = ujk_from_fluxes(l3, [-1] * l3.n_plaquettes)

                # general lattice non bipartite
                l0 = generate_lattice(uniform(20))

                # full of squares
                l1 = bipartite_squarefull(35, ensure_true_bipartite=True)

                # bipartite lattice, reduced squares, periodic boundaries
                l2 = reduce_bipartite(
                    bipartite_squarefull(60, ensure_true_bipartite=True)
                )

                print(f"{l0}\n{l1}\n{l2}\n{l3}\n\n")

                pickle.dump((l0, l1, l2, l3), f)
                n += 1
            except:
                pass


if __name__ == "__main__":
    create_lattices_for_testing()

