from koala import graph_utils as gu
from koala.lattice import INVALID
from koala import example_graphs as eg
from dimer_models.lattice_generation import expand_edges_to_squares
from tqdm import tqdm
from dimer_models.kasteleyn import find_kasteleyn_number, find_local_dimer_probability
import numpy as np
from multiprocessing import Pool
from itertools import repeat
import pickle
# import time
# import datetime

# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import koala.plotting as pl


def _random_flip_mask(lattice, n_plaqs):
    mask = np.zeros(lattice.n_edges)
    for _ in range(n_plaqs):
        found = False
        while not found:
            plaq_suggestion = np.random.randint(lattice.n_plaquettes)
            plaq = lattice.plaquettes[plaq_suggestion]
            p_edges = plaq.edges
            choice_1 = p_edges[::2]
            choice_2 = p_edges[1::2]

            exterior = np.array(
                [lattice.vertices.adjacent_edges[x] for x in plaq.vertices]
            ).flatten()
            exterior = np.setxor1d(exterior, plaq.edges)

            if not np.all(mask[exterior] == 0):
                continue

            poss_1 = np.all(mask[choice_1] == 0)
            poss_2 = np.all(mask[choice_2] == 0)

            if poss_1:
                found = True
                mask[choice_2] = 1 - mask[choice_2]
            elif poss_2:
                found = True
                mask[choice_1] = 1 - mask[choice_1]
    return mask


def _find_correlation(args):

    lattice, vertex_1, vertex_2, distance_between = args

    to_remove = [vertex_1, vertex_2]

    # check distance is correct
    d = gu.distance_matrix(lattice)

    assert d[vertex_1, vertex_2] == distance_between, (
        f"Distance between {vertex_1} and {vertex_2} is {d[vertex_1, vertex_2]}, "
        f"not {distance_between}"
    )

    monomer_lattice = gu.remove_vertices(lattice, to_remove)

    if INVALID in lattice.vertices.adjacent_plaquettes[to_remove]:
        return None

    n_dimers = find_kasteleyn_number(monomer_lattice, True)

    return distance_between, n_dimers, to_remove


def main(init_lattice, n_insertions, n_data_points, pairs_to_check, n_cores):

    results = []
    for _ in range(n_data_points):        
        # generate the lattice
        mask = _random_flip_mask(init_lattice, n_insertions)
        lattice = expand_edges_to_squares(init_lattice, np.where(mask)[0])
        print(lattice)

        # find the sites to remove at random for monomer calculation
        distances = gu.distance_matrix(lattice)
        starting_point, ending_point = np.meshgrid(
            np.arange(lattice.n_vertices), np.arange(lattice.n_vertices)
        )
        p = np.random.permutation(lattice.n_vertices**2)
        distances = distances.flatten()[p]
        starting_point = starting_point.flatten()[p]
        ending_point = ending_point.flatten()[p]
        useful_pairs = np.where((distances % 2 == 1) * (distances > 2))

        distances = distances[useful_pairs]
        starting_point = starting_point[useful_pairs]
        ending_point = ending_point[useful_pairs]

        limit = np.min([len(distances), pairs_to_check])

        args = zip(
            repeat(lattice),
            starting_point[:limit],
            ending_point[:limit],
            distances[:limit],
        )

        with Pool(n_cores) as pool:

            out = list(
                tqdm(
                    pool.imap_unordered(_find_correlation, args),
                    total=limit,
                )
            )
        dimer_probs = find_local_dimer_probability(lattice)
        partition_function = find_kasteleyn_number(lattice, True)

        dists = np.array([x[0] for x in out if x is not None])
        mags = np.array([x[1][0] for x in out if x is not None])
        pows = np.array([x[1][1] for x in out if x is not None])
        pairs = np.array([x[2] for x in out if x is not None])

        results.append(
            {
                "lattice": lattice,
                "partition_func": partition_function,
                "distances": dists,
                "pairs": pairs,
                "mags": mags,
                "powers": pows,
                "dimer_probs": dimer_probs,
            }
        )

    with open(
        f"monomer_results/expand_edges/{init_lattice.n_plaquettes:05}_{n_insertions:05}_res.pkl", "wb"
    ) as f:
        pickle.dump(results, f)


if __name__ == "__main__":

    N_DATA_POINTS = 3
    N_EXPANSION_STEPS = 20
    PAIRS_TO_CHECK = 1000
    N_CORES = 7
    LENGTH = 25

    initial_lattice = eg.honeycomb_lattice(LENGTH)
    number_of_insertions = np.round(
        np.linspace(0, initial_lattice.n_plaquettes, 30)
    ).astype(int)

    print(f"loops to make: {N_DATA_POINTS*N_EXPANSION_STEPS}")

    for n_insert in number_of_insertions:
        if n_insert <218:
            continue
        print(n_insert)
        main(initial_lattice, n_insert, N_DATA_POINTS, PAIRS_TO_CHECK, N_CORES)
