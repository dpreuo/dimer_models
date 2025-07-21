from koala.pointsets import uniform
from koala import graph_utils as gu
from koala import plotting as pl
from koala.voronization import generate_lattice
from koala.lattice import INVALID
from koala import example_graphs as eg
from dimer_models.lattice_generation import (
    expand_edges
)
from tqdm import tqdm
from dimer_models.kasteleyn import find_kasteleyn_number, find_local_dimer_probability
import numpy as np
from multiprocessing import Pool
from itertools import repeat
import pickle
import time
import datetime

import matplotlib.pyplot as plt
import matplotlib as mpl
import koala.plotting as pl

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

    return  distance_between, n_dimers, to_remove


def main(n_sites, lattice_index, check_n_monomers, n_cores, n_data_points):

    relevant_directory = f"monomer_results/reducible/{n_sites:05}"
    with open(f"{relevant_directory}.pkl", "rb") as f:
        try:
            [pickle.load(f) for _ in range(lattice_index-1)]
            lattice_original, dimer = pickle.load(f)
        except EOFError:
            raise Exception('not enough lattices xoxo')


    results = []
    for n in range(n_data_points):
        f =  n / n_data_points
        f = f*(2-f)
        subset = np.random.choice(dimer, np.round(f*len(dimer)).astype(int) , replace=False)
        lattice = expand_edges(lattice_original, subset)

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

        limit = np.min([len(distances), check_n_monomers])

        args = zip(
            repeat(lattice),
            starting_point[:limit],
            ending_point[:limit],
            distances[:limit],
        )

        print(lattice)
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
                "dimer_probs": dimer_probs
            }
        )



    with open(f"monomer_results/reducible/res/{n_sites:05}_{lattice_index}_res.pkl", "wb") as f:
        pickle.dump(results, f)

    

if __name__ == "__main__":
 
    # n_sites = 400
    lattice_index = 2
    n_cores = 10
    n_data_points = 15
    pairs_to_check = 3000

    for n_sites in [2000]:
        print(f"Starting calculations for {n_sites} sites, lattice index {lattice_index} at {datetime.datetime.now()}")
        main(n_sites, lattice_index, pairs_to_check, n_cores,n_data_points)
        print(f"Finished calculations for {n_sites} sites, lattice index {lattice_index} at {datetime.datetime.now()}")



