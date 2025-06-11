from koala.pointsets import uniform
from koala import graph_utils as gu
from koala import plotting as pl
from koala.voronization import generate_lattice
from koala.lattice import INVALID
from koala import example_graphs as eg
from dimer_models.lattice_generation import (
    bipartite_squarefull,
    reduce_bipartite,
    find_expandable_plaquettes
)
from tqdm import tqdm
from dimer_models.kasteleyn import find_kasteleyn_number, find_local_dimer_probability
import numpy as np
from multiprocessing import Pool
from itertools import repeat
import pickle
import time
import datetime

def _find_correlation(args):

    lattice, vertex_1, vertex_2, distance_between = args

    to_remove = [vertex_1, vertex_2]
    monomer_lattice = gu.remove_vertices(lattice, to_remove)

    if INVALID in lattice.vertices.adjacent_plaquettes[to_remove]:
        return None

    n_dimers = find_kasteleyn_number(monomer_lattice, True)

    return (distance_between, n_dimers)


def main(n_sites, lattice_index, check_n_monomers, n_cores, n_reduction_steps):

    
    bc = 'open' if open_bcs else 'closed'
    relevant_directory = f"monomer_results/lattices/{bc}/{n_sites:05}"
    with open(f"{relevant_directory}/{lattice_index}.pkl", "rb") as f:
        lattice = pickle.load(f)
    
    results = []
    while True:
        options = len(find_expandable_plaquettes(lattice))
        print(f'computing:{options}')


        # find the sites to remove at random
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

        results.append(
            {
                "lattice": lattice,
                "n_squares":options,
                "partition_func": partition_function,
                "distances": dists,
                "mags": mags,
                "powers": pows,
                "dimer_probs": dimer_probs
            }
        )
        if options == 0:
            break
        lattice = reduce_bipartite(lattice, n_reduction_steps)


    with open(f"monomer_results/square_removing/{n_sites:05}_bc{open_bcs}.pkl", "wb") as f:
        pickle.dump(results, f)

    

if __name__ == "__main__":

    open_bcs = False
    lattice_index = 3
    n_sites = 2000
    n_cores = 10
    pairs_to_check = 3000
    plaqs_per_reduction = 10

    main(n_sites, lattice_index, pairs_to_check, n_cores, plaqs_per_reduction)

