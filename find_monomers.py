from koala.pointsets import uniform
from koala import graph_utils as gu
from koala import plotting as pl
from koala.voronization import generate_lattice
from koala.lattice import INVALID
from koala import example_graphs as eg
from dimer_models.lattice_generation import (
    bipartite_squarefull,
    reduce_bipartite,
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


def main(ns, lattice_index, open_bcs, check_n_monomers, n_cores):

    print(f"\n Running {ns}")
    bc = 'open' if open_bcs else 'closed'
    relevant_directory = f"monomer_results/lattices/{bc}/{ns:05}"

    with open(f"{relevant_directory}/{lattice_index}.pkl", "rb") as f:
        lat_square = pickle.load(f)
        lat_reduced = pickle.load(f)


    lattices = [lat_square, lat_reduced]
    bipartite = [True, True]

    print(lattices)

    results = []
    for b, lattice in zip(bipartite, lattices):

        distances = gu.distance_matrix(lattice)
        starting_point, ending_point = np.meshgrid(
            np.arange(lattice.n_vertices), np.arange(lattice.n_vertices)
        )
        p = np.random.permutation(lattice.n_vertices**2)
        distances = distances.flatten()[p]
        starting_point = starting_point.flatten()[p]
        ending_point = ending_point.flatten()[p]

        if b:
            useful_pairs = np.where((distances % 2 == 1) * (distances > 2))
        else:
            useful_sites = np.where(distances > 2)

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
                "partition_func": partition_function,
                "distances": dists,
                "mags": mags,
                "powers": pows,
                "dimer_probs": dimer_probs
            }
        )

    with open(f"monomer_results/res_w_inv/n{ns:05}_bc{open_bcs}.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":

    open_bcs = False
    lattice_index = 3

    # for ns in [100, 200]:
    for ns in [100, 200, 400, 600, 1000, 1400, 2000, 4000, 6000]:
    # for ns in [2000, 4000, 6000]:
        t1 = time.time()
        main(ns, lattice_index, open_bcs, 3000, 7)
        t = time.time() - t1
        print(f"Run:{ns}, time:{str(datetime.timedelta(seconds=t))}")
