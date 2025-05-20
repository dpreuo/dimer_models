from koala.pointsets import uniform
from koala.flux_finder import ujk_from_fluxes
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
from dimer_models.kasteleyn import find_kasteleyn_number
import numpy as np
from multiprocessing import Pool
from itertools import repeat
import pickle
import time
from os import listdir
from os.path import isfile, join
import datetime

def main(ns):
    #  = 1000
    open_bcs = False

    force_bipartite = True
    if open_bcs:
        force_bipartite = False
    target = 10

    bc = 'open' if open_bcs else 'closed'
    relevant_directory = f"monomer_results/lattices/{bc}/{ns:05}"
    filenames = sorted([f for f in listdir(relevant_directory) if isfile(join(relevant_directory, f))])
    number_of_valid_lattices = len(filenames)

    while number_of_valid_lattices < target:
        try:    
            lat_square = gu.com_relaxation(
                bipartite_squarefull(ns, ensure_true_bipartite=force_bipartite)
            )
            lat_reduced = gu.com_relaxation(
                reduce_bipartite(
                    bipartite_squarefull(ns * 2, ensure_true_bipartite=force_bipartite)
                )
            )

            if open_bcs:
                lat_square = gu.remove_trailing_edges(gu.cut_boundaries(lat_square))
                lat_reduced = gu.remove_trailing_edges(gu.cut_boundaries(lat_reduced))

            ujk_sq = ujk_from_fluxes(lat_square, [-1]*lat_square.n_plaquettes)
            ujk_reduced = ujk_from_fluxes(lat_reduced, [-1]*lat_reduced.n_plaquettes)

            assert(lat_square.n_vertices%2 == 0)
            assert(lat_reduced.n_vertices%2 == 0)

            with open(f'{relevant_directory}/{number_of_valid_lattices}.pkl', 'wb') as f:
                pickle.dump(lat_square, f)
                pickle.dump(lat_reduced, f)
            
            number_of_valid_lattices += 1

        except:
            pass

    # for x in range(number_of_valid_lattices,10):
    #     lat_square = gu.com_relaxation(
    #         bipartite_squarefull(ns, ensure_true_bipartite=force_bipartite)
    #     )
    #     lat_reduced = gu.com_relaxation(
    #         reduce_bipartite(
    #             bipartite_squarefull(ns * 2, ensure_true_bipartite=force_bipartite)
    #         )
    #     )


if __name__ == "__main__":
    # for ns in [1400, 2000]:
    # for ns in [100, 200]:

    # for ns in [100]:
    for ns in [100, 200, 400, 600, 1000, 1400, 2000, 4000, 6000]:
    # for ns in [2000, 4000, 6000]:
        t1 = time.time()
        main(ns)
        t = time.time() - t1
        print(f"Run:{ns}, time:{str(datetime.timedelta(seconds=t))}")