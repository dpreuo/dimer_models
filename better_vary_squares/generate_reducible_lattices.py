from koala import graph_utils as gu
from dimer_models.lattice_generation import (
    bipartite_squarefull,
)
from tqdm import trange
import numpy as np
import pickle

def count_objects_in_pickle(filepath):
    count = 0
    try:
        with open(filepath, 'rb') as f:
            while True:
                try:
                    pickle.load(f)
                    count += 1
                except EOFError:
                    break
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
    return count

def generate_reducible_lattice(n_sites):

    while True:
        lattice, og = bipartite_squarefull(n_sites, return_pre_expanded=True)
        if og.n_vertices % 2 == 0:
            break

    dimer = np.where(gu.dimerise(og))[0]

    return lattice, dimer


if __name__ == "__main__":

    for ns in [100, 200, 400, 600, 1000, 1400, 2000, 4000, 6000]:
        relevant_directory = f"monomer_results/reducible/{ns:05}"

        max = 10 - count_objects_in_pickle(f"{relevant_directory}.pkl")
        if max < 0: max = 0
        print(ns, count_objects_in_pickle(f"{relevant_directory}.pkl") ,relevant_directory )
        with open(f"{relevant_directory}.pkl", "ab") as f:

            for x in trange(max):
                out = generate_reducible_lattice(ns)
                pickle.dump(out, f)
