from koala import graph_utils as gu
from scipy import linalg as la
import pickle
from dimer_models.lattice_generation import expand_edges_to_squares
from dimer_models.kasteleyn import (
    find_kasteleyn_number,
    find_local_dimer_probability,
)
from koala import example_graphs as eg
import numpy as np
from tqdm import tqdm

def analyze_monomer_probability(window, n):
    center = 0.5 + 1 / (4 * n)
    lattice = eg.honeycomb_lattice(n)
    edge_centers = np.mean(lattice.vertices.positions[lattice.edges.indices], axis=1)

    # kitaev edge labels
    edge_angles = np.arctan2(*lattice.edges.vectors.T) % np.pi
    _, edge_labels = np.unique(np.round(edge_angles, 2), return_inverse=True)

    # Create a mask for edges that are within the specified window around the center
    mask = np.abs(edge_centers[:, 0] - center) < window
    mask = mask * (edge_labels == 0)  # Only keep edges with label 0
    l2 = expand_edges_to_squares(lattice, np.where(mask)[0])
    l2 = gu.com_relaxation(l2)

    # Find the vertex indices for v1 and list of v2 options
    v1 = np.argmin(la.norm(l2.vertices.positions - np.array([center, 0.5]), axis=1))
    distances = gu.distance_matrix(l2)[v1]
    pos_messed = l2.vertices.positions + 10 * (1 - distances[:, None] % 2)
    v2_options = [
        np.argmin(la.norm(pos_messed - np.array([(center - u) % 1, 0.5]), axis=1))
        for u in np.linspace(0, 0.5, 2 * n)
    ]
    v2_options = np.unique(v2_options)
    distances = distances[v2_options]

    # Find the Kasteleyn probabilitiesfor the lattice
    dimer_probs = find_local_dimer_probability(l2)

    # find probabilities for monomers
    monomer_probs = []
    for v2 in v2_options:
        monomer_lattice = gu.remove_vertices(l2, [v1, v2])
        monomer_prob = find_kasteleyn_number(monomer_lattice, True)
        normalisation = find_kasteleyn_number(l2, True)
        p = (monomer_prob[0] / normalisation[0]) * 10 ** (
            monomer_prob[1] - normalisation[1]
        )
        monomer_probs.append(p)
    monomer_probs = np.array(monomer_probs)


    return {
        "lattice": l2,
        "v1": v1,
        "v2_options": v2_options,
        "distances": distances,
        "probs": dimer_probs,
        "monomer_prob": monomer_probs,
    }


if __name__ == "__main__":

    with open("single_strip.pkl", "wb") as f:
        for window in tqdm(np.linspace(0.01, 0.5, 20)):
            out = analyze_monomer_probability(window, n=30)
            pickle.dump(out, f)
