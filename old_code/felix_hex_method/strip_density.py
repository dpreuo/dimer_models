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


def monomer_corr(lattice, verts):
    monomer_lattice = gu.remove_vertices(lattice, verts)
    monomer_prob = find_kasteleyn_number(monomer_lattice, True)
    normalisation = find_kasteleyn_number(lattice, True)
    p = (monomer_prob[0] / normalisation[0]) * 10 ** (
        monomer_prob[1] - normalisation[1]
    )
    return p

def strip_density(every_other, n,options_to_check = 5):

    center = 0.5 + 1 / (4 * n)
    lattice = eg.honeycomb_lattice(n)
    edge_centers = np.mean(lattice.vertices.positions[lattice.edges.indices], axis=1)

    # kitaev edge labels
    edge_angles = np.arctan2(*lattice.edges.vectors.T) % np.pi
    _, edge_labels = np.unique(np.round(edge_angles, 2), return_inverse=True)

    # Create a mask for edges that are chosen
    vert_edge_centers = edge_centers[edge_labels == 0]
    _, edge_column = np.unique(vert_edge_centers[:, 0], return_inverse=True)

    mask = np.zeros(lattice.n_edges, dtype=bool)
    mask[edge_labels == 0] = edge_column % every_other == 0
    mask = mask * (edge_labels == 0)  # Only keep edges with label 0
    l2 = expand_edges_to_squares(lattice, np.where(mask)[0])
    # l2 = gu.com_relaxation(l2)


    # Find the vertex indices for v1 and list of v2 options
    v1 = np.argmin(la.norm(l2.vertices.positions - np.array([center, 0.5]), axis=1))
    distances_matrix = gu.distance_matrix(l2)[v1]
    pos_messed = l2.vertices.positions + 10 * (1 - distances_matrix[:, None] % 2)

    
    v2_options_horizontal = np.array(
        [
            np.argsort(la.norm(pos_messed - np.array([(center - u) % 1, 0.5]), axis=1))[
                :options_to_check
            ]
            for u in np.linspace(0, 0.5, 2 * n)
        ]
    )
    v2_options_horizontal = np.unique(v2_options_horizontal)
    distances_horizontal = distances_matrix[v2_options_horizontal]

    v2_options_vertical = np.array(
        [
            np.argsort(la.norm(pos_messed - np.array([(center) % 1, 0.5 + u]), axis=1))[
                :options_to_check
            ]
            for u in np.linspace(0, 0.5, 2 * n)
        ]
    )
    v2_options_vertical = np.unique(v2_options_vertical)
    distances_vertical = distances_matrix[v2_options_vertical]

    # Find the Kasteleyn probabilitiesfor the lattice
    dimer_probs = find_local_dimer_probability(l2)

    # find probabilities for monomers
    monomer_probs_hor = []
    for v2 in tqdm(v2_options_horizontal):
        p = monomer_corr(l2, [v1, v2])
        monomer_probs_hor.append(p)

    monomer_probs_vert = []
    for v2 in tqdm(v2_options_vertical):
        p = monomer_corr(l2, [v1, v2])
        monomer_probs_vert.append(p)


    monomer_probs_hor = np.array(monomer_probs_hor)[np.argsort(distances_horizontal)]
    monomer_probs_vert = np.array(monomer_probs_vert)[np.argsort(distances_vertical)]

    distances_vertical = np.sort(distances_vertical)
    distances_horizontal = np.sort(distances_horizontal)


    results = {
        "lattice": l2,
        "v1": v1,
        "probs": dimer_probs,
        "v2_options_horizontal": v2_options_horizontal,
        "distances_horizontal": distances_horizontal,
        "monomer_probs_hor": monomer_probs_hor,
        "v2_options_vertical": v2_options_vertical,
        "distances_vertical": distances_vertical,
        "monomer_probs_vert": monomer_probs_vert,
    }

    return results

if __name__ == "__main__":

    with open("strip_density.pkl", "wb") as f:
        for window in tqdm([1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30]):
            out = strip_density(window, n=30, options_to_check=5)
            pickle.dump(out, f)
