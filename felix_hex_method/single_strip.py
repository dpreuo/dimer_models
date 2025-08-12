from koala.pointsets import uniform
from koala import graph_utils as gu
from koala import plotting as pl
from koala.voronization import generate_lattice
from koala.flux_finder import fluxes_from_ujk, ujk_from_fluxes
from koala.lattice import Lattice
from scipy import linalg as la
from pfapack.pfaffian import pfaffian

import pickle
from koala.graph_color import vertex_color
from koala import example_graphs as eg
from dimer_models.koala_plantri import plantri_to_koala, plantri_generator, read_plantri
from dimer_models.lattice_generation import (
    bipartite_squarefull,
    reduce_bipartite,
)

from dimer_models.lattice_generation import expand_edges_to_squares
from dimer_models.kasteleyn import (
    kasteleyn_matrix,
    find_kasteleyn_number,
    find_local_dimer_probability,
)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from copy import copy


def analyze_monomer_probability(window, n=9):
    center = 0.5 + 1 / (4 * n)
    lattice = eg.honeycomb_lattice(n)
    edge_centers = np.mean(lattice.vertices.positions[lattice.edges.indices], axis=1)

    # kitaev edge labels
    edge_angles = np.arctan2(*lattice.edges.vectors.T) % np.pi
    _, edge_labels = np.unique(np.round(edge_angles, 2), return_inverse=True)

    # choose edges to expand
    mask = np.abs(edge_centers[:, 0] - center) < window
    mask = mask * (edge_labels == 0)  # Only keep edges with label 0
    l2 = expand_edges_to_squares(lattice, np.where(mask)[0])
    # l2 = gu.com_relaxation(l2)

    # choose vertices to remove
    v1 = np.argmin(la.norm(l2.vertices.positions - np.array([center, 0.5]), axis=1))
    distances = gu.distance_matrix(l2)[v1]
    pos_messed = l2.vertices.positions + 10 * (1 - distances[:, None] % 2)
    v2 = np.argmin(la.norm(pos_messed - np.array([(center + 0.5) % 1, 0.5]), axis=1))
    distance = distances[v2]

    # calculate dimer probabilities
    probs = find_local_dimer_probability(l2) - (1 / 3)
    probs = 0.5 * probs / np.max(np.abs(probs))  # Normalize to range [-1, 1]

    # find monomer probability
    monomer_lattice = gu.remove_vertices(l2, [v1, v2])
    monomer_prob = find_kasteleyn_number(monomer_lattice, True)
    normalisation = find_kasteleyn_number(l2, True)
    probability = (monomer_prob[0] / normalisation[0]) * 10 ** (
        monomer_prob[1] - normalisation[1]
    )
    print(f"Monomer probability:, {probability: 4f} at distance {distance:.2f}")

    return {
        "lattice": l2,
        "v1": v1,
        "v2": v2,
        "distance": distance,
        "probs": probs,
        "monomer_prob": probability,
    }


if __name__ == "__main__":

    with open("single_strip.pkl", "wb") as f:
        for window in np.linspace(0.01, 0.5, 10):
            out = analyze_monomer_probability(window, n=10)
            pickle.dump(out, f)
