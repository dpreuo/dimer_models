from attr import dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from pachner_metropolis_functions import (
    lattice_squenergy,
    plus_change_squenergy,
    minus_change_squenergy,
    find_plus_candidate,
    find_minus_candidate,
    boltzmann_probability,
    choose_flip,
)

from koala.pointsets import uniform
from koala.voronization import generate_lattice
from koala import graph_utils as gu
from koala import plotting as pl
from koala.lattice import Lattice, INVALID
from koala import example_graphs as eg
from koala import pachner_moves
import pickle

FILENAME = "pachner_lattices_100_3823.pkl"

@dataclass(frozen=True)
class DimerParams():
    pass


def main():
    with open(FILENAME, "rb") as f:
        data = pickle.load(f)
    all_lattices = data["lattices"]
    energies = data["energies"]
    total_plaqs = data["total_plaqs"]
    n_squares = data["n_squares"]
    lattice_params = data["params"]


if __name__ == "__main__":
    main()
