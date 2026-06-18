import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import linalg as la
import copy
from sympy import true
from tqdm import trange
from dataclasses import dataclass, field

from pachner_metropolis_functions import (
    lattice_squenergy,
    boltzmann_probability,
    choose_flip,
    n_squares,
)

from koala.flux_finder import ujk_from_fluxes, fluxes_from_ujk
import mpmath
from dimer_models.kasteleyn import kasteleyn_matrix, fast_pfaffian_as_mpmath
from koala.pointsets import uniform
from koala.voronization import generate_lattice
from koala import graph_utils as gu
from koala import plotting as pl
from koala.lattice import Lattice, INVALID
from koala import example_graphs as eg
from koala import pachner_moves

# mpmath.mp.dps = 50
import pickle

from dimer_models.lattice_generation import bipartite_squarefull





def main():
    pass

if __name__ == "__main__":
    main()
