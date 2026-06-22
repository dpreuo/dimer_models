import copy
import pickle
from dataclasses import dataclass, field

import numpy as np
from koala import graph_utils as gu
from pachner_metropolis_functions import (
    boltzmann_probability,
    choose_flip,
    lattice_squenergy,
    n_squares,
)
from tqdm import trange

from dimer_models.lattice_generation import bipartite_squarefull


@dataclass(frozen=True)
class Parameters:
    n_sites: int
    k_steps: int
    beta_range: tuple

    # optional ones below
    f_range: int = 10
    only_check_squares: bool = True
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    output_location: str = "pachner/lattices/"

    @property
    def attempt_limit(self):
        return self.k_steps * 5

    @property
    def output_file_name(self):
        h = hash(
            (self.n_sites, self.k_steps, self.beta_range, self.f_range, self.only_check_squares)
        )
        return f"pachner_lattices_{self.n_sites}_{str(abs(h))[:4]}"


def main(params: Parameters):

    # generate the starting lattice
    initial_lattice = bipartite_squarefull(params.n_sites)

    lattice = copy.copy(initial_lattice)
    n_each_flip = {"plus": 0, "minus": 0}
    beta_range = np.linspace(*params.beta_range, params.k_steps)
    rng = params.rng

    all_lattices = [initial_lattice.__getstate__()]
    energies = [lattice_squenergy(initial_lattice)]
    total_plaqs = [initial_lattice.n_plaquettes]
    n_squares_list = [n_squares(initial_lattice)]

    tr = trange(params.k_steps)
    for k in tr:
        beta = beta_range[k]  # set inverse temperature

        # keep trying to propose a move until we get an acceptable one

        no_options = True

        for _ in range(params.attempt_limit):
            # propose a flip move
            cand, change_energy, move, flip_type = choose_flip(n_each_flip, f_range=params.f_range)
            candidate = cand(lattice)

            # check energy
            delta_energy = change_energy(lattice, *candidate)
            p = boltzmann_probability(delta_energy, beta)

            if rng.random() < p:
                new_lattice = move(lattice, *candidate)
                new_lattice = gu.tutte_embedding(
                    new_lattice
                )  # relax the lattice to ensure all convex

                # check if the new lattice is broken
                not_broken = new_lattice.n_plaquettes == new_lattice.n_vertices // 2
                if not_broken:
                    no_options = False
                    break

        energy = lattice_squenergy(new_lattice)
        n_squares_new = n_squares(new_lattice)
        tr.set_description(f"Proportion: {n_squares_new / new_lattice.n_plaquettes:.3f}")

        lattice = new_lattice
        n_each_flip[flip_type] += 1
        energies.append(energy)
        all_lattices.append(new_lattice.__getstate__())
        total_plaqs.append(new_lattice.n_plaquettes)
        n_squares_list.append(n_squares_new)

        if no_options:
            print("No valid moves found, stopping.")
            break
        if energy == 0:
            print("Reached the Honeycomb!")
            break

    # now we save it all
    with open(f"{params.output_location}{params.output_file_name}.pkl", "wb") as f:
        pickle.dump(
            {
                "lattices": all_lattices,
                "energies": energies,
                "total_plaqs": total_plaqs,
                "n_squares": n_squares_list,
                "params": params,
            },
            f,
        )


if __name__ == "__main__":
    params = Parameters(
        n_sites=300,
        k_steps=500,
        beta_range=(1, 10),
    )
    main(params)
