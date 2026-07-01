import json
import logging
import os
import pickle
import sys

import numpy as np
from config import Parameters
from koala.lattice import Lattice
from simulation import solve_lattice_data

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)


def choose_lattices(lattice_data, n_lattices):
    n_squares = np.array(lattice_data["n_squares"])
    max_min = (max(n_squares), min(n_squares))
    target_values = np.linspace(*max_min, n_lattices)
    diffs = np.abs(n_squares[:, None] - target_values)
    chosen_lats = np.argmin(diffs, axis=0)
    return chosen_lats


def main(task_id):

    # load the configurations
    with open("/scratch/perudornellas/dimer_models/pachner/full_run/run_config.json") as f:
        data = json.load(f)
        params = Parameters(**data)

    # load the lattices
    with open(params.lattice_location, "rb") as f:
        lattice_data = pickle.load(f)

    # make rng
    ss = np.random.SeedSequence(params.rng_seed)
    child = ss.spawn(params.n_tasks)[task_id]
    rng = np.random.default_rng(child)

    # choose the lattice and edges / plaquettes to go over
    lattice_index = choose_lattices(lattice_data, params.n_tasks)[task_id]
    lattice = Lattice(*lattice_data["lattices"][lattice_index])
    edge_list = np.array(
        [rng.choice(lattice.n_edges, 2, replace=False) for _ in range(params.n_pairs)]
    )
    plaq_list = np.array(
        [rng.choice(lattice.n_plaquettes, 2, replace=False) for _ in range(params.n_pairs)]
    )

    # solve it!
    results = solve_lattice_data(lattice, edge_list, plaq_list)

    results["lattice"] = lattice.compact_rep
    results["edge_list"] = edge_list
    results["plaq_list"] = plaq_list
    results["params"] = params

    # save it!!!
    with open(params.results_location(task_id), "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    job_id = int(os.environ["SLURM_JOB_ID"])
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    print(f"Job: {job_id}, task: {task_id}")
    main(task_id)
