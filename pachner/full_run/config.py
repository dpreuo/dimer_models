import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Parameters:
    lattice_directory: str
    results_directory: str
    job_name: str
    n_tasks: int
    n_pairs: int
    rng_seed: int

    @property
    def lattice_location(self) -> str:
        return f"{self.lattice_directory}{self.job_name}.pkl"

    # this makes sure the output directory exists
    def results_location(self, task_id: int) -> str:
        directory = os.path.join(self.results_directory, self.job_name)
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, f"{task_id:03d}.pkl")
