import numpy as np
import json
import os


class FileParser:
    def __init__(
        self,
        simulate_from_start: bool,
        dish_parameters_file: str = None,
        dish_state_dir: str = None,
        dish_state_number: str = None,
    ):
        self.simulate_from_start = simulate_from_start
        self.dish_parameters_file = dish_parameters_file
        if dish_state_dir and dish_state_number:
            self.dish_state_dir = dish_state_dir
            self.dish_partcile_state_file = os.path.join(
                dish_state_dir, f"{dish_state_number}_pa.npy"
            )
            self.dish_density_state_file = os.path.join(
                dish_state_dir, f"{dish_state_number}_da.npy"
            )

    def parse_and_return(self):
        density_array = None
        particles_array = None
        dish_arguments = None
        with open(self.dish_parameters_file, "r") as paramfile:
            dish_arguments = json.loads(paramfile.read().replace("'", '"'))

        if not self.simulate_from_start:
            with open(self.dish_density_state_file, "rb") as densitystatefile:
                density_array = np.load(densitystatefile)
            with open(self.dish_partcile_state_file, "rb") as particlestatefile:
                particles_array = np.load(particlestatefile, allow_pickle=True)
                return density_array, particles_array

        return dish_arguments, density_array, particles_array
