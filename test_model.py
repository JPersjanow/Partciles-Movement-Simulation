import numpy as np
import matplotlib.pyplot as plt
import random
import os
import glob
import argparse
import sys
import json
from log import setup_custom_logger
from timeit import default_timer as timer


class NoParticlesException(Exception):
    """The referenced space does not contain any Particles"""


class Particle:
    def __init__(self):
        self.charge = self._give_initial_charge()  # Li(0) or Li+(1)

        # speed of the particle that is traveling, vector always pointing outside center
        if self.charge != 0:
            self.velocity = 0.1
        else:
            self.velocity = 0

    @staticmethod
    def _give_initial_charge() -> int:
        return random.randint(0, 1)

    def change_charge(self) -> bool:
        if self.charge == 1:
            self.charge = 0
        else:
            self.charge = 1

    def change_velocity(self, option: str, amount=0.1) -> None:
        match option:
            case "speed_up":
                new_vel = self.velocity + amount
            case "slow_down":
                new_vel = self.velocity - amount
            case "stop":
                new_vel = 0
            case "exact":
                new_vel = amount
        self.velocity = round(new_vel, 1)


class Dish:
    def __init__(
        self,
        number_radial,
        number_angles,
        number_of_iterations,
        min_escape_velocity=2.0,
        min_move_velocity=1.0,
        min_move_density=10,
    ):
        self.log = setup_custom_logger("Dish")
        self.number_radial = number_radial
        self.number_angles = number_angles
        self.min_escape_velocity = (
            min_escape_velocity  # max density of particles in one cell
        )
        self.min_move_velocity = min_move_velocity
        self.min_move_density = min_move_density
        self.number_of_iterations = number_of_iterations
        self.particles_charge_change_count = np.zeros(number_of_iterations, dtype=int)
        self.particles_off_system_count = np.zeros(number_of_iterations, dtype=int)
        self.particles_amount_count = np.zeros(number_of_iterations, dtype=int)
        self.particles_bounce_count = np.zeros(number_of_iterations, dtype=int)

        self.density_array = np.zeros(
            (self.number_radial, self.number_angles), dtype=int
        )
        self.particles_array = np.empty(
            (self.number_radial, self.number_angles), dtype=list
        )

    def add_particles(
        self,
        amount,
        position="centre",
        radial_position=None,
        angle_position=None,
        iteration=0,
    ) -> None:
        if not radial_position:
            radial_position = 0
        if not angle_position:
            angle_position = 0

        match position:
            case "random":
                for i in range(amount):
                    self.log.debug(f"Adding particle, iter: {i}")
                    particle = Particle()
                    random_radial_position = random.randint(0, self.number_radial - 1)
                    random_angle_position = random.randint(0, self.number_angles - 1)
                    self._add_single_particle(
                        random_radial_position,
                        random_angle_position,
                        particle,
                        iteration,
                    )
            case "centre":
                for i in range(amount):
                    self.log.debug(f"Adding particle, iter: {i}")
                    particle = Particle()
                    random_angle_position = random.randint(0, self.number_angles - 1)
                    self._add_single_particle(
                        radial_position, random_angle_position, particle, iteration
                    )
            case "specific":
                for i in range(amount):
                    self.log.debug(f"Adding particle, iter: {i}")
                    particle = Particle()
                    self._add_single_particle(
                        radial_position, angle_position, particle, iteration
                    )
        self._add_to_counter(
            self.particles_amount_count,
            iteration,
            amount=amount,
            should_count_previous=True,
        )

    def _add_single_particle(
        self, radial_pos, angle_pos, particle_object: Particle, iteration=0
    ) -> None:
        self.log.debug(f"Adding particle at position {radial_pos} {angle_pos}")
        self.density_array[radial_pos][angle_pos] += 1

        if not self.particles_array[radial_pos][angle_pos]:
            self.particles_array[radial_pos][angle_pos] = [particle_object]
        else:
            self.particles_array[radial_pos][angle_pos].append(particle_object)

    def _add_to_counter(
        self, counter: np.array, iteration, amount=1, should_count_previous=False
    ):
        if should_count_previous:
            try:
                counter[iteration] = counter[iteration - 1] + amount
            except IndexError:
                counter[iteration] = amount
        else:
            counter[iteration] += 1

    def _return_left_neighbour(self, array, radial_pos, angle_pos) -> tuple:
        coords = {"radial_pos": 0, "angle_pos": 0}

        neighbour_cell = array[radial_pos][angle_pos - 1]
        if angle_pos - 1 < 0:
            coords["angle_pos"] = self.number_angles - 1
        else:
            coords["angle_pos"] = angle_pos - 1
        coords["radial_pos"] = radial_pos

        return neighbour_cell, coords

    def _return_right_neighbour(self, array, radial_pos, angle_pos) -> tuple:
        coords = {"radial_pos": 0, "angle_pos": 0}

        try:
            neighbour_cell = array[radial_pos][angle_pos + 1]
            coords["angle_pos"] = angle_pos + 1
        except IndexError:
            neighbour_cell = array[radial_pos][0]  # we come around in circle
            coords["angle_pos"] = 0
        coords["radial_pos"] = radial_pos

        return neighbour_cell, coords

    def _return_centre_neighbour(self, array, radial_pos, angle_pos) -> tuple:
        coords = {"radial_pos": 0, "angle_pos": 0}

        if radial_pos - 1 < 0:
            neighbour_cell = None  # we are at the top
            coords["radial_pos"] = None
        else:
            neighbour_cell = array[radial_pos - 1][angle_pos]
            coords["radial_pos"] = radial_pos - 1
        coords["angle_pos"] = angle_pos

        return neighbour_cell, coords

    def _return_edge_neighbour(self, array, radial_pos, angle_pos) -> tuple:
        coords = {"radial_pos": 0, "angle_pos": 0}

        try:
            neighbour_cell = array[radial_pos + 1][angle_pos]
            coords["radial_pos"] = radial_pos + 1
        except IndexError:
            neighbour_cell = None  # we are at the edge of the dish
            coords["radial_pos"] = None
        coords["angle_pos"] = angle_pos

        return neighbour_cell, coords

    def _retun_neighbours(self, array, radial_pos, angle_pos) -> tuple:
        # LEFT NEIGHBOUR

        left_neighbour_cell, left_coords = self._return_left_neighbour(
            array, radial_pos, angle_pos
        )

        # RIGHT NEIGHBOUR

        right_neighbour_cell, right_coords = self._return_right_neighbour(
            array, radial_pos, angle_pos
        )

        # UP NEIGHBOUR

        centre_neighbour_cell, centre_coords = self._return_left_neighbour(
            array, radial_pos, angle_pos
        )

        # DOWN NEIGHBOUR

        edge_neighbour_cell, edge_coords = self._return_edge_neighbour(
            array, radial_pos, angle_pos
        )

        coords = {
            "left": left_coords,
            "right": right_coords,
            "centre": centre_coords,
            "edge": edge_coords,
        }

        return [
            left_neighbour_cell,
            right_neighbour_cell,
            centre_neighbour_cell,
            edge_neighbour_cell,
        ], coords

    def _return_random_neighbour(self, array, radial_pos, angle_pos):
        neighbours, neighbours_coords = self._retun_neighbours(
            array, radial_pos, angle_pos
        )

        random_neighbour_choice = random.randint(0, 3)
        random_neighbour = neighbours[random_neighbour_choice]
        random_neighbour_coord: dict = list(neighbours_coords.values())[
            random_neighbour_choice
        ]

        return random_neighbour, random_neighbour_coord

    def _does_contain_particles(self, radial_pos, angle_pos):
        if self.particles_array[radial_pos][angle_pos] is None:
            return False
        else:
            return True

    def return_random_neighbour_praticle_array(self, radial_pos, angle_pos):
        return self._return_random_neighbour(
            self.particles_array, radial_pos=radial_pos, angle_pos=angle_pos
        )

    def return_neighbours_particle_array(self, radial_pos, angle_pos):
        return self._retun_neighbours(
            self.particles_array, radial_pos=radial_pos, angle_pos=angle_pos
        )

    def return_random_neighbour_density_array(self, radial_pos, angle_pos):
        return self._return_random_neighbour(
            self.density_array, radial_pos=radial_pos, angle_pos=angle_pos
        )

    def return_neighbours_density_array(self, radial_pos, angle_pos):
        return self._retun_neighbours(
            self.density_array, radial_pos=radial_pos, angle_pos=angle_pos
        )

    def particle_charge_swap(self, particle_1: Particle, particle_2: Particle) -> bool:
        if particle_1.charge == particle_2.charge:
            return False
        else:
            particle_1.change_charge()
            particle_2.change_charge()
            return True

    def _particle_bounce(self, particle_1: Particle, particle_2: Particle) -> bool:
        """
        Simulates perfectly buoyant impacts between particles
        """

        if particle_1.velocity == 0 and particle_2.velocity == 0:
            return False
        if particle_1.velocity == 0:
            particle_1.velocity = particle_2.velocity
            particle_2.velocity = 0
            return True
        if particle_2.velocity == 0:
            particle_2.velocity = particle_1.velocity
            particle_1.velocity = 0
            return True
        if particle_1 != 0 and particle_2 != 0:
            initial_particle_1_velocity = particle_1.velocity
            initial_particle_2_velocity = particle_2.velocity
            particle_1.velocity = initial_particle_2_velocity
            particle_2.velocity = initial_particle_1_velocity
            return True

    def _change_single_particle_charge(
        self, radial_pos, angle_pos, particle_pos
    ) -> bool:
        """
        Chooses random neighbour cell that has particles inside, chooses random particle, changes charge

        If charge was not change because:
            1) There is no particles in given radial/angle pos
            2) Particle charge is already postivie (no electron jump)
            3) Neighbour to change has no particles
            4) Neighbour chosen particle charge is neutral

        method returns False
        If charge was changed returns True
        """
        if self._does_contain_particles(radial_pos, angle_pos):
            return False

        particle_to_change: Particle = self.particles_array[radial_pos][angle_pos][
            particle_pos
        ]

        if particle_to_change.charge == 1:
            return False

        (
            neighbour_to_change,
            neighbour_to_change_coord,
        ) = self.return_random_neighbour_praticle_array(radial_pos, angle_pos)

        if neighbour_to_change is None:
            return False

        neighbour_particle_to_change: Particle = neighbour_to_change[
            random.randint(0, len(neighbour_to_change))
        ]

        if neighbour_particle_to_change.charge == 0:
            return False

        return self.particle_charge_swap(
            particle_1=particle_to_change, particle_2=neighbour_particle_to_change
        )

    def change_particle_charge(self, radial_pos, angle_pos, iteration=0):
        if self.particles_array[radial_pos][angle_pos] is None:
            return False

        should_change = bool(random.getrandbits(1))
        change = False

        if should_change:
            particle_to_change_pos = (
                random.randint(0, len(self.particles_array[radial_pos][angle_pos])) - 1
            )
            change = self._change_single_particle_charge(
                radial_pos=radial_pos,
                angle_pos=angle_pos,
                particle_pos=particle_to_change_pos,
            )
        if change:
            self._add_to_counter(self.particles_charge_change_count, iteration)

    def bounce_particles(self, radial_pos, angle_pos, iteration=0):
        if self.particles_array[radial_pos][angle_pos] is None:
            return False
        current_density = len(self.particles_array[radial_pos][angle_pos])

        if current_density <= 2:
            return False

        should_bounce = bool(random.getrandbits(1))

        if should_bounce:
            particle_to_bounce_1 = random.choice(
                self.particles_array[radial_pos][angle_pos]
            )
            particle_to_bounce_2 = random.choice(
                self.particles_array[radial_pos][angle_pos]
            )
            bounced = self._particle_bounce(particle_to_bounce_1, particle_to_bounce_2)

            if bounced:
                self._add_to_counter(self.particles_bounce_count, iteration)

    def _escape_single_particle(self, radial_pos, angle_pos, particle_pos) -> None:
        self.particles_array[radial_pos][angle_pos].pop(particle_pos)
        self.density_array[radial_pos][angle_pos] -= 1

    def _move_single_particle(
        self, org_radial_pos, org_angle_pos, new_radial_pos, new_angle_pos, particle_pos
    ) -> None:
        self.density_array[org_radial_pos][org_angle_pos] -= 1
        self.density_array[new_radial_pos][new_angle_pos] += 1

        if self.particles_array[new_radial_pos][new_angle_pos] is None:
            self.particles_array[new_radial_pos][new_angle_pos] = []

        self.particles_array[org_radial_pos][org_angle_pos][
            particle_pos
        ].change_velocity(option="speed_up")

        self.particles_array[new_radial_pos][new_angle_pos].append(
            self.particles_array[org_radial_pos][org_angle_pos].pop(particle_pos)
        )
        return True

    def move_particles(self, radial_pos, angle_pos, iteration=0) -> None:
        """
        Particles movement vector is always directed towards edge of the dish
        """

        if self.particles_array[radial_pos][angle_pos] is None:
            return False
        current_density = len(self.particles_array[radial_pos][angle_pos])
        if current_density == 0:
            return False

        edge_particle_neighbour, edge_neigbour_coord = self._return_edge_neighbour(
            self.particles_array, radial_pos, angle_pos
        )
        try:
            neighbour_density = len(edge_particle_neighbour)
        except TypeError:
            neighbour_density = 0

        particle_to_move_pos = (
            random.randint(0, len(self.particles_array[radial_pos][angle_pos])) - 1
        )

        particle_to_move: Particle = self.particles_array[radial_pos][angle_pos][
            particle_to_move_pos
        ]

        # if particle is at the edge, check velocity, if it is enough, throw the particle away from the dish
        if (radial_pos + 1) >= self.number_radial:
            self.log.info(
                f"Particle at edge position: {radial_pos}, velocity={particle_to_move.velocity}"
            )
            if particle_to_move.velocity >= self.min_escape_velocity:
                self._escape_single_particle(
                    radial_pos, angle_pos, particle_to_move_pos
                )
                self._add_to_counter(self.particles_off_system_count, iteration)
                return True
            else:
                return False

        # Move the particle when:
        # 1) density of neighbour is lower current density
        # 2) if velocity is high enough
        if (
            self.min_move_density <= current_density > neighbour_density
            or particle_to_move.velocity >= self.min_move_velocity
        ):
            self.log.info(f"Moving particle from: [{radial_pos}][{angle_pos}]")
            self._move_single_particle(
                radial_pos,
                angle_pos,
                edge_neigbour_coord["radial_pos"],
                edge_neigbour_coord["angle_pos"],
                particle_to_move_pos,
            )
            return True

    def save_state(self, iteration=0, save_particles=False, save_density=True) -> None:
        self.log.info(f"Saving states for iteration {iteration}")

        if save_particles:
            statefilename_pa = f"states/{iteration}_pa.npy"
            os.makedirs(os.path.dirname(statefilename_pa), exist_ok=True)
            with open(statefilename_pa, "wb") as statefile_pa:
                np.save(statefile_pa, self.particles_array)

        if save_density:
            statefilename_da = f"states/{iteration}_da.npy"
            os.makedirs(os.path.dirname(statefilename_da), exist_ok=True)
            with open(statefilename_da, "wb") as statefile_da:
                np.save(statefile_da, self.density_array)

    def save_counters(self) -> None:
        parciles_amount = np.subtract(
            self.particles_amount_count, self.particles_off_system_count
        )
        counter_filenames = {
            "pcount.npy": parciles_amount,
            "pOFFcount.npy": self.particles_off_system_count,
            "pBcount.npy": self.particles_bounce_count,
            "pCcount.npy": self.particles_charge_change_count,
        }

        for counter_filename, counter in counter_filenames.items():
            with open(counter_filename, "wb") as counterfile:
                np.save(counterfile, counter)


class Plotter:
    def __init__(
        self,
        dish_arguments_dict: dict,
        states_directory: str = "states",
        fig_directory: str = "figs",
    ) -> None:
        self.states_directory = states_directory
        self.fig_directory = fig_directory
        self.all_da_files = glob.glob(os.path.join(states_directory, "*_da.npy"))
        self.all_da_files.sort()
        self.all_pa_files = glob.glob(os.path.join(states_directory, "*_pa.npy"))
        self.all_pa_files.sort()

        self.number_radial = dish_arguments_dict["number_radial"]
        self.number_angles = dish_arguments_dict["number_angles"]

        self.angles_array = np.linspace(0, 2 * np.pi, self.number_angles + 1)
        self.radial_array = np.linspace(0, self.number_radial, self.number_radial + 1)

    def plot_density(self, density_array: np.array, iteration=0, save=True) -> None:
        fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
        cb = ax.pcolormesh(
            self.angles_array,
            self.radial_array,
            density_array,
            edgecolors="k",
            linewidths=1,
        )
        ax.set_yticks([])
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        plt.colorbar(cb, orientation="vertical")

        if save:
            fig_filename = f"{self.fig_directory}/dish-{iteration}.png"
            os.makedirs(os.path.dirname(fig_filename), exist_ok=True)
            plt.savefig(fig_filename)
            plt.close()
        else:
            plt.show()

    def plot_density_change(self):
        for index, da_file in enumerate(self.all_da_files):
            with open(da_file, "rb") as daf:
                density = np.load(daf)

            self.plot_density(density_array=density, iteration=index)
            del density


class Simulation:
    def __init__(
        self,
        number_of_iterations: int,
        dish_arguments_dict: dict,
        logger_cmd_output=False,
    ) -> None:
        self.dish = Dish(
            number_radial=dish_arguments_dict["number_radial"],
            number_angles=dish_arguments_dict["number_angles"],
            min_escape_velocity=dish_arguments_dict["min_escape_velocity"],
            min_move_velocity=dish_arguments_dict["min_move_velocity"],
            min_move_density=dish_arguments_dict["min_move_density"],
            number_of_iterations=number_of_iterations,
        )
        self.dish.add_particles(
            amount=dish_arguments_dict["initial_amount"],
            position=dish_arguments_dict["position"],
        )

        self.iter_number = number_of_iterations
        self.iter_addition_amount = dish_arguments_dict["iter_addition_amount"]
        self.save_state = dish_arguments_dict["save_state"]
        self.save_particles = dish_arguments_dict["save_particles"]
        self.save_iteration = dish_arguments_dict["save_iteration"]
        self.log = setup_custom_logger("Simulation", cmd_output=logger_cmd_output)

    def simulate(self):
        start = timer()
        for current_iteration in range(self.iter_number):
            self.log.info(f"ITERATION NUMBER: {current_iteration}")
            self.iterate_over_dish(current_iteration)

            if self.save_state and current_iteration % self.save_iteration == 0:
                self.dish.save_state(
                    iteration=current_iteration, save_particles=self.save_particles
                )

        end = timer()
        self.log.info(f"Simulation time:")
        self.log.info(end - start)
        self.dish.save_counters()
        if self.save_state:
            self.dish.save_state(
                iteration=current_iteration, save_density=True, save_particles=True
            )

    def iterate_over_dish(self, iteration_number):
        for radial_pos, radial in enumerate(self.dish.density_array):
            for angle_pos, angle in enumerate(radial):
                self.dish.move_particles(
                    radial_pos=radial_pos,
                    angle_pos=angle_pos,
                    iteration=iteration_number,
                )
                self.dish.change_particle_charge(
                    radial_pos=radial_pos,
                    angle_pos=angle_pos,
                    iteration=iteration_number,
                )
                self.dish.bounce_particles(
                    radial_pos=radial_pos,
                    angle_pos=angle_pos,
                    iteration=iteration_number,
                )
        self.dish.add_particles(
            amount=self.iter_addition_amount,
            position="centre",
            iteration=iteration_number,
        )


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate_from_start", "-startsim", action="store_true")
    parser.add_argument(
        "--dish_state_dir", "-dsd", required="--simulate_from_start" not in sys.argv
    )
    parser.add_argument(
        "--dish_state_number", "-dsn", required="--dish_state_dir" in sys.argv
    )
    parser.add_argument("--dish_parameters_file", "-dpf", required=True)
    parser.add_argument("--number_of_iterations", "-niter", required=True)
    args = parser.parse_args()

    fp = FileParser(
        simulate_from_start=args.simulate_from_start,
        dish_parameters_file=args.dish_parameters_file,
        dish_state_dir=args.dish_state_dir,
        dish_state_number=args.dish_state_number,
    )

    dish_arguments, density_array, particles_array = fp.parse_and_return()
    sim = None
    try:
        sim = Simulation(
            number_of_iterations=int(args.number_of_iterations),
            dish_arguments_dict=dish_arguments,
        )
        if not args.simulate_from_start:
            sim.dish.particles_array = particles_array
            sim.dish.density_array = density_array

        # Delete arrays because they might be big (especially particles array as it stores objects)
        del density_array
        del particles_array

        sim.simulate()
    except Exception as e:
        if sim:
            sim.dish.save_state(
                iteration="ERROR", save_density=True, save_particles=True
            )
        raise e

    plot = Plotter(dish_arguments_dict=dish_arguments)
    plot.plot_density_change()
