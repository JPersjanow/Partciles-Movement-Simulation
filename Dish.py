import numpy as np
import random
import os
from Particle import Particle
from log import setup_custom_logger


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
        # TODO: Change to _does_contain_particles
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
        # TODO: Change to _does_contain_particles
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
        # TODO: Change to _does_contain_particles
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
            statefilename_pa = os.path.join("states", f"{iteration}_pa.npy")
            os.makedirs(os.path.dirname(statefilename_pa), exist_ok=True)
            with open(statefilename_pa, "wb") as statefile_pa:
                np.save(statefile_pa, self.particles_array)

        if save_density:
            statefilename_da = os.path.join("states", f"{iteration}_da.npy")
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
            counter_filename = os.path.join("counts", counter_filename)
            os.makedirs(os.path.dirname(counter_filename), exist_ok=True)
            with open(counter_filename, "wb") as counterfile:
                np.save(counterfile, counter)
