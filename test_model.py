import numpy as np
import matplotlib.pyplot as plt
import random
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
                self.velocity += amount
            case "slow_down":
                self.velocity -= amount
            case "stop":
                self.velocity = 0


class Dish:
    def __init__(
        self,
        number_radial,
        number_angles,
        min_escape_velocity=2.0,
        min_move_velocity=1.0,
    ):
        self.log = setup_custom_logger("Dish")
        self.number_radial = number_radial
        self.number_angles = number_angles
        self.min_escape_velocity = (
            min_escape_velocity  # max density of particles in one cell
        )
        self.min_move_velocity = min_move_velocity
        self.particles_charge_change_count = []

        self.density_array = np.zeros(
            (self.number_radial, self.number_angles), dtype=int
        )
        self.particles_array = np.empty(
            (self.number_radial, self.number_angles), dtype=list
        )

        # OPTION 1:
        # array = [
        #     [neutral_density, positive_density, negative_density]
        # ]

        # one cell:\
        #     [[x, y, z], [x1, y1, z1], [x2, y2, z2]]
        #     [Particle, Particle1, Particle3]

        # dtype = np.dtype(([("denisty", float), ("particles", list/np.array(dtype=Particle))]))
        # self.array = np.array(number_angles, number_radial, dtype=)

        self.angles_array = np.linspace(0, 2 * np.pi, number_angles + 1)
        self.radial_array = np.linspace(0, number_radial, number_radial + 1)

    def add_particles(
        self, amount, position="centre", radial_position=None, angle_position=None
    ) -> None:
        if not radial_position:
            radial_position = 0
        if not angle_position:
            angle_position = 0

        match position:
            case "random":
                for i in range(amount):
                    self.log.debug(f"Iter: {i}")
                    particle = Particle()
                    random_radial_position = random.randint(0, self.number_radial - 1)
                    random_angle_position = random.randint(0, self.number_angles - 1)
                    self._add_single_particle(
                        random_radial_position, random_angle_position, particle
                    )
            case "centre":
                for i in range(amount):
                    self.log.debug(f"Adding particle, iter: {i}")
                    particle = Particle()
                    random_angle_position = random.randint(0, self.number_angles - 1)
                    self._add_single_particle(
                        radial_position, random_angle_position, particle
                    )
            case "specific":
                for i in range(amount):
                    self.log.debug(f"Adding particle, iter: {i}")
                    particle = Particle()
                    self._add_single_particle(radial_position, angle_position, particle)

    def _add_single_particle(
        self, radial_pos, angle_pos, particle_object: Particle
    ) -> None:
        self.log.debug(f"Adding particle at position {radial_pos} {angle_pos}")
        self.density_array[radial_pos][angle_pos] += 1

        if not self.particles_array[radial_pos][angle_pos]:
            self.particles_array[radial_pos][angle_pos] = [particle_object]
        else:
            self.particles_array[radial_pos][angle_pos].append(particle_object)

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
            self.log.debug("Changing particle charge on random")
            particle_to_change_pos = (
                random.randint(0, len(self.particles_array[radial_pos][angle_pos])) - 1
            )
            change = self._change_single_particle_charge(
                radial_pos=radial_pos,
                angle_pos=angle_pos,
                particle_pos=particle_to_change_pos,
            )
        if change:
            self.log.debug("Particle charge changed")
            try:
                self.particles_charge_change_count[iteration] += 1
            except IndexError:
                self.particles_charge_change_count.append(1)
        else:
            self.log.debug("Particle not changed")

    def bounce_particles(self, radial_pos, angle_pos, iteration=0):
        if self.particles_array[radial_pos][angle_pos] is None:
            return False
        current_density = len(self.particles_array[radial_pos][angle_pos])
        self.log.debug(current_density)

        if current_density <= 2:
            return False

        should_bounce = bool(random.getrandbits(1))

        if should_bounce:
            self.log.debug("Bouncing particles")
            self.log.debug(
                f"Available particles: {self.particles_array[radial_pos][angle_pos]}"
            )
            particle_to_bounce_1 = random.choice(
                self.particles_array[radial_pos][angle_pos]
            )
            particle_to_bounce_2 = random.choice(
                self.particles_array[radial_pos][angle_pos]
            )
            self._particle_bounce(particle_to_bounce_1, particle_to_bounce_2)

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
        ].velocity += 0.1

        self.particles_array[new_radial_pos][new_angle_pos].append(
            self.particles_array[org_radial_pos][org_angle_pos].pop(particle_pos)
        )
        return True

    def move_particles(self, radial_pos, angle_pos) -> None:
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
            if particle_to_move.velocity >= self.min_escape_velocity:
                self._escape_single_particle(
                    radial_pos, angle_pos, particle_to_move_pos
                )
                return True
            else:
                return False

        self.log.debug(
            f"""
            Current density: {current_density}
            Neighbour density: {neighbour_density}
            Particle vel: {particle_to_move.velocity}
            """
        )

        # if density of neighbour is lower then move the particle or if velocity is high enough
        if (
            current_density > neighbour_density
            or particle_to_move.velocity >= self.min_move_velocity
        ):
            self.log.debug(
                f"""
            Moving particle
            from: {radial_pos} - {angle_pos}
            to: {edge_neigbour_coord["radial_pos"]} - {edge_neigbour_coord["angle_pos"]}
            """
            )
            self._move_single_particle(
                radial_pos,
                angle_pos,
                edge_neigbour_coord["radial_pos"],
                edge_neigbour_coord["angle_pos"],
                particle_to_move_pos,
            )
            return True

    def save_state_to_csv(self) -> None:
        pass

    def plot_density(self, save=True) -> None:
        fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
        cb = ax.pcolormesh(
            self.angles_array,
            self.radial_array,
            self.density_array,
            edgecolors="k",
            linewidths=1,
        )
        ax.set_yticks([])
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        plt.colorbar(cb, orientation="vertical")

        if save:
            plt.savefig(f"dish.png")
        else:
            plt.show()


class Simulation:
    def __init__(
        self,
        number_of_iterations: int,
        dish_arguments_dict: dict,
    ) -> None:
        self.dish = Dish(
            number_radial=dish_arguments_dict["number_radial"],
            number_angles=dish_arguments_dict["number_angles"],
            min_escape_velocity=dish_arguments_dict["min_escape_velocity"],
            min_move_velocity=dish_arguments_dict["min_move_velocity"],
        )
        self.dish.add_particles(
            amount=dish_arguments_dict["initial_amount"],
            position=dish_arguments_dict["position"],
        )

        self.iter_number = number_of_iterations
        self.iter_addition_amount = dish_arguments_dict["iter_addition_amount"]
        self.log = setup_custom_logger("Simulation")

    def simulate(self):
        start = timer()
        for current_iteration in range(self.iter_number):
            self.log.info(f"--------ITERATION NUMBER: {current_iteration}--------")
            self.iterate_over_dish(current_iteration)
        end = timer()
        self.log.info(f"Simulation time:")
        self.log.info(end - start)
        self.dish.plot_density(save=True)

    def iterate_over_dish(self, iteration_number):
        for radial_pos, radial in enumerate(self.dish.density_array):
            for angle_pos, angle in enumerate(radial):
                self.dish.move_particles(radial_pos=radial_pos, angle_pos=angle_pos)
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
        self.dish.add_particles(amount=self.iter_addition_amount, position="centre")


if __name__ == "__main__":
    dish = {
        "number_radial": 10,
        "number_angles": 100,
        "min_escape_velocity": 2.0,
        "min_move_velocity": 1.0,
        "initial_amount": 100,
        "position": "centre",
        "iter_addition_amount": 100,
    }
    sim = Simulation(number_of_iterations=100, dish_arguments_dict=dish)
    sim.simulate()
