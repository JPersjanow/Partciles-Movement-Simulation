import numpy as np
import matplotlib.pyplot as plt
import random


class Particle:
    def __init__(self):
        self.charge = self._give_initial_charge()  # Li(0) or Li+(1)
        self.velocity = 0  # speed of the particle that is traveling, vector always pointing outside center

    @staticmethod
    def _give_initial_charge() -> int:
        return random.randint(0, 1)

    def change_velocity(self, amount: float, option: str) -> None:
        match option:
            case "speed_up":
                self.velocity += amount
            case "slow_down":
                self.velocity -= amount
            case "stop":
                self.velocity = 0


class Dish:
    def __init__(self, number_radial, number_angles, max_density):
        self.number_radial = number_radial
        self.number_angles = number_angles
        self.max_density = max_density  # max density of particles in one cell

        # particles_array = np.empty(max_density, dtype=Particle)
        # data_type = np.dtype([("density", np.float64), ("particles", list)])

        self.density_array = np.zeros((self.number_radial, self.number_angles), dtype=int)
        self.particles_array = np.empty((self.number_radial, self.number_angles), dtype=list)

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

    def add_particles(self, amount, position="centre", radial_position=None, angle_position=None):
        match position:
            case "random":
                for i in range(amount):
                    print(f"Iter: {i}")
                    particle = Particle()
                    random_radial_position = random.randint(0, self.number_radial - 1)
                    random_angle_position = random.randint(0, self.number_angles - 1)
                    self._add_single_particle(random_radial_position, random_angle_position, particle)
            case "centre":
                for i in range(amount):
                    print(f"Adding particle, iter: {i}")
                    particle = Particle()
                    random_angle_position = random.randint(0, self.number_angles - 1)
                    self._add_single_particle(radial_position, random_angle_position, particle)
            case "specific":
                for i in range(amount):
                    print(f"Adding particle, iter: {i}")
                    particle = Particle()
                    self._add_single_particle(radial_position, angle_position, particle)

    def _add_single_particle(self, x_pos, y_pos, particle_object: Particle):
        print(f"Adding particle at position {x_pos} {y_pos}")
        self.density_array[x_pos][y_pos] += 1

        if not self.particles_array[x_pos][y_pos]:
            self.particles_array[x_pos][y_pos] = [particle_object]
        else:
            self.particles_array[x_pos][y_pos].append(particle_object)

    def move_particles(self, coord_x, coord_y):

        pass

    def plot_density(self, save=True):
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
        cb = ax.pcolormesh(self.angles_array, self.radial_array, self.density_array, edgecolors='k', linewidths=1)
        ax.set_yticks([])
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        plt.colorbar(cb, orientation='vertical')

        if save:
            plt.savefig(f'dish.pdf')
        else:
            plt.show()


if __name__ == "__main__":
    dish = Dish(10, 100, 10)
    dish.add_particles(100, position="random")
    dish.plot_density(True)
    print()
