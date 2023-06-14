import matplotlib.pyplot as plt
import numpy as np
import glob
import os


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
            fig_filename = os.path.join(self.fig_directory, f"dish-{iteration}.png")
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
