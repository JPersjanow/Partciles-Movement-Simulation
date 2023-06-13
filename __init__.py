import argparse
import sys
from Plotter import Plotter
from Simluation import Simulation
from ConfigParser import FileParser


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
