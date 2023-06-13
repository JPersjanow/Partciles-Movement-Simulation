from Dish import Dish
from log import setup_custom_logger
from timeit import default_timer as timer


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
