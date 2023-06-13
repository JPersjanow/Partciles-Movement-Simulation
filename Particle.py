import random


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
