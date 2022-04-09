from abc import ABC, abstractmethod

from torch.optim import Optimizer


class BaseScheduler(ABC):
    # TODO docs

    def __repr__(self) -> str:
        return self.__str__()

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def step(self) -> None:
        # TODO docs

        pass

    @abstractmethod
    def set_rate(self, rate: float) -> None:
        # TODO docs

        pass


class IdentityScheduler(BaseScheduler):
    # TODO docs

    def __str__(self) -> str:
        return '<IdentityScheduler()>'

    def step(self) -> None:
        pass

    def set_rate(self, rate: float) -> None:
        pass


class WarmupScheduler(BaseScheduler):
    # TODO docs

    def __init__(self,
                 optimizer: Optimizer,
                 d_model: int,
                 warmup: int,
                 step_size: int,
                 seek: int = 0,
                 factor: float = 1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup = warmup
        self.step_size = step_size
        self.factor = factor

        self.__step = self.__get_rate_step(self.optimizer.param_groups[0]['lr']) + seek
        self.__update_rate()

    def __str__(self) -> str:
        return f'WarmupScheduler(warmup={self.warmup}, step_size={self.step_size}, factor={self.factor})'

    def step(self) -> None:
        self.__step += 1

        if self.__step % self.step_size == 0:
            self.__update_rate()

    def set_rate(self, rate: float) -> None:
        self.__step = self.__get_rate_step(rate)
        self.__update_rate()

    def __update_rate(self) -> None:
        self.__rate = self.__get_step_rate(self.__step)

        for p in self.optimizer.param_groups:
            p['lr'] = self.__rate

    def __get_step_rate(self, step: int) -> float:
        if self.__step == 0:
            return 0

        return self.factor * (self.d_model ** (-0.5)) * min((step / self.step_size) ** (-0.5),
                                                            (step / self.step_size) * self.warmup ** (-1.5))

    def __get_rate_step(self, rate: float) -> int:
        if rate == 0:
            return 0

        return round(self.step_size / (self.d_model * rate ** 2))
