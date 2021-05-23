class Scheduler(object):
    def __init__(self, optimizer, d_model: int, warmup: int, step_size: int, factor: float = 1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup = warmup
        self.step_size = step_size
        self.factor = factor

        self.__step = 0
        self.__rate = 0

    @property
    def lr(self) -> float:
        return self.__rate

    def seek(self, step: int) -> None:
        self.__step = step
        self.__update_rate()

    def step(self) -> None:
        self.__step += 1

        if self.__step % self.step_size == 0:
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
