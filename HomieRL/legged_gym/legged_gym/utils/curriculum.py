class CurriculumScheduler:
    """
    Linear curriculum scheduler
    """

    def __init__(
        self,
        start_step,
        end_step,
        min_value=0.0,
        max_value=1.0,
        warmup=1000,
        ramp=8000,
    ):
        self.start_step = start_step
        self.end_step = end_step
        self.min_value = min_value
        self.max_value = max_value
        self.warmup = warmup
        self.ramp = ramp
    def value(self, global_step,):
        t = global_step - self.start_step
        if t<0:
            return 0
        elif  t < self.warmup:
            return self.min_value
        else:
            alpha = min(1.0,(t-self.warmup)/self.ramp)
            return self.min_value + alpha * (self.max_value - self.min_value)
