def penguin_enabled(self):
    """Returns whether penguin optimization is enabled."""
    return self.zero_optimization() and self.zero_config.penguin is not None

def penguin_shard_size(self):
    """Returns the shard size for penguin optimization."""
    return self.zero_config.penguin.shard_size if self.penguin_enabled() else -1

def penguin_hierarchical_params_gather(self):
    """Returns whether hierarchical parameter gathering is enabled for penguin."""
    return self.zero_config.penguin.hierarchial_params_gather if self.penguin_enabled() else False

def _configure_zero_optimizer(self, *args, **kwargs):
    zero_stage = self.zero_optimization_stage()

    if zero_stage == 3:
        if self.penguin_enabled():
            # Use Penguin optimizer
            if kwargs.get('optimizer') is None:
                optimizer = DummyOptim(list(self.module.parameters()))
            else:
                optimizer = kwargs['optimizer']
                
            optimizer = Penguin_Optimizer(
                module=self.module,
                init_optimizer=optimizer,
                timers=self.timers if self.wall_clock_breakdown() else NoopTimer(),
                ds_config=self.config,
                **self._get_zero_optimizer_config()
            )
            return optimizer
