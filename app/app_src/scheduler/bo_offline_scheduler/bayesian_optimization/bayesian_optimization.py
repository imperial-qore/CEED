import numpy as np
from bayes_opt import BayesianOptimization, Events, UtilityFunction


class CustomBayesianOptimization(BayesianOptimization):
    """
    Override the maximize method of BayesianOptimization to allow for custom acquisition functions.
    """
    def __init__(self,
                 f,
                 pbounds,
                 constraint=None,
                 random_state=None,
                 verbose=2,
                 bounds_transformer=None,
                 allow_duplicate_points=False):
        super(CustomBayesianOptimization, self).__init__(f, pbounds, constraint, random_state, verbose,
                                                         bounds_transformer, allow_duplicate_points)
        self.improvement = - np.inf

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acquisition_function=None,
                 acq=None,
                 kappa=None,
                 kappa_decay=None,
                 kappa_decay_delay=None,
                 xi=None,
                 **gp_params,
                 ):
        """
        ****** In this implementation, we use the gp_params as the parameters we need in the custom maximization ******

        Probes the target space to find the parameters that yield the maximum
        value for the given function.

        Parameters
        ----------
        init_points : int, optional(default=5)
            Number of iterations before the explorations starts the exploration
            for the maximum.

        n_iter: int, optional(default=25)
            Number of iterations where the method attempts to find the maximum
            value.

        acquisition_function: object, optional
            An instance of bayes_opt.util.UtilityFunction.
            If nothing is passed, a default using ucb is used
            kind: {'ucb', 'ei', 'poi'}
            * 'ucb' stands for the Upper Confidence Bounds method
            * 'ei' is the Expected Improvement method
            * 'poi' is the Probability Of Improvement criterion.

            acquisition_function = UtilityFunction(kind='ucb', kappa=2.576, xi=0, kappa_decay=1, kappa_decay_delay=0):

            kappa: float, optional(default=2.576)
                    Parameter to indicate how closed are the next parameters sampled.
                    Higher value = favors spaces that are least explored.
                    Lower value = favors spaces where the regression function is
                    the highest.

            kappa_decay: float, optional(default=1)
                `kappa` is multiplied by this factor every iteration.

            kappa_decay_delay: int, optional(default=0)
                Number of iterations that must have passed before applying the
                decay to `kappa`.

        All other parameters are unused, and are only available to ensure backwards compatability - these
        will be removed in a future release
        """
        self._prime_subscriptions()
        self.dispatch(Events.OPTIMIZATION_START)
        self._prime_queue(init_points)

        # These are from original script which use to avoid using old parameters and gaussian process parameters
        # old_params_used = any([param is not None for param in [acq, kappa, kappa_decay, kappa_decay_delay, xi]])
        #
        # if old_params_used or gp_params:
        #     raise Exception('\nPassing acquisition function parameters or gaussian process parameters to maximize'
        #                     '\nis no longer supported. Instead,please use the "set_gp_params" method to set'
        #                     '\n the gp params, and pass an instance of bayes_opt.util.UtilityFunction'
        #                     '\n using the acquisition_function argument\n')

        if acquisition_function is None:
            util = UtilityFunction(kind='ucb',
                                   kappa=2.576,
                                   xi=0.0,
                                   kappa_decay=1,
                                   kappa_decay_delay=0)
        else:
            util = acquisition_function

        best_target = -np.inf
        steady_count = 0
        iteration = 0
        early_stop_count = gp_params.get("early_stop_count", np.inf)
        start_score = None
        while not self._queue.empty or iteration < n_iter:
            try:
                x_probe = next(self._queue)
            except StopIteration:
                util.update_params()
                x_probe = self.suggest(util)
                iteration += 1
            self.probe(x_probe, lazy=False)

            # add custom code here
            new_target = self.res[-1]['target']
            if start_score is None:
                start_score = new_target if new_target > 0 else 0
            if new_target > best_target:
                best_target = new_target
                steady_count = 0
            else:
                steady_count += 1
            # if the target value does not improve for early_stop_count iterations, stop the optimization
            if steady_count >= early_stop_count:
                break

            if self._bounds_transformer and iteration > 0:
                # The bounds transformer should only modify the bounds after
                # the init_points points (only for the true iterations)
                self.set_bounds(
                    self._bounds_transformer.transform(self._space))
        self.improvement = best_target - start_score
        self.dispatch(Events.OPTIMIZATION_END)
