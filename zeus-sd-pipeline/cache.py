class CacheBus:
    """A Bus class for overall control."""

    def __init__(self):
        # == Tensor Caching ==
        self.prev_epsilon_guided = [None, None]
        self.prev_epsilon = None
        self.prev_interp = None
        self.prev_f = [None, None]
        self.lagrange_x0 = []
        self.prev_x = [None, None]

        # == Estimator ==
        self.pred_m_m_1 = None
        self.taylor_m_m_1 = None
        self.temporal_score = None

        # == Control Signals ==
        self.skip_this_step = False

        # == Control Variables ==
        self.step = 0
        self.cons_skip = 0
        self.cons_prune = 0
        self.lagrange_step = []
        self.last_skip_step = 0  # align with step in cache bus
        self.ind_step = None
        self.c_step = 1  # doesn't really matter

        # == Optimizations ==
        self.m_a = None
        self.u_a = None

        # == Logs ==
        # self.model_outputs = {}
        # self.model_outputs_change = {}
        self.pred_error_list = []
        self.taylor_error_list = []
        self.abs_momentum_list = []
        self.rel_momentum_list = []
        self.skipping_path = []