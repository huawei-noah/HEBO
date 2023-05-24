from febo.algorithms.safety import SafetyMixin
from febo.utils.config import ClassConfigField, ConfigField, assign_config, config_manager
import numpy as np

class ModelMixinConfig:
    model = ClassConfigField('febo.models.GP')
    model_config = ConfigField({})
    constraints_model = ClassConfigField(None, allow_none=True)
    constraints_model_config = ConfigField({})
    noise_model = ClassConfigField(None, allow_none=True)
    noise_model_config = ConfigField({})
    _section = 'algorithm'


@assign_config(ModelMixinConfig)
class ModelMixin:
    """ Algorithm Class which provides a model and an optimizer instance, as configured in config.py
        self.f is the model of the objective function
        self.s is a list of models of the safety constraints (by calling initialize with num_safety_constraints=0,
        no safety constraints models are created)
    """
    def initialize(self, **kwargs):
        """

        Args:
            model: optional. If a model is passed, this is used instead of the model according to the config

        Returns:

        """
        super(ModelMixin, self).initialize(**kwargs)
        self._model_domain = kwargs.get('model_domain', self.domain)

        config_manager.load_data(self.config.model_config)

        # if model was passed as kwarg, use it, else create a new model from config
        self.model = kwargs.get('model', self.config.model(domain=self._model_domain))

        self._has_constraints_model = self.config.constraints_model is not None
        if self._has_constraints_model:
            config_manager.load_data(self.config.constraints_model_config)
            self.s = kwargs.get('constraint_model', [self.config.constraints_model(domain=self._model_domain) for _ in range(self.num_constraints)])

        self._has_noise_model = self.config.noise_model is not None
        if self._has_noise_model:
            config_manager.load_data(self.config.noise_model_config)
            self.noise = kwargs.get('noise_model', self.config.noise_model(domain=self._model_domain))

        # add initial data
        if not self.initial_data is None:
            for evaluation in self.initial_data:
                self._add_data_to_models(evaluation)

        self._initialize_best_prediction_algorithm(kwargs.copy())

    def _initialize_best_prediction_algorithm(self, greedy_initialize_kwargs):
        # avoid mutual imports since Greedy itself uses a ModelMixin
        from febo.algorithms.greedy import SafeGreedy, Greedy
        # initialize an algorithm to calculate best predicted point
        self._best_prediction_algorithm = None
        if not isinstance(self, (SafeGreedy, Greedy)):
            if isinstance(self, SafetyMixin):
                self._best_prediction_algorithm = SafeGreedy()
            else:
                self._best_prediction_algorithm = Greedy()

            # do not pass any initial data to greedy algorithm, as we are using the current model
            if 'initial_data' in greedy_initialize_kwargs:
                del greedy_initialize_kwargs['initial_data']
            greedy_initialize_kwargs['model'] = self.model
            if self._has_noise_model:
                greedy_initialize_kwargs['noise_model'] = self.noise
            if self._has_constraints_model:
                greedy_initialize_kwargs['constraint_model'] = self.s

            self._best_prediction_algorithm.initialize(**greedy_initialize_kwargs)


    def add_data(self, data):
        """ by default just passes the observed data to the model """
        super(ModelMixin, self).add_data(data)
        self._add_data_to_models(data)

    def _add_data_to_models(self, data):
        x = self._get_x_from_data(data)
        if self.model.requires_std:
            self.model.add_data(x, data["y"], self._get_std(data))
        else:
            self.model.add_data(x, data["y"])

        if self._has_constraints_model and self.num_constraints:
            if self.s[0].requires_std:
                for m, s, s_std in zip(self.s, data['s'], data['s_std']):
                    m.add_data(x, s, s_std)
            else:
                for m, s in zip(self.s, data['s']):
                    m.add_data(x, s)

        if self._has_noise_model:
            if self.noise.requires_std:
                self.noise.add_data(x, data['y_std'], data['y_std_std'])
            else:
                self.noise.add_data(x, data['y_std'])

    def _get_x_from_data(self, data):
        return data['x']

    def _get_std(self, data):
        """
        get std of observations from data, potentially computed from some other model
        Args:
            data:

        Returns:

        """
        return data["y_std"]

    def next(self, context=None):
        x, additional_data = super().next(context=context)
        # m,v = self.model.mean_var(x)
        # additional_data['y_model'] = m
        # additional_data['y_std_model'] = np.sqrt(v)
        return x, additional_data

    def optimize_model(self):
        self.model.minimize()

        if self._has_constraints_model:
            for s in self.s:
                s.minimize()

        if self._has_noise_model:
            self.noise.minimize()

    def best_predicted(self):
        if self._best_prediction_algorithm is None:
            raise NotImplementedError

        return self._best_prediction_algorithm.next()[0]

    def _get_dtype_fields(self):
        fields = super()._get_dtype_fields()
        fields.append(('y_model', 'f8'))
        fields.append(('y_std_model', 'f8'))
        return fields

    def get_joined_constrained_cb(self, X):
        joined_ucb = np.empty(shape=(X.shape[0], self.num_constraints))
        joined_lcb = np.empty(shape=(X.shape[0], self.num_constraints))
        for i,s in enumerate(self.s):
            mean, var = s.mean_var(X)
            mean = mean.flatten()
            std = np.sqrt(var.flatten())
            joined_lcb[:,i], joined_ucb[:,i] = mean - s.beta * std, mean + s.beta * std


        # Safe is <=0
        return np.max(joined_lcb, axis=1).reshape(-1,1), np.max(joined_ucb, axis=1).reshape(-1,1)
