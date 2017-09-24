from keras import backend as K
from keras import activations, regularizers, constraints
from keras.engine.topology import Layer, InputSpec
import tensorflow as tf

_GRAPH_EPISODIC_PHASES = {}


def episodic_phase():
    graph = tf.get_default_graph()
    if graph not in _GRAPH_EPISODIC_PHASES:
        phase = tf.placeholder(dtype='bool', name='episodic_phase')
        _GRAPH_EPISODIC_PHASES[graph] = phase
    return _GRAPH_EPISODIC_PHASES[graph]


def set_episodic_phase(value):
    global _GRAPH_EPISODIC_PHASES
    if value not in {0, 1}:
        raise ValueError('Expected episodic phase to be 0 or 1.')
    _GRAPH_EPISODIC_PHASES[tf.get_default_graph()] = value


class EpisodicNoiseDense(Layer):
    def __init__(self, output_dim, init='glorot_uniform',
                 activation=None, weights=None,
                 W_regularizer=None, b_regularizer=None,
                 activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, input_dim=None, sigma=0.01, **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim='2+')]
        self.sigma = sigma

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(EpisodicNoiseDense, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.input_dim = input_dim
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     ndim='2+')]

        self.W = self.add_weight((input_dim, self.output_dim),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((self.output_dim,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.noise = K.random_normal(
            shape=K.shape(self.W), mean=0., std=self.sigma)
        self.built = True

    def call(self, x, mask=None):
        if episodic_phase() is 1:
            self.noise = K.random_normal(
                shape=K.shape(self.W), mean=0., std=self.sigma)
            set_episodic_phase(0)
        if K.learning_phase() is 1:
            output = K.dot(x, self.W + self.noise)
        else:
            output = K.dot(x, self.W)
        if self.bias:
            output += self.b

        return self.activation(output)

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1] and input_shape[-1] == self.input_dim
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias,
                  'input_dim': self.input_dim,
                  'sigma': self.sigma
                  }
        base_config = super(EpisodicNoiseDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
