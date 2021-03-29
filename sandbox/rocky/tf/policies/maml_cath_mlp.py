from collections import OrderedDict

from sandbox.rocky.tf.core.layers_powered import LayersPowered
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.core.network import MLP
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.core.utils import make_dense_layer
from sandbox.rocky.tf.distributions.categorical import Categorical
from sandbox.rocky.tf.policies.base import StochasticPolicy
from rllab.misc import ext
from sandbox.rocky.tf.misc import tensor_utils
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.spaces.discrete import Discrete
import tensorflow as tf
import numpy as np
from collections import OrderedDict

from rllab.misc import ext
from sandbox.rocky.tf.spaces.box import Box

from rllab.core.serializable import Serializable
from sandbox.rocky.tf.policies.base import StochasticPolicy
from sandbox.rocky.tf.distributions.categorical import Categorical#import DiagonalGaussian # This is just a util class. No params.
from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.misc.tensor_utils import flatten_tensors, unflatten_tensors
from sandbox.rocky.tf.misc import tensor_utils

import itertools
import time

from tensorflow.contrib.layers.python import layers as tf_layers
from sandbox.rocky.tf.core.utils import make_input, _create_param, add_param, make_dense_layer, forward_dense_layer, make_param_layer, forward_param_layer
load_params = True

class CategoricalMLPPolicy(StochasticPolicy, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.tanh,
            prob_network=None,
    ):
        """
        :param env_spec: A spec for the mdp.
        :param hidden_sizes: list of sizes for the fully connected hidden layers
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param prob_network: manually specified network for this policy, other network params
        are ignored
        :return:
        """
        Serializable.quick_init(self, locals())

        assert isinstance(env_spec.action_space, Discrete)
        self.obs_dim = env_spec.observation_space.flat_dim
        self.action_dim = env_spec.action_space.n
        self.n_hidden = len(hidden_sizes)
        self.hidden_nonlinearity = hidden_nonlinearity
        self.input_shape = (None ,self.obs_dim  ,)
        self.init_flr_full = 0.5

        self.step_size = 0.5
        self.stop_grad = False
        with tf.variable_scope(name):
            if prob_network is None:

                self.all_params = self.create_MLP(
                    output_dim=env_spec.action_space.n,
                    hidden_sizes=hidden_sizes,
                    name="prob_network", )
                self.hidden_nonlinearity = hidden_nonlinearity
                self.output_nonlinearity = tf.nn.softmax
                self.input_tensor ,_ = self.forward_MLP('prob_network' ,self.all_params ,
                                                        reuse=None  # Need to run this for batch norm
                                                        )
                forward_mean = lambda x ,params ,is_train:self.forward_MLP('prob_network' ,params ,
                                                                           input_tensor=x ,
                                                                           is_training=is_train)[1]
                self.all_param_vals = None

                # unify forward mean and forward std into a single function
                self._forward = forward_mean
            dist_info_sym = self.dist_info_sym(self.input_tensor ,dict() ,is_training=False)
            output = dist_info_sym["prob"]
            self._l_prob = output
            self._l_obs = self.input_tensor

            self._f_prob = tensor_utils.compile_function(
                [self.input_tensor],
                output
            )
            self._dist = Categorical(env_spec.action_space.n)
            # pre-update policy
            self._init_f_dist = self._f_prob

            self._cur_f_dist = self._init_f_dist
            ####
            # self.init_flr_full = 0.5

            super(CategoricalMLPPolicy, self).__init__(env_spec)
            LayersPowered.__init__(self, [output])

    @property
    def vectorized(self):
        return True

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars=None,is_training=True, all_params=None):
        return_params = True
        if all_params is None:
            return_params=False
            all_params = self.all_params
        output  = self._forward(obs_var, all_params, is_training)
        if return_params:
            return dict(prob=output), all_params
        else:
            return dict(prob=output)

    @overrides
    def dist_info(self, obs, state_infos=None):
        return dict(prob=self._f_prob(obs))

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        prob = self._f_prob([flat_obs])[0]
        action = self.action_space.weighted_sample(prob)
        return action, dict(prob=prob)

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        probs = self._f_prob(flat_obs)
        actions = list(map(self.action_space.weighted_sample, probs))
        return actions, dict(prob=probs)

    @property
    def distribution(self):
        return self._dist


    ##########################
    def set_init_surr_obj(self, input_list, surr_objs_tensor):
        """ Set the surrogate objectives used the update the policy
        """
        self.input_list_for_grad = input_list
        self.surr_objs = surr_objs_tensor

    def compute_updated_dists(self, samples):
        """ Compute fast gradients once per iteration and pull them out of tensorflow for sampling with the post-update policy.
        """
        num_tasks = len(samples)
        # param_keys = self.all_params.keys()
        param_keys_temp = list(self.all_params.keys())
        param_keys = []
        for key in param_keys_temp:
            if 'stepsize' not in key:
                param_keys.append(key)

        update_param_keys = param_keys
        no_update_param_keys = []
        sess = tf.get_default_session()

        obs_list, action_list, adv_list = [], [], []
        for i in range(num_tasks):
            inputs = ext.extract(samples[i],
                    'observations', 'actions', 'advantages')
            obs_list.append(inputs[0])
            action_list.append(inputs[1])
            adv_list.append(inputs[2])

        inputs = obs_list + action_list + adv_list
        #inputs = theta0_dist_info_list + theta_l_dist_info_list + obs_list + action_list + adv_list

        # To do a second update, replace self.all_params below with the params that were used to collect the policy.
        init_param_values = None
        if self.all_param_vals is not None:
            init_param_values = self.get_variable_values(self.all_params)

        step_size = self.step_size
        for i in range(num_tasks):
            if self.all_param_vals is not None:
                self.assign_params(self.all_params, self.all_param_vals[i])

        step_sizes_sym = {}
        for key in param_keys:
            step_sizes_sym[key] = self.all_params[key + '_stepsize']

        if 'all_fast_params_tensor' not in dir(self):
            # make computation graph once
            self.all_fast_params_tensor = []
            for i in range(num_tasks):
                gradients = dict(zip(update_param_keys, tf.gradients(self.surr_objs[i], [self.all_params[key] for key in update_param_keys])))
                fast_params_tensor = OrderedDict(zip(update_param_keys, [self.all_params[key] - step_sizes_sym[key]*gradients[key] for key in update_param_keys]))
                for k in no_update_param_keys:
                    fast_params_tensor[k] = self.all_params[k]
                self.all_fast_params_tensor.append(fast_params_tensor)

        # pull new param vals out of tensorflow, so gradient computation only done once ## first is the vars, second the values
        # these are the updated values of the params after the gradient step

        self.all_param_vals = sess.run(self.all_fast_params_tensor, feed_dict=dict(list(zip(self.input_list_for_grad, inputs))))

        if init_param_values is not None:
            self.assign_params(self.all_params, init_param_values)

        outputs = []
        self._cur_f_dist_i = {}
        inputs = tf.split(self.input_tensor, num_tasks, 0)
        for i in range(num_tasks):
            # TODO - use a placeholder to feed in the params, so that we don't have to recompile every time.
            task_inp = inputs[i]
            info, _ = self.dist_info_sym(task_inp, dict(), all_params=self.all_param_vals[i],
                    is_training=False)

            outputs.append([info['prob']])

        self._cur_f_dist = tensor_utils.compile_function(
            inputs = [self.input_tensor],
            outputs = outputs,
        )
        #logger.record_tabular("ComputeUpdatedDistTime", total_time)

    def get_variable_values(self ,tensor_dict):
            sess = tf.get_default_session()
            result = sess.run(tensor_dict)
            return result

    def assign_params(self, tensor_dict, param_values):
        if 'assign_placeholders' not in dir(self):
            # make computation graph, if it doesn't exist; then cache it for future use.
            self.assign_placeholders = {}
            self.assign_ops = {}
            for key in tensor_dict.keys():
                self.assign_placeholders[key] = tf.placeholder(tf.float32)
                self.assign_ops[key] = tf.assign(tensor_dict[key], self.assign_placeholders[key])

        feed_dict = {self.assign_placeholders[key]:param_values[key] for key in tensor_dict.keys()}
        sess = tf.get_default_session()
        sess.run(self.assign_ops, feed_dict)

    def switch_to_init_dist(self):
        # switch cur policy distribution to pre-update policy
        self._cur_f_dist = self._init_f_dist
        self._cur_f_dist_i = None
        self.all_param_vals = None



    def updated_dist_info_sym(self ,task_id ,surr_obj ,new_obs_var ,params_dict=None ,is_training=True):
        """ symbolically create MAML graph, for the meta-optimization, only called at the beginning of
        meta-training.
        Called more than once if you want to do more than one grad step.
        """
        old_params_dict = params_dict

        step_size = self.step_size

        if old_params_dict == None:
            old_params_dict = self.all_params
        # param_keys = self.all_params.keys()
        param_keys_temp = list(self.all_params.keys())
        param_keys = []

        for key in param_keys_temp:
            if 'stepsize' not in key:
                param_keys.append(key)

        update_param_keys = param_keys
        no_update_param_keys = []

        grads = tf.gradients(surr_obj ,[old_params_dict[key] for key in update_param_keys])
        if self.stop_grad:
            grads = [tf.stop_gradient(grad) for grad in grads]

        gradients = dict(zip(update_param_keys ,grads))

        params_dict = dict(zip(update_param_keys ,
                               [old_params_dict[key] - self.all_params[key + '_stepsize'] * gradients[key] for
                                key in update_param_keys]))
        for k in no_update_param_keys:
            params_dict[k] = old_params_dict[k]

        return self.dist_info_sym(new_obs_var ,all_params=params_dict ,is_training=is_training)

    def get_params_internal(self ,all_params=False ,**tags):
        if tags.get('trainable' ,False):
            params = tf.trainable_variables()
        else:
            params = tf.global_variables()

        params = [p for p in params if p.name.startswith('mean_network')]
        # params = [p for p in params if p.name.startswith('mean_network') or p.name.startswith(
        # 'output_std_param')]
        params = [p for p in params if 'Adam' not in p.name]
        params = [p for p in params if 'main_optimizer' not in p.name]

        return params

    # This makes all of the parameters.
    def create_MLP(self ,name ,output_dim ,hidden_sizes ,
                   hidden_W_init=tf_layers.xavier_initializer() ,hidden_b_init=tf.zeros_initializer() ,
                   output_W_init=tf_layers.xavier_initializer() ,output_b_init=tf.zeros_initializer() ,
                   weight_normalization=False ,
                   ):
        all_params = OrderedDict()

        cur_shape = self.input_shape
        with tf.variable_scope(name):
            for idx ,hidden_size in enumerate(hidden_sizes):
                W ,b ,cur_shape = make_dense_layer(
                    cur_shape ,
                    num_units=hidden_size ,
                    name="hidden_%d" % idx ,
                    W=hidden_W_init ,
                    b=hidden_b_init ,
                    weight_norm=weight_normalization ,
                )
                all_params['W' + str(idx)] = W
                all_params['b' + str(idx)] = b
                all_params['W' + str(idx) + '_stepsize'] = tf.Variable(self.init_flr_full * tf.ones_like(W) ,
                                                                       name="W" + str(idx) + '_stepsize')
                all_params['b' + str(idx) + '_stepsize'] = tf.Variable(self.init_flr_full * tf.ones_like(b) ,
                                                                       name="b" + str(idx) + '_stepsize')
            W ,b ,_ = make_dense_layer(
                cur_shape ,
                num_units=output_dim ,
                name='output' ,
                W=output_W_init ,
                b=output_b_init ,
                weight_norm=weight_normalization ,
            )
            all_params['W' + str(len(hidden_sizes))] = W
            all_params['b' + str(len(hidden_sizes))] = b
            all_params['W' + str(len(hidden_sizes)) + '_stepsize'] = tf.Variable(
                self.init_flr_full * tf.ones_like(W) ,name="W" + str(len(hidden_sizes)) + '_stepsize')
            all_params['b' + str(len(hidden_sizes)) + '_stepsize'] = tf.Variable(
                self.init_flr_full * tf.ones_like(b) ,name="b" + str(len(hidden_sizes)) + '_stepsize')

            #all_params['bias_transformation'] = tf.Variable(tf.ones(self.latent_dim) ,
            #                                                name="bias_transformation")
            #all_params['bias_transformation_stepsize'] = tf.Variable(
            #    self.init_flr_full * tf.ones_like(all_params['bias_transformation']) ,
            #    name="bias_transformation_stepsize")

        return all_params

    def forward_MLP(self ,name ,all_params ,input_tensor=None ,
                    batch_normalization=False ,reuse=True ,is_training=False):
        # is_training and reuse are for batch norm, irrelevant if batch_norm set to False
        # set reuse to False if the first time this func is called.
        with tf.variable_scope(name):
            if input_tensor is None:
                l_in = make_input(shape=(None ,self.obs_dim ,) ,input_var=None ,name='input')
            else:
                l_in = input_tensor

            #bs = tf.shape(l_in)[0]
            #conc_bias = tf.tile(all_params['bias_transformation'][None ,:] ,(bs ,1))
            l_hid = l_in# tf.concat([l_in ,conc_bias] ,axis=1)

            for idx in range(self.n_hidden):
                l_hid = forward_dense_layer(l_hid ,all_params['W' + str(idx)] ,all_params['b' + str(idx)] ,
                                            batch_norm=batch_normalization ,
                                            nonlinearity=self.hidden_nonlinearity ,
                                            scope=str(idx) ,reuse=reuse ,
                                            is_training=is_training
                                            )
            output = forward_dense_layer(l_hid ,all_params['W' + str(self.n_hidden)] ,
                                         all_params['b' + str(self.n_hidden)] ,
                                         batch_norm=False ,nonlinearity=self.output_nonlinearity ,
                                         )
            return l_in ,output

    def get_params(self ,all_params=False ,**tags):
        """
        Get the list of parameters (symbolically), filtered by the provided tags.
        Some common tags include 'regularizable' and 'trainable'
        """
        tag_tuple = tuple(sorted(list(tags.items()) ,key=lambda x:x[0]))
        if tag_tuple not in self._cached_params:
            self._cached_params[tag_tuple] = self.get_params_internal(all_params ,**tags)
        return self._cached_params[tag_tuple]

    def get_param_values(self ,all_params=False ,**tags):
        params = self.get_params(all_params ,**tags)
        param_values = tf.get_default_session().run(params)
        return flatten_tensors(param_values)

    def log_diagnostics(self ,paths ,prefix=''):
        pass


    def get_param_dtypes(self, all_params=False, **tags):
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_param_dtypes:
            params = self.get_params(all_params, **tags)
            param_values = tf.get_default_session().run(params)
            self._cached_param_dtypes[tag_tuple] = [val.dtype for val in param_values]
        return self._cached_param_dtypes[tag_tuple]

    def get_param_shapes(self, all_params=False, **tags):
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_param_shapes:
            params = self.get_params(all_params, **tags)
            param_values = tf.get_default_session().run(params)
            self._cached_param_shapes[tag_tuple] = [val.shape for val in param_values]
        return self._cached_param_shapes[tag_tuple]

    def set_param_values(self, flattened_params, all_params=False, **tags):
        debug = tags.pop("debug", False)
        param_values = unflatten_tensors(
            flattened_params, self.get_param_shapes(all_params, **tags))
        ops = []
        feed_dict = dict()
        for param, dtype, value in zip(
                self.get_params(all_params, **tags),
                self.get_param_dtypes(all_params, **tags),
                param_values):
            if param not in self._cached_assign_ops:
                assign_placeholder = tf.placeholder(dtype=param.dtype.base_dtype)
                assign_op = tf.assign(param, assign_placeholder)
                self._cached_assign_ops[param] = assign_op
                self._cached_assign_placeholders[param] = assign_placeholder
            ops.append(self._cached_assign_ops[param])
            feed_dict[self._cached_assign_placeholders[param]] = value.astype(dtype)
            if debug:
                print("setting value of %s" % param.name)
        tf.get_default_session().run(ops, feed_dict=feed_dict)

    def flat_to_params(self, flattened_params, all_params=False, **tags):
        return unflatten_tensors(flattened_params, self.get_param_shapes(all_params, **tags))

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        global load_params
        if load_params:
            d["params"] = self.get_param_values(all_params=True)
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        global load_params
        if load_params:
            tf.get_default_session().run(tf.variables_initializer(self.get_params(all_params=True)))
            self.set_param_values(d["params"], all_params=True)