from __future__ import division

import os
import warnings

import numpy as np
import keras.backend as K
import keras.optimizers as optimizers

from rl.core import Agent
from rl.util import *
from rl.random import OrnsteinUhlenbeckProcess
import json

from copy import deepcopy


from keras.callbacks import History

from rl.callbacks import TrainEpisodeLogger, TrainIntervalLogger, \
    Visualizer, CallbackList

from layer import set_episodic_phase, EpisodicNoiseDense


def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))


class MultiActorCriticAgent(Agent):
    def __init__(self, agent_number, actions_shape, observation_shape,
                 actor, critic, critic_action_input, memory,
                 gamma=.99, batch_size=32, nb_steps_warmup_critic=200,
                 nb_steps_warmup_actor=200,
                 train_interval=1, memory_interval=1, delta_range=None,
                 delta_clip=np.inf,
                 random_process=None,
                 custom_model_objects={
                     'EpisodicNoiseDense': EpisodicNoiseDense},
                 target_model_update=.001, **kwargs):
        if critic_action_input not in critic.input:
            raise ValueError('Critic "{}" does not have designated \
                action input "{}".'.format(
                critic, critic_action_input))
        if not hasattr(critic.input, '__len__') or len(critic.input) < 2:
            raise ValueError(
                'Critic "{}" does not have enough inputs. The critic must \
                have at exactly two inputs, one for the action and one for \
                the observation.'.format(critic))
        self.agent_number = agent_number
        super(MultiActorCriticAgent, self).__init__(**kwargs)

        # Soft vs hard target model updates.
        if target_model_update < 0:
            raise ValueError('`target_model_update` must be >= 0.')
        elif target_model_update >= 1:
            # Hard update every `target_model_update` steps.
            target_model_update = int(target_model_update)
        else:
            # Soft update with `(1 - target_model_update) * old +
            # target_model_update * new`.
            target_model_update = float(target_model_update)

        if delta_range is not None:
            warnings.warn(
                '`delta_range` is deprecated. Please use `delta_clip` \
                instead, which takes a single scalar. For now we\'re falling \
                back to `delta_range[1] = {}`'.format(delta_range[1]))
            delta_clip = delta_range[1]

        # Parameters.
        self.observation_shape = observation_shape
        self.actions_shape = actions_shape
        self.nb_steps_warmup_actor = nb_steps_warmup_actor
        self.nb_steps_warmup_critic = nb_steps_warmup_critic
        self.random_process = random_process
        self.delta_clip = delta_clip
        self.gamma = gamma
        self.target_model_update = target_model_update
        self.batch_size = batch_size
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.custom_model_objects = custom_model_objects

        # Related objects.
        self.actor = actor
        self.critic = critic
        self.critic_action_input = critic_action_input
        self.critic_action_input_idx = self.critic.input.index(
            critic_action_input)
        self.memory = memory

        # State.
        self.compiled = False
        self.reset_states()

    def record_steps(self, env, nb_episodes=1, nb_max_episode_steps=1000, action_repetition=1,
                     file_path='./record_steps.json'):
        if not self.compiled:
            raise RuntimeError('Your tried to test your agent but it hasn\'t been compiled yet. Please call `compile()` before `test()`.')

        self.training = False
        self.step = 0
        print(self.actor.layers[3], self.actor.layers[9])
        get_hiddens = K.function(
                [self.actor.layers[0].input, K.learning_phase()],
                [self.actor.layers[3].output, self.actor.layers[9].output]
            )

        for episode in range(nb_episodes):
            episode_reward = 0.
            episode_step = 0

            # Obtain the initial observation by resetting the environment.
            self.reset_states()
            observation = deepcopy(env.reset())
            if self.processor is not None:
                observation = self.processor.process_observation(observation)
            assert observation is not None
            # Run the episode until we're done.
            done = False
            # todo: record the hidden outputs
            while not done:
                hiddens = get_hiddens([np.array([[observation]]), 0])[0]
                action = self.forward(observation)
                reward = 0.
                accumulated_info = {}
                for _ in range(action_repetition):
                    observation, r, d, info = env.step(action)
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, r, d, info = self.processor.process_step(observation, r, d, info)
                    reward += r
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    if d:
                        done = True
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    done = True
                self.backward(reward, terminal=done)
                episode_reward += reward
                units_myself = {uid: [ut.health, ut.groundCD, ut.x, ut.y] for uid, ut in env.state['units_myself'].items()}
                units_enemy = {uid: [ut.health, ut.groundCD, ut.x, ut.y] for uid, ut in env.state['units_enemy'].items()}
                step_logs = {
                    'action': action.tolist(),
                    'observation': observation.tolist(),
                    'reward': reward.tolist(),
                    'episode': episode,
                    'info': accumulated_info,
                    'units_myself': units_myself,
                    'units_enemy': units_enemy,
                    'attack_cmds': deepcopy(env.cmds_attack),
                    'hiddens': [h.tolist() for h in hiddens],
                    'step': self.step,
                    'episode_step': episode_step
                }
                with open(file_path, 'a') as f:
                    f.write(json.dumps(step_logs))
                    f.write('\n')
                episode_step += 1
                self.step += 1

            self.forward(observation)
            self.backward(0., terminal=False)

            # Report end of episode.
            episode_logs = {
                'episode_reward': episode_reward,
                'nb_steps': episode_step,
            }

        return True

    def test(self, env, nb_episodes=1, action_repetition=1, callbacks=None, visualize=True,
             nb_max_episode_steps=None, nb_max_start_steps=0, start_step_policy=None, verbose=1):
        if not self.compiled:
            raise RuntimeError('Your tried to test your agent but it hasn\'t been compiled yet. Please call `compile()` before `test()`.')
        if action_repetition < 1:
            raise ValueError('action_repetition must be >= 1, is {}'.format(action_repetition))

        self.training = False
        self.step = 0

        callbacks = [] if not callbacks else callbacks[:]

        if verbose >= 1:
            callbacks += [TestLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_episodes': nb_episodes,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)

        self._on_test_begin()
        callbacks.on_train_begin()
        for episode in range(nb_episodes):
            callbacks.on_episode_begin(episode)
            episode_reward = 0.
            episode_step = 0

            # Obtain the initial observation by resetting the environment.
            self.reset_states()
            observation = deepcopy(env.reset())
            if self.processor is not None:
                observation = self.processor.process_observation(observation)
            assert observation is not None

            # Perform random starts at beginning of episode and do not record them into the experience.
            # This slightly changes the start position between games.
            # nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(nb_max_start_steps)
            # for _ in range(nb_random_start_steps):
            #     if start_step_policy is None:
            #         action = env.action_space.sample()
            #     else:
            #         action = start_step_policy(observation)
            #     callbacks.on_action_begin(action)
            #     observation, r, done, info = env.step(action)
            #     observation = deepcopy(observation)
            #     if self.processor is not None:
            #         observation, r, done, info = self.processor.process_step(observation, r, done, info)
            #     callbacks.on_action_end(action)
            #     if done:
            #         warnings.warn('Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(nb_random_start_steps))
            #         observation = deepcopy(env.reset())
            #         if self.processor is not None:
            #             observation = self.processor.process_observation(observation)
            #         break

            # Run the episode until we're done.
            done = False
            while not done:
                callbacks.on_step_begin(episode_step)

                action = self.forward(observation)
                reward = 0.
                accumulated_info = {}
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation, r, d, info = env.step(action)
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, r, d, info = self.processor.process_step(observation, r, d, info)
                    callbacks.on_action_end(action)
                    reward += r
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    if d:
                        done = True
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    done = True
                self.backward(reward, terminal=done)
                episode_reward += reward

                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'episode': episode,
                    'info': accumulated_info,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1

            # We are in a terminal state but the agent hasn't yet seen it. We therefore
            # perform one more forward-backward call and simply ignore the action before
            # resetting the environment. We need to pass in `terminal=False` here since
            # the *next* state, that is the state of the newly reset environment, is
            # always non-terminal by convention.
            self.forward(observation)
            self.backward(0., terminal=False)

            # Report end of episode.
            episode_logs = {
                'episode_reward': episode_reward,
                'nb_steps': episode_step,
            }
            callbacks.on_episode_end(episode, episode_logs)
        callbacks.on_train_end()
        self._on_test_end()

        return history

    def fit(self, env, nb_steps, action_repetition=1,
            callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0,
            start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None):
        if not self.compiled:
            raise RuntimeError(
                'Your tried to fit your agent but it hasn\'t been compiled \
                yet. Please call `compile()` before `fit()`.')
        if action_repetition < 1:
            raise ValueError(
                'action_repetition must be >= 1, is\
                 {}'.format(action_repetition))

        self.training = True

        callbacks = [] if not callbacks else callbacks[:]

        if verbose == 1:
            callbacks += [TrainIntervalLogger(interval=log_interval)]
        elif verbose > 1:
            callbacks += [TrainEpisodeLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_steps': nb_steps,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)
        self._on_train_begin()
        callbacks.on_train_begin()

        episode = 0
        self.step = 0
        observation = None
        episode_reward = None
        episode_step = None
        did_abort = False
        try:
            while self.step < nb_steps:
                if observation is None:  # start of a new episode
                    callbacks.on_episode_begin(episode)
                    episode_step = 0
                    episode_reward = 0.

                    # Obtain the initial observation by resetting the
                    # environment.
                    self.reset_states()
                    observation = deepcopy(env.reset())
                    if self.processor is not None:
                        observation = self.processor.process_observation(
                            observation)
                    assert observation is not None

                    nb_random_start_steps = 0 if nb_max_start_steps == 0 \
                        else np.random.randint(
                            nb_max_start_steps)
                    for _ in range(nb_random_start_steps):
                        if start_step_policy is None:
                            action = env.action_space.sample()
                        else:
                            action = start_step_policy(observation)
                        callbacks.on_action_begin(action)
                        observation, reward, done, info = env.step(action)
                        observation = deepcopy(observation)
                        if self.processor is not None:
                            observation, reward, done, info =\
                                self.processor.process_step(
                                    observation, reward, done, info)
                        callbacks.on_action_end(action)
                        if done:
                            warnings.warn('Env ended before {} random \
                                steps could be performed at the start. You \
                                should probably lower the `nb_max_start_steps`\
                                 parameter.'.format(
                                nb_random_start_steps))
                            observation = deepcopy(env.reset())
                            if self.processor is not None:
                                observation = \
                                    self.processor.process_observation(
                                        observation)
                            break

                # At this point, we expect to be fully initialized.
                assert episode_reward is not None
                assert episode_step is not None
                assert observation is not None

                # Run a single step.
                callbacks.on_step_begin(episode_step)
                action = self.forward(observation)
                reward = 0.
                accumulated_info = {}
                done = False
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation, r, done, info = env.step(action)
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, r, done, info = \
                            self.processor.process_step(
                                observation, r, done, info)
                    # for key, value in info.items():
                    #     if not np.isreal(value):
                    #         continue
                    #     if key not in accumulated_info:
                    #         accumulated_info[key] = np.zeros_like(value)
                    #     accumulated_info[key] += value
                    callbacks.on_action_end(action)
                    reward += r
                    if done:
                        break
                if nb_max_episode_steps and \
                        episode_step >= nb_max_episode_steps - 1:
                    # Force a terminal state.
                    done = True
                metrics = self.backward(reward, terminal=done)
                episode_reward += reward

                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'metrics': metrics,
                    'episode': episode,
                    'info': accumulated_info,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1

                # if self.step % 20000 == 0 and self.random_process is not None:
                    # self.random_process = OrnsteinUhlenbeckProcess(size=env.action_space.shape, theta=.25, mu=0., sigma=self.random_process.sigma + 0.01)

                if done:
                    set_episodic_phase(1)
                    self.forward(observation)
                    self.backward([0.] * self.agent_number, terminal=False)

                    # This episode is finished, report and reset.
                    episode_logs = {
                        'episode_reward': episode_reward,
                        'nb_episode_steps': episode_step,
                        'nb_steps': self.step,
                    }
                    callbacks.on_episode_end(episode, episode_logs)
                    episode += 1
                    observation = None
                    episode_step = None
                    episode_reward = None
        except KeyboardInterrupt:
            did_abort = True
        callbacks.on_train_end(logs={'did_abort': did_abort})
        self._on_train_end()

        return history

    @property
    def uses_learning_phase(self):
        return self.actor.uses_learning_phase or \
            self.critic.uses_learning_phase

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]

        if type(optimizer) in (list, tuple):
            if len(optimizer) != 2:
                raise ValueError(
                    'More than two optimizers provided. Please only \
                    provide a maximum of two optimizers, the first one \
                    for the actor and the second one for the critic.')
            actor_optimizer, critic_optimizer = optimizer
        else:
            actor_optimizer = optimizer
            critic_optimizer = clone_optimizer(optimizer)
        if type(actor_optimizer) is str:
            actor_optimizer = optimizers.get(actor_optimizer)
        if type(critic_optimizer) is str:
            critic_optimizer = optimizers.get(critic_optimizer)
        assert actor_optimizer != critic_optimizer

        if len(metrics) == 2 and hasattr(metrics[0], '__len__') \
                and hasattr(metrics[1], '__len__'):
            actor_metrics, critic_metrics = metrics
        else:
            actor_metrics = critic_metrics = metrics

        def clipped_error(y_true, y_pred):
            return K.mean(huber_loss(y_true, y_pred, self.delta_clip), axis=-1)

        self.target_actor = clone_model(self.actor, self.custom_model_objects)
        self.target_actor.compile(optimizer='sgd', loss='mse')
        self.target_critic = clone_model(
            self.critic, self.custom_model_objects)
        self.target_critic.compile(optimizer='sgd', loss='mse')

        self.actor.compile(optimizer='sgd', loss='mse')

        # Compile the critic.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently
            # soft-update the target model.
            critic_updates = get_soft_target_model_updates(
                self.target_critic, self.critic, self.target_model_update)
            critic_optimizer = AdditionalUpdatesOptimizer(
                critic_optimizer, critic_updates)
        self.critic.compile(optimizer=critic_optimizer,
                            loss=clipped_error, metrics=critic_metrics)

        # Combine actor and critic so that we can get the policy gradient.
        combined_inputs = []
        critic_inputs = []
        for i in self.critic.input:
            if i == self.critic_action_input:
                combined_inputs.append(self.actor.output)
            else:
                combined_inputs.append(i)
                critic_inputs.append(i)
        combined_output = self.critic(combined_inputs)
        if K._BACKEND == 'tensorflow':
            grads = K.gradients(combined_output, self.actor.trainable_weights)
            grads = [g / float(self.batch_size)
                     for g in grads]  # since TF sums over the batch
        elif K._BACKEND == 'theano':
            import theano.tensor as T
            grads = T.jacobian(combined_output.flatten(),
                               self.actor.trainable_weights)
            grads = [K.mean(g, axis=0) for g in grads]
        else:
            raise RuntimeError(
                'Unknown Keras backend "{}".'.format(K._BACKEND))

        # We now have the gradients (`grads`) of the combined model wrt
        # to the actor's weights and
        # the output (`output`). Compute the necessary updates using a clone of
        # the actor's optimizer.
        clipnorm = getattr(actor_optimizer, 'clipnorm', 0.)
        clipvalue = getattr(actor_optimizer, 'clipvalue', 0.)

        def get_gradients(loss, params):
            # We want to follow the gradient, but the optimizer
            # goes in the opposite direction to
            # minimize loss. Hence the double inversion.
            assert len(grads) == len(params)
            modified_grads = [-g for g in grads]
            if clipnorm > 0.:
                norm = K.sqrt(sum([K.sum(K.square(g))
                                   for g in modified_grads]))
                modified_grads = [optimizers.clip_norm(
                    g, clipnorm, norm) for g in modified_grads]
            if clipvalue > 0.:
                modified_grads = [K.clip(g, -clipvalue, clipvalue)
                                  for g in modified_grads]
            return modified_grads

        actor_optimizer.get_gradients = get_gradients
        updates = actor_optimizer.get_updates(
            self.actor.trainable_weights, self.actor.constraints, None)
        if self.target_model_update < 1.:
            # Include soft target model updates.
            updates += get_soft_target_model_updates(
                self.target_actor, self.actor, self.target_model_update)
        # include other updates of the actor, e.g. for BN
        updates += self.actor.updates


        # Finally, combine it all into a callable function.
        inputs = self.actor.inputs[:] + critic_inputs
        if self.uses_learning_phase:
            inputs += [K.learning_phase()]
        self.actor_train_fn = K.function(
            inputs, [self.actor.output], updates=updates)
        self.actor_optimizer = actor_optimizer

        self.compiled = True

    def load_weights(self, filepath):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.load_weights(actor_filepath)
        self.critic.load_weights(critic_filepath)
        self.update_target_models_hard()

    def save_weights(self, filepath, overwrite=False):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.save_weights(actor_filepath, overwrite=overwrite)
        self.critic.save_weights(critic_filepath, overwrite=overwrite)

    def update_target_models_hard(self):
        self.target_critic.set_weights(self.critic.get_weights())
        self.target_actor.set_weights(self.actor.get_weights())

    # TODO: implement pickle

    def reset_states(self):
        if self.random_process is not None:
            self.random_process.reset_states()
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.actor.reset_states()
            self.critic.reset_states()
            self.target_actor.reset_states()
            self.target_critic.reset_states()

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def select_action(self, state):
        batch = self.process_state_batch([state])
        action = self.actor.predict_on_batch(batch)
        # assert action.shape == self.actions_shape

        # Apply noise, if a random process is set.
        if self.training and self.random_process is not None:
            noise = self.random_process.sample()
            noise = noise.reshape(action.shape)
            # assert noise.shape == action.shape
            action += noise

        return action


    def forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        action = self.select_action(state)  # TODO: move this into policy
        if self.processor is not None:
            action = self.processor.process_action(action)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action

    @property
    def metrics_names(self):
        names = self.critic.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names

    def backward(self, reward, terminal=False):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        can_train_either = self.step > self.nb_steps_warmup_critic or self.step > self.nb_steps_warmup_actor
        if can_train_either and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized
            # implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                zeros = [0.] * self.agent_number
                ones = [1.] * self.agent_number
                terminal1_batch.append(zeros if e.terminal1 else ones)

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            action_batch = np.array(action_batch)
            # assert reward_batch.shape == (self.batch_size, self.agent_number)
            # assert terminal1_batch.shape == reward_batch.shape
            # assert action_batch.shape == (
            #     self.batch_size, 1) + self.actions_shape

            # Update critic, if warm up is over.
            if self.step > self.nb_steps_warmup_critic:
                target_actions = self.target_actor.predict_on_batch(
                    state1_batch)
                assert target_actions.shape == (
                    self.batch_size,) + self.actions_shape
                if len(self.critic.inputs) >= 3:
                    state1_batch_with_action = state1_batch[:]
                else:
                    state1_batch_with_action = [state1_batch]
                state1_batch_with_action.insert(
                    self.critic_action_input_idx, target_actions)
                target_q_values = self.target_critic.predict_on_batch(
                    state1_batch_with_action)
                assert target_q_values.shape == (
                    self.batch_size, self.agent_number)

                # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target ys accordingly,
                # but only for the affected output units (as given by
                # action_batch).
                discounted_reward_batch = self.gamma * target_q_values
                discounted_reward_batch *= terminal1_batch
                if not discounted_reward_batch.shape == reward_batch.shape:
                    print('discounted_reward_batch.shape step reward_batch.shape')

                targets = (
                    reward_batch + discounted_reward_batch).reshape(self.batch_size, self.agent_number)

                # Perform a single batch update on the critic network.
                if len(self.critic.inputs) >= 3:
                    state0_batch_with_action = state0_batch[:]
                else:
                    state0_batch_with_action = [state0_batch]
                state0_batch_with_action.insert(self.critic_action_input_idx,
                                                action_batch.reshape(
                                                    (self.batch_size,) + self.actions_shape))
                metrics = self.critic.train_on_batch(
                    state0_batch_with_action, targets)
                if self.processor is not None:
                    metrics += self.processor.metrics

            # Update actor, if warm up is over.
            if self.step > self.nb_steps_warmup_actor:
                # TODO: implement metrics for actor
                if len(self.actor.inputs) >= 2:
                    inputs = state0_batch[:] + state0_batch[:]
                else:
                    inputs = [state0_batch, + state0_batch]
                if self.uses_learning_phase:
                    inputs += [self.training]
                action_values = self.actor_train_fn(inputs)[0]
                assert action_values.shape == (
                    self.batch_size,) + self.actions_shape

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_models_hard()

        return metrics
