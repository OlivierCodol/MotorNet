import numpy as np
import torch as th
import gymnasium as gym
from typing import Any
from collections.abc import Sequence
from motornet.effector import Effector

class Environment(gym.Env, th.nn.Module):
  """Base class for environments.

  Args:
    effector: :class:`motornet.effector.Effector` object class or subclass. This is the effector that will evolve in
      the environment.
    q_init: `Tensor` or `numpy.ndarray`, the desired initial joint states for the environment, if a single set
      of pre-defined initial joint states is desired. If `None`, the initial joint states will be drawn from the
      :class:`motornet.nets.layers.Network.get_initial_state` method at each call of :meth:`generate`. This parameter
      will be ignored on :meth:`generate` calls where a `joint_state` is provided as input
      argument.
    name: `String`, the name of the environment object instance.
    differentiable: `Boolean`, whether the environment will be differentiable or not. This will usually be useful for
      reinforcement learning, where the differentiability is not needed.
    max_ep_duration: `Float`, the maximum duration of an episode, in seconds.
    action_noise: `Float` of `list`, the standard deviation of the Gaussian noise added to the action input at each step of the 
      simulation. If this is a `list`, it should have as many elements as the action vector and will 
      indicate the standard deviation of each action element independently.
    obs_noise: `Float` or `list`, the standard deviation of the Gaussian noise added to the observation vector at each
      step of the simulation. If this is a `list`, it should have as many elements as the observation vector and will 
      indicate the standard deviation of each observation element independently.
    action_frame_stacking: `Integer`, the number of past action steps to add to the observation vector.
    proprioception_delay: `Float`, the delay in seconds for the proprioceptive feedback to be added to the observation 
      vector. If `None`, no delay will occur.
    vision_delay: `Float`, the delay in seconds for the visual feedback to be added to the observation vector. If 
      `None`, no delay will occur.
    proprioception_noise: `Float`, the standard deviation of the Gaussian noise added to the proprioceptive feedback at
      each step of the simulation.
    vision_noise: `Float`, the standard deviation of the Gaussian noise added to the visual feedback at each step of the
      simulation.
    **kwargs: This is passed as-is to the :class:`torch.nn.Module` parent class.
  """
  def __init__(
    self,
    effector: Effector,
    q_init: np.ndarray | th.Tensor | Sequence | None = None,
    name: str = 'Env',
    differentiable: bool = True,
    max_ep_duration: float = 1.,
    action_noise: float | Sequence[float] = 0.,
    obs_noise: float | Sequence[float] = 0.,
    action_frame_stacking: int = 0,
    proprioception_delay: float | None = None,
    vision_delay: float | None = None,
    proprioception_noise: float = 0.,
    vision_noise: float = 0.,
    **kwargs,
  ):
    
    super().__init__(**kwargs)

    self.__name__ = name
    self.effector = effector
    self.dt = self.effector.dt
    self.differentiable = differentiable
    self.max_ep_duration = max_ep_duration
    self.q_init = q_init
    self._action_noise = action_noise
    self._obs_noise = obs_noise
    self.proprioception_noise = [proprioception_noise]
    self.vision_noise = [vision_noise]
    self.action_frame_stacking = action_frame_stacking

    # default is no delay
    proprioception_delay = self.dt if proprioception_delay is None else proprioception_delay
    vision_delay = self.dt if vision_delay is None else vision_delay

    self.proprioception_delay = round(proprioception_delay / self.dt)
    self.vision_delay = round(vision_delay / self.dt)
    assert np.isclose(proprioception_delay / self.dt, self.proprioception_delay), f'`proprioception_delay` was {proprioception_delay} and `dt` was {self.dt}'
    assert np.isclose(vision_delay / self.dt, self.vision_delay), f'`vision_delay` was {vision_delay} and `dt` was {self.dt}'
    self.goal_delay = 1
    
    self._build_spaces()

  def detach(self, x) -> np.ndarray:
    return x.cpu().detach().numpy() if th.is_tensor(x) else x
  
  def _build_spaces(self):
    self.action_space = gym.spaces.Box(low=0., high=1., shape=(self.effector.input_dim,), dtype=np.float32)

    obs, _ = self.reset(options={"deterministic": True})
    self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs.shape[-1],), dtype=np.float32)

    def handle_noise_arg(noise: Sequence[float] | float, space: gym.spaces.Box) -> Sequence[float]:
      return noise if isinstance(noise, Sequence) else [noise] * space.shape[0] 

    self.action_noise = handle_noise_arg(self._action_noise, self.action_space)
    self.obs_noise = handle_noise_arg(self._obs_noise, self.observation_space)
  
  # # # ========================================================================================
  # # # ========================================================================================
  # # # The methods below are those MOST likely to be overwritten by users creating custom tasks
  # # # ========================================================================================
  # # # ========================================================================================
  def get_proprioception(self) -> th.Tensor:
    """
    Returns a `(batch_size, n_features)` `tensor` containing the instantaneous (non-delayed) proprioceptive 
    feedback. By default, this is the normalized muscle length for each muscle, followed by the normalized
    muscle velocity for each muscle as well. `.i.i.d.` Gaussian noise is added to each element in the `tensor`,
    using the :attr:`proprioception_noise` attribute.
    """
    if not hasattr(self, '_mlen_idx'):
      self._mlen_idx = self.muscle.state_name.index('muscle length')
    mlen = self.states["muscle"][:, self._mlen_idx:self._mlen_idx + 1] / self.muscle.l0_ce
    if not hasattr(self, '_mvel_idx'):
      self._mvel_idx = self.muscle.state_name.index('muscle velocity')
    mvel = self.states["muscle"][:, self._mvel_idx:self._mvel_idx + 1] / self.muscle.vmax
    prop = th.cat([mlen, mvel], dim=-1).squeeze(dim=1)
    return self.apply_noise(prop, self.proprioception_noise)

  def get_vision(self) -> th.Tensor:
    """
    Returns a `(batch_size, n_features)` `tensor` containing the instantaneous (non-delayed) visual 
    feedback. By default, this is the cartesian position of the end-point effector, that is, the fingertip.
    `.i.i.d.` Gaussian noise is added to each element in the `tensor`, using the
    :attr:`vision_noise` attribute.
    """
    vis = self.states["fingertip"]
    return self.apply_noise(vis, self.vision_noise)

  def get_obs(self, action=None, deterministic: bool = False) -> th.Tensor | np.ndarray:
    """
    Returns a `(batch_size, n_features)` `tensor` containing the (potientially time-delayed) observations.
    By default, this is the task goal, followed by the output of the :meth:`get_proprioception()` method, 
    the output of the :meth:`get_vision()` method, and finally the last :attr:`action_frame_stacking` action sets,
    if a non-zero `action_frame_stacking` keyword argument was passed at initialization of this class instance.
    `.i.i.d.` Gaussian noise is added to each element in the `tensor`,
    using the :attr:`obs_noise` attribute.
    """
    # There is a major problem with this implementation in that `vision_noise` and `proprioception_noise`
    # are not modulated by `deterministic` in update_obs_buffer()
    self.update_obs_buffer(action=action)

    obs_as_list = [
      self.obs_buffer["goal"][0],
      self.obs_buffer["vision"][0],
      self.obs_buffer["proprioception"][0],
      ] + self.obs_buffer["action"]
    
    obs = th.cat(obs_as_list, dim=-1)

    if deterministic is False:
      obs = self.apply_noise(obs, self.obs_noise)

    return obs if self.differentiable else self.detach(obs)

  def step(
      self,
      action: th.Tensor | np.ndarray | Sequence[float],
      *,
      deterministic: bool = False,
      **kwargs,
    ) -> tuple[th.Tensor | np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
    """
    Perform one simulation step. This method is likely to be overwritten by any subclass to implement user-defined 
    computations, such as reward value calculation for reinforcement learning, custom truncation or termination
    conditions, or time-varying goals.
    
    Args:
      action: `Tensor` or `numpy.ndarray`, the input drive to the actuators.
      deterministic: `Boolean`, whether observation, action, proprioception, and vision noise are applied.
      **kwargs: This is passed as-is to the :meth:`motornet.effector.Effector.step()` call. This is maily useful to pass
      `endpoint_load` or `joint_load` kwargs.
  
    Returns:
      - The observation vector as `tensor` or `numpy.ndarray`, if the :class:`Environment` is set as differentiable or 
        not, respectively. It has dimensionality `(batch_size, n_features)`.
      - A `numpy.ndarray` with the reward information for the step, with dimensionality `(batch_size, 1)`. This is 
        `None` if the :class:`Environment` is set as differentiable. By default this always returns `0.` in the 
        :class:`Environment`.
      - A `boolean` indicating if the simulation has been terminated or truncated. If the :class:`Environment` is set as
        differentiable, this returns `True` when the simulation time reaches `max_ep_duration` provided at 
        initialization.
      - A `boolean` indicating if the simulation has been truncated early or not. This always returns `False` if the
        :class:`Environment` is set as differentiable.
      - A `dictionary` containing this step's information.
    """
    
    self.elapsed += self.dt

    action = action if isinstance(action, th.Tensor) else th.tensor(action, dtype=th.float32, device=self.device)
    noisy_action = action
    if deterministic is False:
      noisy_action = self.apply_noise(noisy_action, self.action_noise)
    
    self.effector.step(noisy_action, **kwargs)

    reward = self._get_reward(self.effector.batch_size)
    obs = self.get_obs(action=noisy_action, deterministic=deterministic)
    truncated = False if self.differentiable else False
    terminated = not truncated and (self.elapsed > self.max_ep_duration or bool(np.isclose(self.elapsed, self.max_ep_duration)))

    info = {
      "states": self._maybe_detach_states(),
      "action": action if self.differentiable else self.detach(action),
      "noisy action": noisy_action if self.differentiable else self.detach(noisy_action),
      "goal": self.goal if self.differentiable else self.detach(self.goal),
      }

    return obs, reward, terminated, truncated, info

  def _get_reward(self, batch_size: int):
    return np.zeros((batch_size, 1))

  def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
    """
    Initialize the task goal and :attr:`effector` states for a (batch of) simulation episode(s). The :attr:`effector`
    states (joint, cartesian, muscle, geometry) are initialized to be biomechanically compatible with each other.
    This method is likely to be overwritten by any subclass to implement user-defined computations, such as defining 
    a custom initial goal or initial states.

    Args:
      seed: `Integer`, the seed that is used to initialize the environment's PRNG (`np_random`).
        If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
        a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
        However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset.
        If you pass an integer, the PRNG will be reset even if it already exists.
        Usually, you want to pass an integer *right after the environment has been initialized and then never again*.
      options: `Dictionary`, optional kwargs specific to motornet environments. This is mainly useful to pass
        `batch_size`, `joint_state`, and `deterministic` kwargs if desired, as described below.

    Options:
      - **batch_size**: `Integer`, the desired batch size. Default: `1`.
      - **joint_state**: The joint state from which the other state values are inferred. If `None`, the `q_init` value 
        declared during the class instantiation will be used. If `q_init` is also `None`, random initial joint
        states are drawn, from which the other state values are inferred. Default: `None`.
      - **deterministic**: `Boolean`, whether observation, proprioception, and vision noise are applied.
        Default: `False`.

    Returns:
      - The observation vector as `tensor` or `numpy.ndarray`, if the :class:`Environment` is set as differentiable or 
        not, respectively. It has dimensionality `(batch_size, n_features)`.
      - A `dictionary` containing the initial step's information.
    """
    options = {} if options is None else options
    batch_size: int = options.get("batch_size", 1)
    joint_state: th.Tensor | np.ndarray | Sequence | None = options.get("joint_state", self.q_init)
    deterministic: bool = options.get("deterministic", False)

    joint_state = th.as_tensor(joint_state) if isinstance(joint_state, np.ndarray) else joint_state

    self.effector.reset(seed=seed, options={"batch_size": batch_size, "joint_state": joint_state})
    self.goal = self._get_goal(self.effector.batch_size)
    self.elapsed = 0.

    # initialize buffer
    assert isinstance(self.action_space, gym.spaces.Box), f'`action_space` is not appropriately initialized.'
    action = th.zeros((self.effector.batch_size, self.action_space.shape[0]), device=self.device)
    self.obs_buffer = {
      "proprioception": [self.get_proprioception() for _ in range(self.proprioception_delay)],
      "goal": [self.goal] * self.goal_delay,
      "vision": [self.get_vision() for _ in range(self.vision_delay)],
      "action": [action] * self.action_frame_stacking
    }

    action = action if self.differentiable else self.detach(action)

    obs = self.get_obs(deterministic=deterministic)
    info = {
        "states": self._maybe_detach_states(),
        "action": action,
        "noisy action": action,
        "goal": self.goal if self.differentiable else self.detach(self.goal),
        }
    return obs, info
  
  def _get_goal(self, batch_size: int) -> th.Tensor:
    return th.zeros((batch_size, self.space_dim), device=self.device)

  # # # ========================================================================================
  # # # ========================================================================================
  # # # The methods below are those LESS likely to be overwritten by users creating custom tasks
  # # # ========================================================================================
  # # # ========================================================================================
  def update_obs_buffer(self, action=None):
    self.obs_buffer["proprioception"] = self.obs_buffer["proprioception"][1:] + [self.get_proprioception()]
    self.obs_buffer["vision"] = self.obs_buffer["vision"][1:] + [self.get_vision()]
    self.obs_buffer["goal"] = self.obs_buffer["goal"][1:] + [self.goal]

    if action is not None:
      assert isinstance(self.action_space, gym.spaces.Box), f'`action_space` is not appropriately initialized.'
      self.obs_buffer["action"] = self.obs_buffer["action"][1:] + [action]

  @property
  def muscle(self):
    """Shortcut to the :class:`motornet.effector.Effector`'s `muscle` attribute."""
    return self.effector.muscle
  
  @property
  def skeleton(self):
    """Shortcut to the :class:`motornet.effector.Effector`'s `skeleton` attribute."""
    return self.effector.skeleton
  
  @property
  def space_dim(self):
    """Shortcut to the :class:`motornet.skeleton.Skeleton`'s `space_dim` attribute."""
    return self.skeleton.space_dim
  
  @property
  def states(self):
    """Shortcut to the :class:`motornet.effector.Effector`'s `states` attribute."""
    return self.effector.states

  def _maybe_detach_states(self):
    return self.states if self.differentiable else {key: self.detach(val) for key, val in self.states.items()}
  
  def joint2cartesian(self, joint_states):
    """Shortcut to :meth:`motornet.effector.Effector.joint2cartesian()` method."""
    return self.effector.joint2cartesian(joint_states)
  
  def _set_generator(self, seed: int | None):
    if seed is not None:
      self.effector._set_generator(seed)

  @property
  def np_random(self) -> np.random.Generator:
    """Returns the environment's internal :attr:`_np_random` that if not set will initialise with a random seed.

    Returns:
      Instances of `np.random.Generator`
    """
    return self.effector.np_random

  @np_random.setter
  def np_random(self, value: np.random.Generator) -> None:
      self.effector.np_random = value

  def apply_noise(self, loc: th.Tensor, scale: float | Sequence[float]) -> th.Tensor:
    """Applies element-wise Gaussian noise to the input `loc`.

    Args:
      loc: input on which the Gaussian noise is applied, which in probabilistic terms make it the mean of the
        Gaussian distribution.
      scale: `Float` or `list`, the standard deviation (spread or “width”) of the distribution. Must be 
        non-negative. If this is a `list, it must contain as many elements as the second axis of `loc`, and the 
        Gaussian distribution for each column of `loc` will have a different standard deviation. Note that the 
        elements within each column of `loc` will still be independent and identically distributed (`i.i.d.`).

    Returns:
      A noisy version of `loc` as a `tensor`.
    """
    white_noise = self.np_random.normal(size=loc.shape, scale=scale)
    return loc + th.tensor(white_noise, dtype=th.float32, device=self.device)

  def get_attributes(self):
    """Gets all non-callable attributes declared in the object instance, excluding `gym.spaces.Space` attributes,
    the effector, muscle, and skeleton attributes.

    Returns:
      - A `list` of attribute names as `string` elements.
      - A `list` of attribute values.
    """
    blacklist = ["effector", "muscle", "skeleton", "np_random", "states", "goal", "obs_buffer", "unwrapped"]
    attributes = [
      a for a in dir(self)
      if not a.startswith('_')
      and not callable(getattr(self, a))
      and not blacklist.__contains__(a)
      and not isinstance(getattr(self, a), gym.spaces.Space)  # Spaces are not JSON serializable
    ]
    values = [getattr(self, a) for a in attributes]
    return attributes, values

  def print_attributes(self):
    """Prints all non-callable attributes declared in the object instance, excluding `gym.spaces.Space` attributes,
      the effector, muscle, and skeleton attributes."""
    attributes = [a for a in dir(self) if not a.startswith('_') and not callable(getattr(self, a))]
    blacklist = []

    for a in attributes:
      if not blacklist.__contains__(a):
        print(a + ": ", getattr(self, a))

      for elem in blacklist:
        print("\n" + elem + ":\n", getattr(self, elem))

  def get_save_config(self):
    """Gets the environment object's configuration as a `dictionary`.

    Returns:
      A `dictionary` containing the  parameters of the environment's configuration. All parameters held as 
      non-callable attributes by the object instance will be included in the `dictionary`, excluding
      `gym.spaces.Space` attributes, the effector, muscle, and skeleton attributes.
    """

    cfg: dict[str, Any] = {'name': self.__name__}
    attributes, values = self.get_attributes()
    for attribute, value in zip(attributes, values):
      value = self.detach(value)  # tensors are not JSON serializable
      if isinstance(value, np.ndarray):
        value = value.tolist()
      if attribute in ("T_destination", "device"):
        value = str(value)
      cfg[attribute] = value

    cfg["effector"] = self.effector.get_save_config()
    return cfg
  
  @property
  def device(self) -> th.device:
    """Returns the device of the module's `effector`.
    """
    return self.effector.device
      

class RandomTargetReach(Environment):
  """A reach to a random target from a random starting position.
  """
  def _get_goal(self, batch_size: int) -> th.Tensor:
    return self.joint2cartesian(self.effector.draw_random_uniform_states(batch_size)).chunk(2, dim=-1)[0]