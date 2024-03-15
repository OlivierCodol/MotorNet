import torch as th
from numpy.random import Generator, SeedSequence
from typing import Any
from collections.abc import Sequence
from gymnasium.utils import seeding
from torch.nn.parameter import Parameter
from motornet.skeleton import TwoDofArm, PointMass, Skeleton
from motornet.muscle import CompliantTendonHillMuscle, ReluMuscle, Muscle


DEVICE = th.device("cpu")


class Effector(th.nn.Module):
  """Base class for `Effector` objects.

  Args:
    skeleton: A :class:`motornet.skeleton.Skeleton` object class or subclass. This defines the type of 
      skeleton that the muscles will wrap around.
    muscle: A :class:`motornet.muscle.Muscle` object class or subclass. This defines the type of 
      muscle that will be added each time the :meth:`add_muscle` method is called.
    name: `String`, the name of the object instance.
    timestep: `Float`, size of a single timestep (in sec).
    n_ministeps" `Integer`, number of integration ministeps per timestep. This assumes the action input is constant
      across ministeps.
    integration_method: `String`, "euler" to specify that numerical integration should be done using the Euler
      method, or "rk4", "rungekutta4", "runge-kutta4", or "runge-kutta-4" to specify the Runge-Kutta 4 method
      instead. This argument is case-insensitive.
    damping: `Float`, the damping coefficient applied at each joint, proportional to joint velocity. This value 
      should be positive to reduce joint torques proportionally to joint velocity.
    pos_upper_bound: `Float`, `list` or `tuple`, indicating the upper boundary of the skeleton's joint position.
      This should be a `n`-elements vector or `list`, with `n` the number of joints of the skeleton. For instance,
      for a two degrees-of-freedom arm, we would have `n=2`.
    pos_lower_bound: `Float`, `list` or `tuple`, indicating the lower boundary of the skeleton's joint position.
      This should be a `n`-elements vector or `list`, with `n` the number of joints of the skeleton. For instance,
      for a two degrees-of-freedom arm, we would have `n=2`.
    vel_upper_bound: `Float`, `list` or `tuple`, indicating the upper boundary of the skeleton's joint velocity.
      This should be a `n`-elements vector or `list`, with `n` the number of joints of the skeleton. For instance,
      for a two degrees-of-freedom arm, we would have `n=2`.
    vel_lower_bound: `Float`, `list` or `tuple`, indicating the lower boundary of the skeleton's joint velocity.
      This should be a `n`-elements vector or `list`, with `n` the number of joints of the skeleton. For instance,
      for a two degrees-of-freedom arm, we would have `n=2`.
  """
    
  def __init__(
    self,
    skeleton: Skeleton,
    muscle: Muscle,
    name: str = 'Effector',
    n_ministeps: int = 1,
    timestep: float = 0.01,
    integration_method: str = 'euler',
    damping: float = 0.,
    pos_lower_bound: float | Sequence[float] | th.Tensor | None = None,
    pos_upper_bound: float | Sequence[float] | th.Tensor | None = None,
    vel_lower_bound: float | Sequence[float] | th.Tensor | None = None,
    vel_upper_bound: float | Sequence[float] | th.Tensor | None = None,
    device: th.device | None = None
  ):
    
    super().__init__()

    self.__name__ = name
    device = self.device if device is None else device
    self.damping = Parameter(th.tensor(damping, dtype=th.float32, device=device), requires_grad=False)
    self.skeleton = skeleton.to(device)
    self.dof = self.skeleton.dof
    self.space_dim = self.skeleton.space_dim
    self.state_dim = self.skeleton.state_dim
    self.output_dim = self.skeleton.output_dim
    self.n_ministeps = n_ministeps
    self.dt = timestep
    self.minidt = self.dt / self.n_ministeps
    self.half_minidt = self.minidt / 2  # to reduce online calculations for RK4 integration
    self.integration_method = integration_method.casefold()  # make string fully in lower case
    self.seed = None

    # build skeleton
    self.skeleton.build(
      timestep=self.dt,
      pos_upper_bound=pos_upper_bound,
      pos_lower_bound=pos_lower_bound,
      vel_upper_bound=vel_upper_bound,
      vel_lower_bound=vel_lower_bound,
    )
    
    # initialize muscle system
    self.muscle = muscle.to(device)
    self.n_muscles = 0
    self.input_dim = 0
    self.muscle_name = []
    self.geometry_state_name = [
      'musculotendon length',
      'musculotendon velocity'
      ] + [
      'moment for joint ' + str(d) for d in range(self.dof)
      ]
    self.tobuild__muscle = self.muscle.to_build_dict
    self.tobuild__default = self.muscle.to_build_dict_default

    # geometry Parameters
    self.path_fixation_body = Parameter(th.empty((1, 1, 0), dtype=th.int, device=device), requires_grad=False)
    self.path_coordinates = Parameter(th.empty((1, self.space_dim, 0), dtype=th.float32, device=device), requires_grad=False)
    self.muscle_transitions = Parameter(th.empty((1, 1, 0), dtype=th.int, device=device), requires_grad=False)
    self.section_splits = Parameter(th.empty(0, dtype=th.int, device=device), requires_grad=False)
    self._muscle_config_is_empty = True

    self.default_endpoint_load = Parameter(th.zeros((1, self.space_dim), device=device), requires_grad=False)
    self.default_joint_load = Parameter(th.zeros((1, self.dof), device=device), requires_grad=False)
    
    if self.integration_method == 'euler':
      self._integrate = self._euler
    elif self.integration_method in ('rk4', 'rungekutta4', 'runge-kutta4', 'runge-kutta-4'):
      self._integrate = self._rungekutta4
    else:
      raise ValueError("Provided integration method not recognized : {}".format(self.integration_method))
    
    self.states: dict[str, th.Tensor] = {}

  def step(self, action: th.Tensor, **kwargs):
    endpoint_load = kwargs.get('endpoint_load', self.default_endpoint_load)
    joint_load = kwargs.get('joint_load', self.default_joint_load)

    action = self.muscle.clip_activation(action)
    
    for _ in range(self.n_ministeps):
      self.integrate(action, endpoint_load, joint_load)

  def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
    """Sets initial states (joint, cartesian, muscle, geometry) that are biomechanically compatible with each other.

    Args:
      seed: `Integer`, the seed that is used to initialize the environment's PRNG (`np_random`).
        If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
        a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
        However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset.
        If you pass an integer, the PRNG will be reset even if it already exists.
        Usually, you want to pass an integer *right after the environment has been initialized and then never again*.
      options: `Dictionary`, optional kwargs. This is mainly useful to pass `batch_size` and `joint_state` kwargs if 
        desired, as described below.

    Options:
      - **batch_size**: `Integer`, the desired batch size. Default: `1`.
      - **joint_state**: The joint state from which the other state values are inferred. If `None`, random initial 
        joint states are drawn, from which the other state values are inferred. Default: `None`.
    """
    # Initialize the RNG if the seed is manually passed
    if seed is not None:
      self._set_generator(seed)

    options = {} if options is None else options
    batch_size: int = options.get('batch_size', 1)
    joint_state: Sequence | th.Tensor | None = options.get('joint_state', None)

    if joint_state is not None:
      joint_state = th.as_tensor(joint_state, dtype=th.float32, device=self.device)
      if joint_state.ndim != 1 and joint_state.ndim != 2:
        raise ValueError(f'`joint_state` should have 1 or 2 dimensions but has {joint_state.ndim}.')
      batch_size = joint_state.shape[0] if joint_state.ndim == 2 and joint_state.shape[0] > 1 else batch_size

    joint0 = self._parse_initial_joint_state(joint_state=joint_state, batch_size=batch_size)
    geometry0 = self.get_geometry(joint0)
    muscle0 = self.muscle.get_initial_muscle_state(batch_size=batch_size, geometry_state=geometry0)
    states = {"joint": joint0, "muscle": muscle0, "geometry": geometry0}

    self._set_state(states)
    self.batch_size = batch_size

  def _set_generator(self, seed: int):
    self.np_random = seeding.np_random(seed)[0]

  @property
  def np_random(self) -> Generator:
    """Returns the environment's internal :attr:`_np_random` that if not set will initialise with a random seed.

    Returns:
      Instances of `numpy.random.Generator`
    """
    if not hasattr(self, '_np_random'):
      self._np_random, self.seed = seeding.np_random()
    return self._np_random

  @np_random.setter
  def np_random(self, rng: Generator):
    self._np_random = rng
    assert isinstance(rng.bit_generator.seed_seq, SeedSequence), f'Expected to find `numpy.random.SeedSequence` but found `{rng.bit_generator.seed_seq.__class__}`. Maybe a non-standard Generator is being used?'
    self.seed = rng.bit_generator.seed_seq.entropy

  @property
  def device(self) -> th.device:
    """Returns the device of the first parameter in the module or the 1st CPU device if no parameter is yet declared.
    The parameter search includes children modules.
    """
    try:
      return next(self.parameters()).device
    except:
      return DEVICE

  def add_muscle(self, path_fixation_body: Sequence[int], path_coordinates: Sequence[Sequence[float]], name: str | None = None, **kwargs):
    """Adds a muscle to the effector.

    Args:
      path_fixation_body: `List`, containing the index of the fixation body (or fixation bone) for each fixation
        point in the muscle. The index `0` always stands for the worldspace, *i.e.* a fixation point outside of
        the skeleton.
      path_coordinates:  A `List` of `lists`. There should be as many lists in the main list as there are fixation
        points for that muscle. Each nested `list` within the main `list` should contain a series of `n`
        coordinate `float` values, with `n` being the dimensionality of the worldspace. For instance, in a 2D
        environment, we would need two coordinate values. The coordinate system of each bone is centered on that
        bone's origin point. Its first dimension is always alongside the length of the bone, and the next
        dimension(s) proceed(s) from there on orthogonally to that first dimension.
      name: `String`, the name to give to the muscle being added. If ``None`` is given, the name defaults to
        `"muscle_m"`, with `m` being a counter for the number of muscles.
      **kwargs: This is used to pass the set of properties required to build the type of muscle specified at
        initialization. What it should contain varies depending on the muscle type being used. A `TypeError`
        will be raised by this method if a muscle property pertaining to the muscle type specified is missing.

    Raises:
      TypeError: If an argument is missing to build the type of muscle specified at initialization.
    """

    # extract/format inputs
    n_points = len(path_fixation_body)
    self.n_muscles += 1
    self.input_dim += self.muscle.input_dim
    path_fixation_body_new = th.tensor(path_fixation_body, dtype=th.int, device=self.device).reshape(1, 1, -1)
    path_coordinates_new = th.tensor(path_coordinates, dtype=th.float32, device=self.device).T[None, :, :]
    if path_coordinates_new.shape[1] != self.space_dim:
      raise ValueError(f'The model dimensionality is {self.space_dim}, but the `path_coordinates` argument is in {path_coordinates_new.shape[1]}d.')
    if path_coordinates_new.shape[2] != n_points:
      raise ValueError(f'Number of points in argument `path_fixation_body` ({n_points}) must match number in `path_coordinates` ({path_coordinates_new.shape[2]}).')
    one, zeros = th.ones(1, dtype=th.int, device=self.device), th.zeros(n_points - 1, dtype=th.int, device=self.device)
    muscle_transitions_new = (zeros if self.n_muscles == 1 else th.cat([one, zeros])).reshape(1, 1, -1)
    section_splits_new = th.empty(1, dtype=th.int, device=self.device).fill_(n_points)

    # update Parameters
    for param, new_data in zip(
      [self.path_fixation_body, self.path_coordinates, self.muscle_transitions, self.section_splits],
      [path_fixation_body_new, path_coordinates_new, muscle_transitions_new, section_splits_new]
    ):
      param.data = th.cat([param.data, new_data], dim=-1)

    # kwargs loop
    for key, val in kwargs.items():
      if key in self.tobuild__muscle:
        self.tobuild__muscle[key].append(val)
    for key, val in self.tobuild__muscle.items():
      # if not added in the kwargs loop
      if len(val) < self.n_muscles:
        # if the muscle object contains a default, use it, else raise error
        if key in self.tobuild__default:
          self.tobuild__muscle[key].append(self.tobuild__default[key])
        else:
          raise TypeError('Missing keyword argument ' + key + '.')
    self.muscle.build(timestep=self.minidt, **self.tobuild__muscle)

    name = name if name is not None else 'muscle_' + str(self.n_muscles)
    self.muscle_name.append(name)
    self._muscle_config_is_empty = False

  def get_muscle_cfg(self) -> dict[str, Any]:
    """Gets the wrapping configuration of muscles added through the :meth:`add_muscle` method.

    Returns:
      A `dictionary` containing a key for each muscle name, associated to a nested dictionary containing
      information fo that muscle.
    """

    cfg = {}
    for m, (n_points, fixation_body, coordinates) in enumerate(zip(
      self.section_splits,
      self.path_fixation_body.split(self.section_splits, dim=-1),
      self.path_coordinates.split(self.section_splits, dim=-1)
    )):
      d = {
        "n_fixation_points": n_points,
        "fixation body": fixation_body.squeeze().tolist(),
        "coordinates": coordinates.squeeze().T.tolist(),
      }

      if not self._muscle_config_is_empty:
        for param, value in self.tobuild__muscle.items():
          d[param] = value[m]

      cfg[self.muscle_name[m]] = d
    if not cfg:
      cfg = {"Placeholder Message": "No muscles were added using the `add_muscle` method."}
    return cfg

  def print_muscle_wrappings(self):
    """Prints the wrapping configuration of the muscles added using the :meth:`add_muscle` method in a readable
    format."""

    cfg = self.get_muscle_cfg()
    if self._muscle_config_is_empty:
      print(cfg)
      return

    for muscle, params in cfg.items():
      print("MUSCLE NAME: " + muscle)
      print("-" * (13 + len(muscle)))
      for key, param in params.items():
          print(key + ": ", param)
      print("\n")

  def get_geometry(self, joint_state: th.Tensor) -> th.Tensor:
    """Computes the geometry state from the joint state.
    Geometry state dimensionality is `[n_batch, n_timesteps, n_states, n_muscles]`. By default, there are as many
    states as there are moments (that is, one per degree of freedom in the effector) plus two for musculotendon length
    and musculotendon velocity. However, note that how many states and what they represent may vary depending on
    the :class:`Effector` subclass. This should be available via the
    :attr:`geometry_state_names` attribute.

    Args:
      joint_state: `Tensor`, the joint state from which the geometry state is computed.

    Returns:
      The geometry state corresponding to the joint state provided.
    """
    return self._get_geometry(joint_state)

  def _get_geometry(self, joint_state: th.Tensor) -> th.Tensor:
    # dxy_ddof --> (n_batches, n_dof, n_dof, n_points)
    xy, dxy_dt, dxy_ddof = self.skeleton.path2cartesian(self.path_coordinates, self.path_fixation_body, joint_state)
    diff_pos = xy[:, :, 1:] - xy[:, :, :-1]
    diff_vel = dxy_dt[:, :, 1:] - dxy_dt[:, :, :-1]
    diff_ddof = dxy_ddof[:, :, :, 1:] - dxy_ddof[:, :, :, :-1]

    # length, velocity and moment of each path segment
    # -----------------------
    # segment length is just the euclidian distance between the two points
    segment_len = (diff_pos**2).sum(dim=1, keepdim=True)**0.5
    # segment velocity is trickier: we are not after radial velocity but relative velocity.
    # https://math.stackexchange.com/questions/1481701/time-derivative-of-the-distance-between-2-points-moving-over-time
    # Formally, if segment_len=0 then segment_vel is not defined. We could substitute with 0 here because a
    # muscle segment will never flip backward, so the velocity can only be positive afterwards anyway.
    # segment_vel = tf.where(segment_len == 0, tf.zeros(1), segment_vel)
    segment_vel = (diff_pos * diff_vel / segment_len).sum(dim=1, keepdim=True)
    # for moment arm calculation, see Sherman, Seth, Delp (2013) -- DOI:10.1115/DETC2013-13633
    segment_moments = (diff_ddof * diff_pos[:, :, None]).sum(dim=1) / segment_len

    # remove differences between points that don't belong to the same muscle
    segment_len_cleaned = th.where(self.muscle_transitions, 0., segment_len)
    segment_vel_cleaned = th.where(self.muscle_transitions, 0., segment_vel)
    segment_mom_cleaned = th.where(self.muscle_transitions, 0., segment_moments)

    # sum up the contribution of all the segments belonging to a given muscle, looping over all the muscles
    # NOTE: using a loop is not ideal here, and ideally this should move toward using nested tensors, which can 
    # accommodate ragged dimensions along the number of fixation points. However, PyTorch's nested tensor
    # functionality is still in prototype stage and will likely change in the future. so we are sticking with this
    # slower solution for now. This may result in incrementally large compute time as the number of muscles grow in
    # an effector instance.
    # NOTE 2: It seems for-loops are faster than tensorflow's ragged tensor approach, even for a large number of 
    # muscles. If this is also true for PyTorch's final implementation of a nested tensor then maybe we will want to
    # stick with the current approach.
    musculotendon_len_as_list = [th.sum(y, dim=-1) for y in segment_len_cleaned.split(self.section_splits, dim=-1)]
    musculotendon_vel_as_list = [th.sum(y, dim=-1) for y in segment_vel_cleaned.split(self.section_splits, dim=-1)]
    moment_arms_as_list = [th.sum(y, dim=-1) for y in segment_mom_cleaned.split(self.section_splits, dim=-1)]

    # bring back into a single tensor
    musculotendon_len = th.stack(musculotendon_len_as_list, dim=-1)
    musculotendon_vel = th.stack(musculotendon_vel_as_list, dim=-1)
    moment_arms = th.stack(moment_arms_as_list, dim=-1)

    # pack all this into one state array and flip the dimensions back (batch_size * n_features * n_muscles)
    geometry_state = th.concat([musculotendon_len, musculotendon_vel, moment_arms], dim=1)
    return geometry_state
  
  def _set_state(self, states):
    for key, val in states.items():
      self.states[key] = val
    self.states["cartesian"] = self.joint2cartesian(joint_state=states["joint"])
    self.states["fingertip"] = self.states["cartesian"].chunk(2, dim=-1)[0]

  def integrate(self, action: th.Tensor, endpoint_load: th.Tensor, joint_load: th.Tensor):
    """Integrates the effector over one timestep. To do so, it first calls the :meth:`update_ode` method to obtain
    state derivatives from evaluation of the Ordinary Differential Equations. Then it performs the numerical
    integration over one timestep using the :meth:`integration_step` method, and updates the states to the
    resulting values.

    Args:
      action: `Tensor`, the input to the muscles (motor command). Typically, this should be the output of
        the controller or policy network's forward pass.
      endpoint_load: `Tensor`, the load(s) to apply at the skeleton's endpoint.
      joint_load: `Tensor`, the load(s) to apply at the joints.
    """
    self._integrate(action, endpoint_load, joint_load)

  def _euler(self, action, endpoint_load, joint_load):
    states0 = self.states
    state_derivative = self.ode(action, states0, endpoint_load, joint_load)
    states = self.integration_step(self.minidt, state_derivative=state_derivative, states=states0)
    self._set_state(states)

  def _rungekutta4(self, action, endpoint_load, joint_load):
    states0 = self.states
    k1 = self.ode(action, states=states0, endpoint_load=endpoint_load, joint_load=joint_load)
    states = self.integration_step(self.half_minidt, state_derivative=k1, states=states0)
    k2 = self.ode(action, states=states, endpoint_load=endpoint_load, joint_load=joint_load)
    states = self.integration_step(self.half_minidt, state_derivative=k2, states=states)
    k3 = self.ode(action, states=states, endpoint_load=endpoint_load, joint_load=joint_load)
    states = self.integration_step(self.minidt, state_derivative=k3, states=states)
    k4 = self.ode(action, states=states, endpoint_load=endpoint_load, joint_load=joint_load)
    k = {key: (k1[key] + 2 * (k2[key] + k3[key]) + k4[key]) / 6 for key in k1.keys()}
    states = self.integration_step(self.minidt, state_derivative=k, states=states0)
    self._set_state(states)

  def integration_step(self, dt: float, state_derivative: dict[str, th.Tensor], states: dict[str, th.Tensor]) -> dict[str, th.Tensor]:
    """Performs one numerical integration step for the :class:`motornet.muscle.Muscle` object class or
    subclass, and then for the :class:`motornet.skeleton.Skeleton` object class or subclass.

    Args:
      dt: `Float`, size of a single timestep (in sec).
      state_derivative: `Dictionary`, contains the derivatives of the joint, muscle, and geometry states as
        `tensor` arrays, mapped to a "joint", "muscle", and "geometry" key, respectively. This is usually
        obtained using the :meth:`update_ode` method.
      states: A `Dictionary` containing the joint, muscle, and geometry states as `Tensor`, mapped to a "joint",
        "muscle", and "geometry" key, respectively.

    Returns:
      A `dictionary` containing the updated joint, muscle, and geometry states following integration.
    """

    new_states = {
      "muscle": self.muscle.integrate(dt, state_derivative["muscle"], states["muscle"], states["geometry"]),
      "joint": self.skeleton.integrate(dt, state_derivative["joint"], states["joint"])}
    new_states["geometry"] = self.get_geometry(new_states["joint"])
    return new_states

  def ode(self, action: th.Tensor, states: dict[str, th.Tensor], endpoint_load: th.Tensor, joint_load: th.Tensor) -> dict[str, th.Tensor]:
    """Computes state derivatives by evaluating the Ordinary Differential Equations of the
    ``motornet.muscle.Muscle`` object class or subclass, and then of the
    :class:`motornet.skeleton.Skeleton` object class or subclass.

    Args:
      action: `Tensor`, the input to the muscles (motor command). Typically, this should be the output of
        the controller or policy network's forward pass.
      states: `Dictionary` contains the joint, muscle, and geometry states as `Tensor`, mapped to a "joint",
        "muscle", and "geometry" key, respectively.
      endpoint_load: `Tensor`, the load(s) to apply at the skeleton's endpoint.
      joint_load: `Tensor`, the load(s) to apply at the joints.

    Returns:
      A `dictionary` containing the derivatives of the the joint, muscle, and geometry states as `Tensor`, 
      mapped to a "joint", "muscle", and "geometry" key, respectively.
    """

    moments = states["geometry"][:, 2:, :]
    force_index = self.muscle.state_name.index('force')  # column index of muscle state containing output force
    forces = states["muscle"][:, force_index:force_index + 1, :]
    joint_vel = states["joint"].chunk(2, dim=-1)[-1]

    generalized_forces = - th.sum(forces * moments, dim=-1) + joint_load - self.damping * joint_vel
    
    state_derivative = {
      "muscle": self.muscle.ode(action, states["muscle"]),
      "joint": self.skeleton.ode(generalized_forces, states["joint"], endpoint_load=endpoint_load)
      }
    return state_derivative

  def draw_random_uniform_states(self, batch_size: int) -> th.Tensor:
    """Draws joint states according to a random uniform distribution, bounded by the position and velocity boundary
    attributes defined at initialization.

    Args:
      batch_size: `Integer`, the desired batch size.

    Returns:
      A `tensor` containing `batch_size` joint states.
    """
    sz = (batch_size, self.dof)
    rnd = th.tensor(self.np_random.uniform(size=sz), dtype=th.float32, device=self.device)
    pos = (self.skeleton.pos_upper_bound - self.skeleton.pos_lower_bound) * rnd + self.skeleton.pos_lower_bound
    vel = th.zeros_like(pos)
    return th.cat([pos, vel], dim=1)

  def _parse_initial_joint_state(self, joint_state: th.Tensor | None, batch_size: int) -> th.Tensor:
    if joint_state is None:
      return self.draw_random_uniform_states(batch_size=batch_size)
    elif joint_state.shape[-1] == self.state_dim:
      position, velocity = joint_state.chunk(2, dim=-1)
      return self.draw_fixed_states(position=position, velocity=velocity, batch_size=batch_size)
    elif joint_state.shape[-1] == self.state_dim // 2:
      return self.draw_fixed_states(position=joint_state, batch_size=batch_size)
    else:
      raise ValueError(f'Allowable state dimensions are {self.state_dim} and {self.state_dim // 2} but `joint_state` state dimension is {joint_state.shape[-1]}.')

  def draw_fixed_states(self, batch_size: int, position: th.Tensor, velocity: th.Tensor | None = None) -> th.Tensor:
    """Creates a joint state `tensor` corresponding to the specified position, tiled `batch_size` times.

    Args:
      position: The position to tile in the state `tensor`.
      velocity: The velocity to tile in the state `tensor`. If `None`, this will default to `0` (null velocity).
      batch_size: `Integer`, the desired batch size.

    Returns:
      A `tensor` containing `batch_size` joint states.
    """
    if velocity is None:
      velocity = th.zeros_like(position)

    if (position < self.skeleton.pos_lower_bound).any() or (position > self.skeleton.pos_upper_bound).any():
      raise ValueError(f'Position {position} should be between {self.skeleton.pos_lower_bound} and {self.skeleton.pos_upper_bound}')
    if (velocity < self.skeleton.vel_lower_bound).any() or (velocity > self.skeleton.vel_upper_bound).any():
      raise ValueError(f'Velocity {velocity} should be between {self.skeleton.vel_lower_bound} and {self.skeleton.vel_upper_bound}')

    states = th.cat((position, velocity), dim=-1)
    return states.expand(batch_size, self.state_dim)

  def get_save_config(self):
    """Gets the effector object's configuration as a `dictionary`.

    Returns:
      A `dictionary` containing the skeleton and muscle configurations as nested `dictionary` objects, and
      parameters of the effector's configuration. Specifically, the size of the timestep (sec), the name
      of each muscle added via the :meth:`add_muscle` method, the number of muscles, the visual and 
      proprioceptive delay, the standard deviation of the excitation noise, and the muscle wrapping configuration
      as returned by :meth:`get_muscle_cfg`.
    """
    muscle_cfg = self.muscle.get_save_config()
    skeleton_cfg = self.skeleton.get_save_config()
    cfg = {'muscle': muscle_cfg,
            'skeleton': skeleton_cfg,
            'dt': self.dt, 'n_ministeps': self.n_ministeps,
            'minidt': self.minidt, 'half_minidt': self.half_minidt,
            'muscle_names': self.muscle_name,
            'n_muscles': self.n_muscles,
            'muscle_wrapping_cfg': self.get_muscle_cfg()}
    return cfg

  def joint2cartesian(self, joint_state: th.Tensor) -> th.Tensor:
    """Computes the cartesian state given the joint state.

    Args:
      joint_state: `Tensor`, the current joint configuration.

    Returns:
      The current cartesian configuration (position, velocity) as a `tensor`.
    """
    return self.skeleton.joint2cartesian(joint_state=joint_state)

  def setattr(self, name: str, value):
    """Changes the value of an attribute held by this object.

    Args:
      name: `String`, attribute to set to a new value.
      value: Value that the attribute should take.
    """
    self.__setattr__(name, value)

  def _merge_muscle_kwargs(self, muscle_kwargs: dict):
    """
    Merges the muscle_kwargs argument with the default muscle_kwargs argument, and stores the result in the
    tobuild__muscle attribute.

    Args:
      muscle_kwargs: `Dictionary`, contains the muscle_kwargs argument passed to the
        :meth:`motornet.muscle.Muscle.build()` method.
    """
    # kwargs loop
    for key, val in muscle_kwargs.items():
      if key in self.tobuild__muscle.keys():
        self.tobuild__muscle[key] = val
      else:
        raise KeyError('Unexpected key "' + key + '" in muscle_kwargs argument.')
      
    for key, val in self.tobuild__default.items():
      # if not added in the kwargs loop
      if key not in muscle_kwargs:
        self.tobuild__muscle[key] = val


class ReluPointMass24(Effector):
  """This object implements a 2D point-mass skeleton attached to 4 ``motornet.muscle.ReluMuscle`` muscles
  in a "X" configuration. The outside attachement points are the corners of a
  `(2, 2) -> (2, -2) -> (-2, -2) -> (-2, 2)` frame, and the point-mass is constrained to a
  `(1, 1) -> (1, -1) -> (-1, -1) -> (-1, 1)` space.

  Args:
    timestep: `Float`, size of a single timestep (in sec).
    max_isometic_force: `Float`, the maximum force (N) that each muscle can produce.
    mass: `Float`, the mass (kg) of the point-mass.
    **kwargs: The `kwargs` inputs are passed as-is to the parent :class:`motornet.Effector` class.
  """

  def __init__(self, timestep: float = 0.01, max_isometric_force: float = 500, mass: float = 1, **kwargs):
    skeleton = PointMass(space_dim=2, mass=mass)
    super().__init__(skeleton=skeleton, muscle=ReluMuscle(), timestep=timestep, **kwargs)

    # path coordinates for each muscle
    ur = [[2, 2], [0, 0]]
    ul = [[-2, 2], [0, 0]]
    lr = [[2, -2], [0, 0]]
    ll = [[-2, -2], [0, 0]]

    f = max_isometric_force
    self.add_muscle(path_fixation_body=[0, 1], path_coordinates=ur, name='UpperRight', max_isometric_force=f)
    self.add_muscle(path_fixation_body=[0, 1], path_coordinates=ul, name='UpperLeft', max_isometric_force=f)
    self.add_muscle(path_fixation_body=[0, 1], path_coordinates=lr, name='LowerRight', max_isometric_force=f)
    self.add_muscle(path_fixation_body=[0, 1], path_coordinates=ll, name='LowerLeft', max_isometric_force=f)


class RigidTendonArm26(Effector):
  """This pre-built effector class is an implementation of a 6-muscles, "lumped-muscle" model from `[1]`. Because
  lumped-muscle models are functional approximations of biological reality, this class' geometry does not rely on the
  default geometry methods, but on its own, custom-made geometry. The moment arm approximation is based on a set of
  polynomial functions. The default integration method is Euler.

  If no `skeleton` input is provided, this object will use a :class:`motornet.skeleton.TwoDofArm` 
  skeleton, with the following parameters (from `[1]`):

  - `m1 = 1.82`
  - `m2 = 1.43`
  - `L1g = 0.135`
  - `L2g = 0.165`
  - `I1 = 0.051`
  - `I2 = 0.057`
  - `L1 = 0.309`
  - `L2 = 0.333`

  The default shoulder and elbow lower limits are defined as `0`, and their default upper limits as `135` and `155`
  degrees, respectively.

  The `kwargs` inputs are passed as-is to the parent :class:`Effector` class.

  References:
    [1] `Nijhof, E.-J., & Kouwenhoven, E. Simulation of Multijoint Arm Movements (2000). In J. M. Winters & P. E.
    Crago, Biomechanics and Neural Control of Posture and Movement (pp. 363–372). Springer New York.
    doi: 10.1007/978-1-4612-2104-3_29`

  Args:
    muscle: A :class:`motornet.muscle.Muscle` object class or subclass. This defines the type of muscle
      that will be added each time the :meth:`add_muscle` method is called.
    skeleton: A :class:`motornet.skeleton.Skeleton` object class or subclass. This defines the type of
      skeleton that the muscles will wrap around. See above for details on what this argument defaults to if no
      argument is passed.
    timestep: `Float`, size of a single timestep (in sec).
    muscle_kwargs: `Dictionary`, contains the muscle parameters to be passed to the 
      :meth:`motornet.muscle.Muscle.build() method.`
    **kwargs: All contents are passed to the parent :class:`Effector` class. Also allows for some backward
      compatibility.
  """

  def __init__(self, muscle: Muscle, skeleton: Skeleton | None = None, timestep=0.01, muscle_kwargs: dict = {}, **kwargs):
    bounds = th.deg2rad(th.as_tensor([
      [0, 135],  # shoulder - used to be -90 180
      [0, 155],  # elbow
    ])).T
    pos_lower_bound = kwargs.pop('pos_lower_bound', bounds[0])
    pos_upper_bound = kwargs.pop('pos_upper_bound', bounds[1])

    if skeleton is None:
      skeleton = TwoDofArm(m1=1.82, m2=1.43, L1g=.135, L2g=.165, I1=.051, I2=.057, L1=.309, L2=.333)

    super().__init__(
      skeleton=skeleton,
      muscle=muscle,
      timestep=timestep,
      pos_lower_bound=pos_lower_bound,
      pos_upper_bound=pos_upper_bound,
      **kwargs)

    # build muscle system
    self.n_muscles = 6
    self.input_dim = self.n_muscles

    self.muscle_name = ['pectoralis', 'deltoid', 'brachioradialis', 'tricepslat', 'biceps', 'tricepslong']
    
    self._merge_muscle_kwargs(muscle_kwargs)

    self.tobuild__muscle['max_isometric_force'] = [838, 1207, 1422, 1549, 414, 603]
    self.tobuild__muscle['tendon_length'] = [0.039, 0.066, 0.172, 0.187, 0.204, 0.217]
    self.tobuild__muscle['optimal_muscle_length'] = [0.134, 0.140, 0.092, 0.093, 0.137, 0.127]

    self.muscle.build(timestep=self.dt, **self.tobuild__muscle)

    a0 = [0.151, 0.2322, 0.2859, 0.2355, 0.3329, 0.2989]
    a1 = [-.03, .03, 0, 0, -.03, .03, 0, 0, -.014, .025, -.016, .03]
    a2 = [0, 0, 0, 0, 0, 0, 0, 0, -4e-3, -2.2e-3, -5.7e-3, -3.2e-3]
    a3 = [th.pi / 2, 0]
    self.a0 = Parameter(th.tensor(a0, dtype=th.float32, device=self.device).reshape(1, 1, 6), requires_grad=False)
    self.a1 = Parameter(th.tensor(a1, dtype=th.float32, device=self.device).reshape(1, 2, 6), requires_grad=False)
    self.a2 = Parameter(th.tensor(a2, dtype=th.float32, device=self.device).reshape(1, 2, 6), requires_grad=False)
    self.a3 = Parameter(th.tensor(a3, dtype=th.float32, device=self.device).reshape(1, 2, 1), requires_grad=False)

  def _get_geometry(self, joint_state):
    old_pos, old_vel = joint_state[:, :, None].chunk(2, dim=1)
    old_pos = old_pos - self.a3
    moment_arm = old_pos * self.a2 * 2 + self.a1
    musculotendon_len = th.sum((self.a1 + old_pos * self.a2) * old_pos, dim=1, keepdim=True) + self.a0
    musculotendon_vel = th.sum(old_vel * moment_arm, dim=1, keepdim=True)
    return th.cat([musculotendon_len, musculotendon_vel, moment_arm], dim=1)


class CompliantTendonArm26(RigidTendonArm26):
  """This is the compliant-tendon version of the :class:`RigidTendonArm26` class. Note that the default integration
  method is Runge-Kutta 4, instead of Euler.

  Args:
    timestep: `Float`, size of a single timestep (in sec).
    skeleton: A :class:`motornet.skeleton.Skeleton` object class or subclass. This defines the type of
      skeleton that the muscles will wrap around. If no skeleton is passed, this will default to the skeleton
      used in the parent :class:`RigidTendonArm26` class.
    
    **kwargs: All contents are passed to the parent :class:`RigidTendonArm26` class. This also
      allows for some backward compatibility.
  """

  def __init__(self, timestep=0.0002, skeleton: Skeleton | None = None, muscle_kwargs: dict = {}, **kwargs):
    integration_method = kwargs.pop('integration_method', 'rk4')
    if skeleton is None:
      skeleton = TwoDofArm(m1=1.82, m2=1.43, L1g=.135, L2g=.165, I1=.051, I2=.057, L1=.309, L2=.333)
      # skeleton = TwoDofArm(m1=2.10, m2=1.65, L1g=.146, L2g=.179, I1=.024, I2=.025, L1=.335, L2=.263)

    super().__init__(
      muscle=CompliantTendonHillMuscle(),
      skeleton=skeleton,
      timestep=timestep,
      muscle_kwargs=muscle_kwargs,
      integration_method=integration_method,
      **kwargs)

    # build muscle system
    self.tobuild__muscle['tendon_length'] = [0.070, 0.070, 0.172, 0.187, 0.204, 0.217]
    self.muscle.build(timestep=timestep, **self.tobuild__muscle)

    # Adjust some parameters to relax overly stiff tendon values.
    # This should greatly help with stability during numerical integration.
    a0 = [0.182, 0.2362, 0.2859, 0.2355, 0.3329, 0.2989]
    self.a0 = Parameter(th.tensor(a0, dtype=th.float32, device=self.device).reshape(1, 1, 6), requires_grad=False)
