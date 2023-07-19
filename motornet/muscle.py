import torch as th
import numpy as np
from torch.nn.parameter import Parameter


DEVICE = th.device("cpu")


class Muscle(th.nn.Module):
  """Base class for `Muscle` objects. If a effector contains several muscles, this object will contain all of
  those in a vectorized format, meaning for any given effector there will always be only `one` muscle object, 
  regardless of the number of muscles wrapped around the skeleton.

  The dimensionality of the muscle states produced by this object and subclasses will always be
  `n_batches * n_timesteps * n_states * n_muscles`.

  Args:
    input_dim: `Integer`, the dimensionality of the drive input to the muscle. For instance, if the muscle is only
      driven by an excitation signal, then this value should be `1`.
    output_dim: `Integer`, the dimensionality of the output. Since this object does not rely on a ``call`` method,
      but on an :meth:`integrate` method, this is the dimensionality of the :meth:`integrate` method's output.
      The output of that method should be a `muscle state`, so `output_dim` is usually the number of states
      of the :class:`Muscle` object class or subclass.
    min_activation: `Float`, the minimum activation value that this muscle can have. Any activation value lower than
      this value will be clipped.
    tau_activation: `Float`, the time constant for activation of the muscle. This is used for the Ordinary
      Differential Equation of the muscle activation method :meth:`activation_ode`.
    tau_deactivation: `Float`, the time constant for deactivation of the muscle. This is used for the Ordinary
      Differential Equation of the muscle activation method :meth:`activation_ode`.
  """

  def __init__(
    self, input_dim: int = 1,
    output_dim: int = 1,
    min_activation: float = 0.,
    tau_activation: float = 0.015,
    tau_deactivation: float = 0.05
  ):
    
    super().__init__()

    self.input_dim = input_dim
    self.state_name = []
    self.output_dim = output_dim
    self.min_activation = Parameter(th.tensor(min_activation, dtype=th.float32), requires_grad=False)
    self.tau_activation = Parameter(th.tensor(tau_activation, dtype=th.float32), requires_grad=False)
    self.tau_deactivation = Parameter(th.tensor(tau_deactivation, dtype=th.float32), requires_grad=False)
    self.to_build_dict = {'max_isometric_force': []}
    self.to_build_dict_default = {}
    self.dt = None
    self.n_muscles = None
    self.max_iso_force = None
    self.vmax = None
    self.l0_se = None
    self.l0_ce = None
    self.l0_pe = None
    self.built = False

  def clip_activation(self, a):
    return th.clamp(a, self.min_activation, 1.)
  
  @property
  def device(self):
    """Returns the device of the first parameter in the module or the 1st CPU device if no parameter is yet declared.
    The parameter search includes children modules.
    """
    try:
      return next(self.parameters()).device
    except:
      return DEVICE

  def build(self, timestep, max_isometric_force, **kwargs):
    """Build the muscle given parameters from the ``motornet.effector.Effector`` wrapper object. This should be 
    called by the :meth:`motornet.effector.Effector.add_muscle` method to build the muscle scructure according to 
    the parameters of that effector.

    Args:
      timestep: `Float`, the size of a single timestep in seconds.
      max_isometric_force: `Float` or `list` of `float`, the maximum amount of force (N) that this particular
        muscle can use. If several muscles are being built, then this should be a `list` containing as many
        elements as there are muscles.
      **kwargs: Optional keyword arguments. This allows for extra parameters to be passed in the :meth:`build`
        method for a :class:`Muscle` subclass, if needed.
    """
    max_isometric_force = np.array(max_isometric_force).reshape(1, 1, -1)
    self.n_muscles = np.array(max_isometric_force).size
    self.max_iso_force = Parameter(th.tensor(max_isometric_force, dtype=th.float32), requires_grad=False)
    self.dt = timestep
    self.vmax = Parameter(th.ones((1, 1, self.n_muscles)), requires_grad=False)
    self.l0_se = Parameter(th.ones((1, 1, self.n_muscles)), requires_grad=False)
    self.l0_ce = Parameter(th.ones((1, 1, self.n_muscles)), requires_grad=False)
    self.l0_pe = Parameter(th.ones((1, 1, self.n_muscles)), requires_grad=False)
    self.built = True

  def get_initial_muscle_state(self, batch_size, geometry_state):
    """Infers the `muscle state` matching a provided `geometry state` array.

    Args:
      batch_size: `Integer`, the size of the batch passed in `geometry state`.
      geometry_state: `Tensor`, the `geometry state` array from which the matching initial `muscle state` is
        inferred.

    Returns:
      A `tensor` containing the initial `muscle state` matching the input `geometry state` array.
    """
    return self._get_initial_muscle_state(batch_size, geometry_state)

  def _get_initial_muscle_state(self, batch_size, geometry_state):
    raise NotImplementedError

  def integrate(self, dt, state_derivative, muscle_state, geometry_state):
    """Performs one integration step for the muscle step.

    Args:
      dt: `Float`, size of the timestep in seconds for this integration step.
      state_derivative: `Tensor`, the derivatives of the `muscle state`. These are usually obtained using this
        object's :meth:`ode` method.
      muscle_state: `Tensor`, the `muscle state` used as the initial state value for the numerical integration.
      geometry_state: `Tensor`, the `geometry state` used as the initial state value for the numerical
        integration.

    Returns:
      A `tensor` containing the new `muscle state` following numerical integration.
    """
    return self._integrate(dt, state_derivative, muscle_state, geometry_state)
  
  def _integrate(self, dt, state_derivative, muscle_state, geometry_state):
    raise NotImplementedError

  def ode(self, action, muscle_state):
    """Computes the derivatives of `muscle state` using the corresponding Ordinary Differential Equations.

    Args:
      action: `Tensor`, the descending excitation drive to the muscle(s).
      muscle_state: `Tensor`, the `muscle state` used as the initial value for the evaluation of the Ordinary
        Differential Equations.

    Returns:
      A `tensor` containing the derivatives of the `muscle state`.
    """
    return self._ode(action, muscle_state)

  def _ode(self, action, muscle_state):
    activation = muscle_state[:, :1, :]
    return self.activation_ode(action, activation)

  def activation_ode(self, action, activation):
    """Computes the new activation value of the (set of) muscle(s) according to the Ordinary Differential Equation
    shown in equations 1-2 in `[1]`. Note that this is incidentally the same activation function as used for the 
    `muscle` actuators in MuJoCo `[2]`.

    References:
      [1] `Thelen DG. Adjustment of muscle mechanics model parameters to simulate dynamic contractions in older
      adults. J Biomech Eng. 2003 Feb;125(1):70-7. doi: 10.1115/1.1531112. PMID: 12661198.`
      [2] https://mujoco.readthedocs.io/en/stable/modeling.html#muscle-actuators

    Args:
      excitation: `Float` or `list` of `float`, the descending excitation drive to the muscle(s). If several
        muscles are declared in the parent effector object, then this should be a `list` containing as many
        elements as there are muscles in that parent effector object.
      muscle_state: `Tensor`, the `muscle state` that provides the initial activation value for the Ordinary
        Differential Equation.

    Returns:
      A `tensor` containing the updated activation values.
    """
    action = self.clip_activation(th.reshape(action, (-1, 1, self.n_muscles)))
    activation = self.clip_activation(activation)
    tmp = 0.5 + 1.5 * activation
    tau = th.where(action > activation, self.tau_activation * tmp, self.tau_deactivation / tmp)
    return (action - activation) / tau

  def setattr(self, name: str, value):
    """Changes the value of an attribute held by this object.

    Args:
      name: `String`, attribute to set to a new value.
      value: Value that the attribute should take.
    """
    self.__setattr__(name, value)

  def get_save_config(self):
    """Gets the object instance's configuration. This is the set of configuration entries that will be useful
    for any muscle objects or subclasses.

    Returns:
       - A `dictionary` containing the muscle object's name and state names.
    """
    cfg = {'name': str(self.__name__), 'state names': self.state_name}
    return cfg


class ReluMuscle(Muscle):
  """A "rectified linear" muscle whose force output :math:`F` is a linear function of its activation value, which
  itself is bounded between `0` and `1`. Specifically:

  .. math::
    F = m * activation

  with :math:`m` the maximum isometric force of the muscle. Note that the maximum isometric force is not declared at
  initialization but via the :meth:`Muscle.build` call, which is inherited from the parent :class:`Muscle` class.
  The :math:`activation` value is the result of an Ordinary Differential Equation computed by the
  :meth:`Muscle.activation_ode` method. It is not directly the `action` input drive.

  Args:
    **kwargs: All contents are passed to the parent :class:`Muscle` class.
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.__name__ = 'ReluMuscle'
    self.state_name = [
      'activation',
      'muscle length',
      'muscle velocity',
      'force'
      ]
    self.state_dim = len(self.state_name)

  def _integrate(self, dt, state_derivative, muscle_state, geometry_state):
    activation = muscle_state[:, :1, :] + state_derivative * dt
    activation = self.clip_activation(activation)
    forces = activation * self.max_iso_force
    len_vel = geometry_state[:, :2, :]
    return th.cat([activation, len_vel, forces], dim=1)

  def _get_initial_muscle_state(self, batch_size, geometry_state):
    activation0 = th.ones((batch_size, 1, self.n_muscles), device=self.device) * self.min_activation
    force0 = th.zeros((batch_size, 1, self.n_muscles), device=self.device)
    len_vel = geometry_state[:, 0:2, :]
    return th.cat([activation0, len_vel, force0], dim=1)



class MujocoHillMuscle(Muscle):
  """This pre-built muscle class is an implementation of a Hill-type muscle model as detailed in the MuJoCo 
  documentation`[1]`. It is a rigid tendon Hill-type model.

  References:
    [1] https://mujoco.readthedocs.io/en/stable/modeling.html#muscle-actuators

  Args:
    min_activation: `Float`, the minimum activation value that this muscle can have. Any activation value lower than
      this value will be clipped.
    passive_forces: `Float`, a scalar coefficient to tune the contribution of passive forces to the total force
      output.
    **kwargs: All contents are passed to the parent :class:`Muscle` class.
  """

  def __init__(
    self,
    min_activation: float = 0.,
    passive_forces: float = 1.,
    tau_activation: float = 0.01,
    tau_deactivation: float = 0.04,
    **kwargs,
  ):
    
    super().__init__(
      min_activation=min_activation,
      tau_activation=tau_activation,
      tau_deactivation=tau_deactivation,
      **kwargs,
      )
    self.__name__ = 'MujocoHillMuscle'

    self.state_name = [
      'activation',
      'muscle length',
      'muscle velocity',
      'force-length PE',
      'force-length CE',
      'force-velocity CE',
      'force'
      ]
    self.state_dim = len(self.state_name)

    self.to_build_dict = {
      'max_isometric_force': [],
      'optimal_muscle_length': [],
      "tendon_length": [],
      'normalized_slack_muscle_length': [],
      "lmin": [],
      "lmax": [],
      "vmax": [],
      "fvmax": [],
      }
    self.to_build_dict_default = {
      'normalized_slack_muscle_length': 1.3,
      "lmin": 0.5,
      "lmax": 1.6,
      "vmax": 1.5,
      "fvmax": 1.2,
      }
    self.built = False
    self.passive_forces = passive_forces

  def build(
    self,
    timestep,
    max_isometric_force,
    tendon_length,
    optimal_muscle_length,
    normalized_slack_muscle_length,
    lmin,
    lmax,
    vmax,
    fvmax,
  ):
    """Build the muscle using arguments from the :class:`motornet.effector.Effector` wrapper object. This
    should be called by the :meth:`motornet.effector.Effector.add_muscle` method to build the muscle 
    structure according to the parameters of that effector.

    Args:
      timestep: `Float`, the size of a single timestep in seconds.
      max_isometric_force: `Float` or `list` of `float`, the maximum amount of force (N) that this particular
        muscle can use. If several muscles are being built, then this should be a list containing as many
        elements as there are muscles.
      tendon_length: `Float` or `list` of `float`, the tendon length (m) of the muscle(s). If several
        muscles are declared in the parent effector object, then this should be a list containing as many 
        elements as there are muscles in that parent effector object.
      optimal_muscle_length: `Float` or `list` of `float`, the optimal length (m) of the muscle(s). This defines
        the length at which the muscle will output the maximum amount of force given the same excitation. If
        several muscles are declared in the parent effector object, then this should be a list containing as
        many elements as there are muscles in that parent effector object.
      normalized_slack_muscle_length: `Float` or `list` of `float`, the muscle length (m) past which the
        muscle(s) will start to developp passive forces. If several muscles are declared in the parent effector
        object, then this should be a list containing as many elements as there are muscles in that parent
        effector object.
      lmin: `Float`, lower bound on the operating range of the muscle length (normalized by its optimal length).
      lmax: `Float`, upper bound on the operating range of the muscle length (normalized by its optimal length).
      vmax: `Float`, shortening velocity at which muscle force drops to zero (normalized by its optimal length per
        sec).
      fvmax: `Float`, active force generated at saturating lengthening velocity, relative to its maximum isometric
        force.
    """

    self.l0_pe = normalized_slack_muscle_length
    self.l0_ce = Parameter(th.tensor(optimal_muscle_length, dtype=th.float32), requires_grad=False)
    self.l0_se = Parameter(th.tensor(tendon_length, dtype=th.float32), requires_grad=False)
    self.lmin = lmin
    self.lmax = lmax
    self.vmax = vmax
    self.fvmax = fvmax

    self.n_muscles = np.array(self.l0_se).size
    self.dt = timestep

    max_isometric_force = np.array(max_isometric_force, dtype=np.float32).reshape((1, 1, self.n_muscles))
    self.max_iso_force = Parameter(th.tensor(max_isometric_force), requires_grad=False)

    # derived quantities
    # a = 0.5*(lmin+1)
    self.b = 0.5 * (1 + self.lmax)
    self.c = self.fvmax - 1
    self.p1 = self.b - 1
    self.p2 = 0.25 * self.l0_pe
    self.zero_as_tensor = Parameter(th.tensor(0., dtype=th.float32), requires_grad=False)
    self.mid = 0.5 * (self.lmin + 0.95)

    self.built = True

  def _get_initial_muscle_state(self, batch_size, geometry_state):
    shape = geometry_state[:, :1, :].shape
    muscle_state = th.ones(shape, dtype=th.float32, device=self.device) * self.min_activation
    state_derivatives = th.zeros(shape, dtype=th.float32, device=self.device)
    return self.integrate(self.dt, state_derivatives, muscle_state, geometry_state)

  def _integrate(self, dt, state_derivative, muscle_state, geometry_state):
    activation = muscle_state[:, :1, :] + state_derivative * dt
    activation = self.clip_activation(activation)

    # musculotendon geometry
    musculotendon_len = geometry_state[:, :1, :]
    muscle_len = th.clip(musculotendon_len - self.l0_se, min=0.001) / self.l0_ce
    muscle_vel = geometry_state[:, 1:2, :] / self.vmax

    # muscle forces
    x = th.where(
      muscle_len <= 1,
      input=self.zero_as_tensor,
      other=th.where(
        th.less_equal(muscle_len, self.b),
        input=(muscle_len - 1) / self.p1,
        other=(muscle_len - self.b) / self.p1,
        )
      )
    flpe = th.where(
      th.less_equal(muscle_len, 1),
      input=self.zero_as_tensor,
      other=th.where(
        muscle_len <= self.b,
        input=self.p2 * x**3,
        other=self.p2 * (1 + 3*x),
        )
      )

    # length-active
    flce = self._bump(muscle_len, mid=1, lmax=self.lmax) + 0.15 * self._bump(muscle_len, mid=self.mid, lmax=0.95)

    # velocity-active
    fvce = th.where(
      th.less_equal(muscle_vel, -1),
      input=self.zero_as_tensor,
      other=th.where(
        th.less_equal(muscle_vel, 0.),
        input=(muscle_vel+1) * (muscle_vel+1),
        other=th.where(
          th.less_equal(muscle_vel, self.c),
          input=self.fvmax - (self.c-muscle_vel)*(self.c-muscle_vel)/self.c,
          other=self.fvmax,
          )
        )
      )
    force = (activation * flce * fvce + self.passive_forces * flpe) * self.max_iso_force
    return th.cat([activation, muscle_len * self.l0_ce, muscle_vel * self.vmax, flpe, flce, fvce, force], dim=1)

    
  def _bump(self, L, mid, lmax):
    """Skewed bump function: quadratic spline."""

    left = 0.5*(self.lmin+mid)
    right = 0.5*(mid+lmax)

    out_of_range = th.logical_or(th.less_equal(L, self.lmin), th.greater_equal(L, lmax))
    less_than_left = th.less(L, left)
    less_than_mid = th.less(L, mid)
    less_than_right = th.less(L, right)

    x = th.where(out_of_range, input=self.zero_as_tensor,
      other=th.where(less_than_left, input=(L-self.lmin) / (left-self.lmin),
        other=th.where(less_than_mid, input=(mid-L) / (mid-left),
          other=th.where(less_than_right, input=(L-mid) / (right-mid),
            other=(lmax-L) / (lmax-right),
            )
          )
        )
      )
    pfivexx = 0.5 * x * x
    y = th.where(out_of_range, input=self.zero_as_tensor,
      other=th.where(less_than_left, input=pfivexx,
        other=th.where(less_than_mid, input=1 - pfivexx,
          other=th.where(less_than_right, input=1 - pfivexx,
            other=pfivexx,
            )
          )
        )
      )
    return y


class RigidTendonHillMuscle(Muscle):
  """This pre-built muscle class is an implementation of a Hill-type muscle model as detailed in `[1]`, adjusted to
  behave as a rigid tendon version of the original model.

  References:
    [1] `Kistemaker DA, Wong JD, Gribble PL. The central nervous system does not minimize energy cost in arm
    movements. J Neurophysiol. 2010 Dec;104(6):2985-94. doi: 10.1152/jn.00483.2010. Epub 2010 Sep 8. PMID:
    20884757.`

  Args:
    min_activation: `Float`, the minimum activation value that this muscle can have. Any activation value lower than
      this value will be clipped.
    **kwargs: All contents are passed to the parent :class:`Muscle` class.
  """

  def __init__(self, min_activation=0.001, **kwargs):
    super().__init__(min_activation=min_activation, **kwargs)
    self.__name__ = 'RigidTendonHillMuscle'

    self.state_name = [
      'activation',
      'muscle length',
      'muscle velocity',
      'force-length PE',
      'force-length CE',
      'force-velocity CE',
      'force'
    ]
    self.state_dim = len(self.state_name)

    # parameters for the passive element (PE) and contractile element (CE)
    self.pe_k = 5.
    self.pe_1 = self.pe_k / 0.66
    self.pe_den = np.exp(self.pe_k) - 1
    self.ce_gamma = 0.45
    self.ce_Af = 0.25
    self.ce_fmlen = 1.4

    # pre-define attributes:
    self.musculotendon_slack_len = None
    self.k_pe = None
    self.s_as = 0.001
    self.f_iso_n_den = .66 ** 2
    self.k_se = 1 / (0.04 ** 2)
    self.q_crit = 0.3
    self.b_rel_st_den = 5e-3 - self.q_crit
    self.min_flce = 0.01

    self.to_build_dict = {
      'max_isometric_force': [],
      'tendon_length': [],
      'optimal_muscle_length': [],
      'normalized_slack_muscle_length': [],
      }
    self.to_build_dict_default = {'normalized_slack_muscle_length': 1.4}

    self.built = False

  def build(
    self,
    timestep,
    max_isometric_force,
    tendon_length,
    optimal_muscle_length,
    normalized_slack_muscle_length,
  ):
    """Build the muscle using arguments from the :class:`motornet.effector.Effector` wrapper object. This
    should be called by the :meth:`motornet.effector.Effector.add_muscle` method to build the muscle
    structure according to the parameters of that effector.

    Args:
      timestep: `Float`, the size of a single timestep in seconds.
      max_isometric_force: `Float` or `list` of `float`, the maximum amount of force (N) that this particular
        muscle can use. If several muscles are being built, then this should be a list containing as many
        elements as there are muscles.
      tendon_length: `Float` or `list` of `float`, the tendon length (m) of the muscle(s). If several
        muscles are declared in the parent effector object, then this should be a list containing as many
        elements as there are muscles in that parent effector object.
      optimal_muscle_length: `Float` or `list` of `float`, the optimal length (m) of the muscle(s). This defines
        the length at which the muscle will output the maximum amount of force given the same excitation. If
        several muscles are declared in the parent effector object, then this should be a list containing as 
        many elements as there are muscles in that parent effector object.
      normalized_slack_muscle_length: `Float` or `list` of `float`, the muscle length (m) past which the
        muscle(s) will start to developp passive forces. If several muscles are declared in the parent effector
        object, then this should be a list containing as many elements as there are muscles in that parent 
        effector object.
    """
    self.n_muscles = np.array(tendon_length).size
    shape = (1, 1, self.n_muscles)

    self.dt = timestep
    self.max_iso_force = Parameter(th.tensor(max_isometric_force, dtype=th.float32).reshape(shape), requires_grad=False)
    self.l0_ce = Parameter(th.tensor(optimal_muscle_length, dtype=th.float32).reshape(shape), requires_grad=False)
    self.l0_se = Parameter(th.tensor(tendon_length, dtype=th.float32).reshape(shape), requires_grad=False)
    self.l0_pe = Parameter(th.tensor(normalized_slack_muscle_length, dtype=th.float32)*self.l0_ce, requires_grad=False)
    self.k_pe = 1 / ((1.66 - self.l0_pe / self.l0_ce) ** 2)
    self.musculotendon_slack_len = self.l0_pe + self.l0_se
    self.vmax = 10 * self.l0_ce
    self.built = True

  def _get_initial_muscle_state(self, batch_size, geometry_state):
    shape = geometry_state[:, :1, :].shape
    muscle_state = th.ones(shape, device=self.device) * self.min_activation
    state_derivatives = th.zeros(shape, device=self.device)
    return self.integrate(self.dt, state_derivatives, muscle_state, geometry_state)

  def _integrate(self, dt, state_derivative, muscle_state, geometry_state):
    activation = self.clip_activation(muscle_state[:, :1, :] + state_derivative * dt)

    # musculotendon geometry
    musculotendon_len = geometry_state[:, :1, :]
    muscle_vel = geometry_state[:, 1:2, :]
    muscle_len = th.clip(musculotendon_len - self.l0_se, min=0.)
    muscle_strain = th.clip((muscle_len - self.l0_pe) / self.l0_ce, min=0.)
    muscle_len_n = muscle_len / self.l0_ce
    muscle_vel_n = muscle_vel / self.vmax

    # muscle forces
    # flpe = tf.minimum(self.k_pe * (muscle_strain ** 2), 3.)
    flpe = self.k_pe * (muscle_strain ** 2)
    flce = th.clip(1 + (- muscle_len_n ** 2 + 2 * muscle_len_n - 1) / self.f_iso_n_den, min=self.min_flce)

    a_rel_st = th.where(
      condition=muscle_len_n > 1.,
      input=.41 * flce,
      other=.41
      )
    b_rel_st = th.where(
      condition=activation < self.q_crit,
      input=5.2 * (1 - .9 * ((activation - self.q_crit) / (5e-3 - self.q_crit))) ** 2,
      other=5.2
      )
    dfdvcon0 = activation * (flce + a_rel_st) / b_rel_st  # inv of slope at isometric point wrt concentric curve

    f_x_a = flce * activation  # to speed up computation

    tmp_p_nom = f_x_a * .5
    tmp_p_den = self.s_as - dfdvcon0 * 2.

    p1 = - tmp_p_nom / tmp_p_den
    p2 = (tmp_p_nom ** 2) / tmp_p_den
    p3 = - 1.5 * f_x_a

    nom = th.where(
      condition=muscle_vel_n < 0,
      input=muscle_vel_n * activation * a_rel_st + f_x_a * b_rel_st,
      other=-p1 * p3 + p1 * self.s_as * muscle_vel_n + p2 - p3 * muscle_vel_n + self.s_as * muscle_vel_n ** 2
      )
    den = th.where(
      condition=muscle_vel_n < 0,
      input=b_rel_st - muscle_vel_n,
      other=p1 + muscle_vel_n
      )

    active_force = th.clip(nom / den, min=0.)
    force = (active_force + flpe) * self.max_iso_force
    return th.cat([activation, muscle_len, muscle_vel, flpe, flce, active_force, force], dim=1)


class RigidTendonHillMuscleThelen(Muscle):
  """This pre-built muscle class is an implementation of a Hill-type muscle model as detailed in `[1]`, adjusted to
  behave as a rigid tendon version of the original model.

  References:
    [1] `Thelen DG. Adjustment of muscle mechanics model parameters to simulate dynamic contractions in older
    adults. J Biomech Eng. 2003 Feb;125(1):70-7. doi: 10.1115/1.1531112. PMID: 12661198.`

  Args:
    min_activation: `Float`, the minimum activation value that this muscle can have. Any activation value lower than
      this value will be clipped.
    **kwargs: All contents are passed to the parent :class:`Muscle` class.
  """

  def __init__(self, min_activation=0.001, **kwargs):
    super().__init__(min_activation=min_activation, **kwargs)
    self.__name__ = 'RigidTendonHillMuscleThelen'

    self.state_name = [
      'activation',
      'muscle length',
      'muscle velocity',
      'force-length PE',
      'force-length CE',
      'force-velocity CE',
      'force'
      ]
    self.state_dim = len(self.state_name)

    # parameters for the passive element (PE) and contractile element (CE)
    self.pe_k = Parameter(th.tensor(5., dtype=th.float32), requires_grad=False)
    self.pe_1 = Parameter(self.pe_k / 0.6, requires_grad=False)  # divided by epsilon_0^M in Thelen (2003) eq. 3
    self.pe_den = Parameter(th.exp(self.pe_k) - 1, requires_grad=False)
    self.ce_gamma = Parameter(th.tensor(0.45, dtype=th.float32), requires_grad=False)
    self.ce_Af = Parameter(th.tensor(0.25, dtype=th.float32), requires_grad=False)
    self.ce_fmlen = Parameter(th.tensor(1.4, dtype=th.float32), requires_grad=False)

    # pre-define attributes:
    self.musculotendon_slack_len = None
    self.ce_0 = None
    self.ce_1 = None
    self.ce_2 = None
    self.ce_3 = None
    self.ce_4 = None
    self.ce_5 = None

    self.to_build_dict = {
      'max_isometric_force': [],
      'tendon_length': [],
      'optimal_muscle_length': [],
      'normalized_slack_muscle_length': []
      }
    self.to_build_dict_default = {'normalized_slack_muscle_length': 1.}
    self.built = False

  def build(
    self,
    timestep,
    max_isometric_force,
    tendon_length,
    optimal_muscle_length,
    normalized_slack_muscle_length
  ):
    """Build the muscle using arguments from the :class:`motornet.effector.Effector` wrapper object. This
    should be called by the :meth:`motornet.effector.Effector.add_muscle` method to build the muscle 
    structure according to the parameters of that effector.

    Args:
      timestep: `Float`, the size of a single timestep in seconds.
      max_isometric_force: `Float` or `list` of `float`, the maximum amount of force (N) that this particular
        muscle can use. If several muscles are being built, then this should be a list containing as many
        elements as there are muscles.
      tendon_length: `Float` or `list` of `float`, the tendon length (m) of the muscle(s). If several
        muscles are declared in the parent effector object, then this should be a list containing as many
        elements as there are muscles in that parent effector object.
      optimal_muscle_length: `Float` or `list` of `float`, the optimal length (m) of the muscle(s). This defines
        the length at which the muscle will output the maximum amount of force given the same excitation. If
        several muscles are declared in the parent effector object, then this should be a list containing as 
        many elements as there are muscles in that parent effector object.
      normalized_slack_muscle_length: `Float` or `list` of `float`, the muscle length (m) past which the
        muscle(s) will start to developp passive forces. If several muscles are declared in the parent effector
        object, then this should be a list containing as many elements as there are muscles in that parent 
        effector object.
    """
    self.n_muscles = np.array(tendon_length).size
    self.dt = timestep

    max_isometric_force = th.tensor(max_isometric_force, dtype=th.float32).reshape(1, 1, self.n_muscles)
    optimal_muscle_length = th.tensor(optimal_muscle_length, dtype=th.float32).reshape(1, 1, self.n_muscles)
    tendon_length = th.tensor(tendon_length, dtype=th.float32).reshape(1, 1, self.n_muscles)
    normalized_slack_muscle_length = th.tensor(normalized_slack_muscle_length, dtype=th.float32)

    self.max_iso_force = Parameter(max_isometric_force, requires_grad=False)
    self.l0_ce = Parameter(optimal_muscle_length, requires_grad=False)
    self.l0_se = Parameter(tendon_length, requires_grad=False)
    self.l0_pe = Parameter(self.l0_ce * normalized_slack_muscle_length, requires_grad=False)
    self.musculotendon_slack_len = Parameter(self.l0_pe + self.l0_se, requires_grad=False)
    self.vmax = Parameter(10 * self.l0_ce, requires_grad=False)

    # pre-computed for speed
    self.ce_1 = Parameter(3 * self.vmax, requires_grad=False)
    self.ce_0 = Parameter(self.ce_Af * self.vmax, requires_grad=False)
    self.ce_2 = Parameter(3 * self.ce_Af * self.vmax * self.ce_fmlen - 3. * self.ce_Af * self.vmax, requires_grad=False)
    self.ce_3 = Parameter(8 * self.ce_Af * self.ce_fmlen + 8. * self.ce_fmlen, requires_grad=False)
    self.ce_4 = Parameter(self.ce_Af * self.ce_fmlen * self.vmax - self.ce_1, requires_grad=False)
    self.ce_5 = Parameter(8 * (self.ce_Af + 1.), requires_grad=False)

    self.built = True

  def _get_initial_muscle_state(self, batch_size, geometry_state):
    shape = geometry_state[:, :1, :].shape
    muscle_state = th.ones(shape, device=self.device) * self.min_activation
    state_derivatives = th.zeros(shape, device=self.device)
    return self.integrate(self.dt, state_derivatives, muscle_state, geometry_state)

  def _integrate(self, dt, state_derivative, muscle_state, geometry_state):
    activation = muscle_state[:, :1, :] + state_derivative * dt
    activation = self.clip_activation(activation)

    # musculotendon geometry
    musculotendon_len = geometry_state[:, :1, :]
    muscle_len = th.clip(musculotendon_len - self.l0_se, min=0.001)
    muscle_vel = geometry_state[:, 1:2, :]

    # muscle forces
    a3 = activation * 3.
    condition = muscle_vel <= 0
    nom = th.where(
      condition,
      input=self.ce_Af * (activation * self.ce_0 + 4. * muscle_vel + self.vmax),
      other=self.ce_2 * activation + self.ce_3 * muscle_vel + self.ce_4
      )
    den = th.where(
      condition,
      input=a3 * self.ce_1 + self.ce_1 - 4. * muscle_vel,
      other=self.ce_4 * a3 + self.ce_5 * muscle_vel + self.ce_4
      )
    fvce = th.clip(nom / den, min=0.)
    flpe = th.clip((th.exp(self.pe_1 * (muscle_len - self.l0_pe) / self.l0_ce) - 1) / self.pe_den, min=0.)
    flce = th.exp((- ((muscle_len / self.l0_ce) - 1) ** 2) / self.ce_gamma)
    force = (activation * flce * fvce + flpe) * self.max_iso_force
    return th.cat([activation, muscle_len, muscle_vel, flpe, flce, fvce, force], dim=1)


class CompliantTendonHillMuscle(RigidTendonHillMuscle):
  """This pre-built muscle class is an implementation of a Hill-type muscle model as detailed in `[1]`. Unlike its
  parent class, this class implements a full compliant tendon version of the model, as formulated in the
  reference article.

  References:
    [1] `Kistemaker DA, Wong JD, Gribble PL. The central nervous system does not minimize energy cost in arm
    movements. J Neurophysiol. 2010 Dec;104(6):2985-94. doi: 10.1152/jn.00483.2010. Epub 2010 Sep 8. PMID:
    20884757.`

  Args:
    min_activation: `Float`, the minimum activation value that this muscle can have. Any activation value lower than
      this value will be clipped.
    **kwargs: All contents are passed to the parent :class:`Muscle` class.
  """

  def __init__(self, min_activation=0.01, **kwargs):
    super().__init__(min_activation=min_activation, **kwargs)
    self.__name__ = 'CompliantTendonHillMuscle'

    self.state_name = [
      'activation',
      'muscle length',
      'muscle velocity',
      'force-length PE',
      'force-length SE',
      'active force',
      'force']
    self.state_dim = len(self.state_name)
    self.built = False

  def _integrate(self, dt, state_derivative, muscle_state, geometry_state):
    # Compute musculotendon geometry
    muscle_len = muscle_state[:, 1:2, :]
    muscle_len_n = muscle_len / self.l0_ce
    musculotendon_len = geometry_state[:, :1, :]
    tendon_len = musculotendon_len - muscle_len
    tendon_strain = th.clip((tendon_len - self.l0_se) / self.l0_se, min=0.)
    muscle_strain = th.clip((muscle_len - self.l0_pe) / self.l0_ce, min=0.)

    # Compute forces
    flse = th.clip(self.k_se * (tendon_strain ** 2), max=1.)
    # flpe = tf.minimum(self.k_pe * (muscle_strain ** 2), 1.)
    flpe = self.k_pe * (muscle_strain ** 2)
    active_force = th.clip(flse - flpe, min=0.)

    # Integrate
    d_activation = state_derivative[:, 0:1, :]
    muscle_vel_n = state_derivative[:, 1:2, :]
    activation = muscle_state[:, 0:1, :] + d_activation * dt
    activation = self.clip_activation(activation)
    new_muscle_len = (muscle_len_n + dt * muscle_vel_n) * self.l0_ce

    muscle_vel = muscle_vel_n * self.vmax
    force = flse * self.max_iso_force
    return th.concat([activation, new_muscle_len, muscle_vel, flpe, flse, active_force, force], dim=1)

  def _ode(self, excitation, muscle_state):
    activation = muscle_state[:, 0:1, :]
    d_activation = self.activation_ode(excitation, activation)
    muscle_len_n = muscle_state[:, 1:2, :] / self.l0_ce
    active_force = muscle_state[:, 5:6, :]
    new_muscle_vel_n = self._normalized_muscle_vel(muscle_len_n, activation, active_force)
    return th.concat([d_activation, new_muscle_vel_n], dim=1)

  def _get_initial_muscle_state(self, batch_size, geometry_state):
    musculotendon_len = geometry_state[:, 0:1, :]
    activation = th.ones_like(musculotendon_len, device=self.device) * self.min_activation
    d_activation = th.zeros_like(musculotendon_len, device=self.device)

    # if musculotendon length is negative, raise an error.
    # if musculotendon length is less than tendon slack length, assign all (most of) the length to the tendon.
    # if musculotendon length is more than tendon slack length and less than musculotendon slack length, assign to
    #  the tendon up to the tendon slack length, and the rest to the muscle length.
    # if musculotendon length is more than tendon slack length and muscle slack length combined, find the muscle
    #  length that satisfies equilibrium between tendon passive forces and muscle passive forces.
    muscle_len = th.where(
      condition=th.less(musculotendon_len, 0.),
      input=th.tensor(-1., dtype=th.float32, device=self.device),
      other=th.where(
        condition=th.less(musculotendon_len, self.l0_se),
        input=0.001 * self.l0_ce,
        other=th.where(
          condition=th.less(musculotendon_len, self.l0_se + self.l0_pe),
          input=musculotendon_len - self.l0_se,
          other=(self.k_pe * self.l0_pe * self.l0_se ** 2 -
            self.k_se * (self.l0_ce ** 2) * musculotendon_len +
            self.k_se * self.l0_ce ** 2 * self.l0_se -
            self.l0_ce * self.l0_se * th.sqrt(self.k_pe * self.k_se)
            * (-musculotendon_len + self.l0_pe + self.l0_se)) /
           (self.k_pe * self.l0_se ** 2 - self.k_se * self.l0_ce ** 2))))

    # tf.debugging.assert_non_negative(muscle_len, message='initial muscle length was < 0.')
    tendon_len = musculotendon_len - muscle_len
    tendon_strain = th.clip((tendon_len - self.l0_se) / self.l0_se, min=0.)
    muscle_strain = th.clip((muscle_len - self.l0_pe) / self.l0_ce, min=0.)

    # Compute forces
    flse = th.clip(self.k_se * (tendon_strain ** 2), max=1.)
    flpe = th.clip(self.k_pe * (muscle_strain ** 2), max=1.)
    active_force = th.clip(flse - flpe, min=0.)

    muscle_vel_n = self._normalized_muscle_vel(muscle_len / self.l0_ce, activation, active_force)
    muscle_state = th.concat([activation, muscle_len], dim=1)
    state_derivative = th.concat([d_activation, muscle_vel_n], dim=1)

    return self.integrate(self.dt, state_derivative, muscle_state, geometry_state)

  def _normalized_muscle_vel(self, muscle_len_n, activation, active_force):
    flce = th.clip(1. + (- muscle_len_n ** 2 + 2 * muscle_len_n - 1) / self.f_iso_n_den, min=self.min_flce)
    a_rel_st = th.where(th.less(muscle_len_n, 1.), input=.41 * flce, other=.41)
    b_rel_st = th.where(
      condition=th.less(activation, self.q_crit),
      input=5.2 * (1 - .9 * ((activation - self.q_crit) / (5e-3 - self.q_crit))) ** 2,
      other=5.2)
    # inv of slope at isometric point wrt concentric curve
    f_x_a = flce * activation  # to speed up computation
    dfdvcon0 = (f_x_a + activation * a_rel_st) / b_rel_st

    p1 = - f_x_a * .5 / (self.s_as - dfdvcon0 * 2.)
    p3 = - 1.5 * f_x_a
    p2_containing_term = (4 * ((f_x_a * 0.5) ** 2) * (- self.s_as)) / (self.s_as - dfdvcon0 * 2)

    # defensive code to avoid propagation of negative square root in the non-selected tf.where outcome
    # the assertion is to ensure that any selected item is indeed not a negative root.
    sqrt_term = active_force ** 2 + 2 * active_force * p1 * self.s_as + \
      2 * active_force * p3 + p1 ** 2 * self.s_as ** 2 + 2 * p1 * p3 * self.s_as +\
      p2_containing_term + p3 ** 2
    #cond = th.logical_or(th.less(sqrt_term, 0.), th.greater_equal(active_force, f_x_a))
    #th._assert(cond, message='root that should be used is negative.')
    sqrt_term = th.clip(sqrt_term, min=0.)

    new_muscle_vel_nom = th.where(
      condition=th.less(active_force, f_x_a),
      input=b_rel_st * (active_force - f_x_a),
      other=- active_force + p1 * self.s_as - p3 - th.sqrt(sqrt_term))
    new_muscle_vel_den = th.where(
      condition=th.less(active_force, f_x_a),
      input=active_force + activation * a_rel_st,
      other=- 2 * self.s_as)

    return new_muscle_vel_nom / new_muscle_vel_den
