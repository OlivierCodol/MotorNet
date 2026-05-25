# Changelog

## Development releases


<!--
### <font size="4">Version 0.1.6</font>
*YYYY, Month DDth*

- 
-->


### <font size="4">Version 0.3.0</font>
*2026, May 25th*

This update adds two new effector classes, a unit test suite, bug fixes, and code quality improvements including type hints, descriptive error messages,
and decoupling of the `Environment` initialization from the `reset()` call.

- Added a `Reacher` effector class. This is a four-muscle two-degree-of-freedom arm (`TwoDofArm`) with constant
moment arms, where shoulder flexor/extensor and elbow flexor/extensor are each represented by a single muscle.
This provides a minimal, interpretable model of planar arm reaching.

- Added a `FreePointMass24` effector class. This is a four-muscle two-dimensional point mass where each muscle
drives the mass along one cardinal direction (right, up, left, down), with constant moment arms. It extends the
existing `ReluPointMass24` class with no pathing geometry.

- Added a unit test suite covering effector construction and simulation (`test_effector.py`,
`test_simulation.py`), environment properties and spaces (`test_environment.py`), and non-differentiable
(reinforcement learning) mode including reward shape, numpy output, multi-episode cycling, action noise,
and deterministic seeding.

- Fixed a bug where independent sensory noise channels were correlated due to incorrect sampling. Noise is
now correctly decorrelated across channels. Thanks to Chris Versteeg for finding and fixing this!

- Fixed incorrect parsing of list-valued arguments passed to muscle constructors, which could silently produce
wrong parameter shapes.

- Fixed `Environment` initialization dependence on `reset()`. Previously, `_build_spaces()` called
`self.reset()` via virtual dispatch, causing subclass `reset()` to run during the parent `__init__()`. This
has been replaced by an analytical `_get_obs_size()` method. A `UserWarning` is now issued at class definition
time if a subclass overrides `get_obs()` without also overriding `_get_obs_size()`.

- Improved device management by overriding `nn.Module.to()` to ensure all internal tensors and sub-modules
are correctly moved when calling `.to(device)`. Thanks to Jonathan Cornford for flagging this issue!

- Gradient-free parameters (e.g., moment arm matrices, geometry constants) have been moved from
`nn.Parameter(requires_grad=False)` to `register_buffer`, which is the idiomatic PyTorch approach for
constant tensors that should participate in device and dtype management without accumulating gradients.

- Replaced bare `assert` statements throughout the codebase with descriptive `raise` statements. This provides
clearer error messages when invalid arguments are passed.

- Added type hints to the public API for improved IDE support and static analysis.

- Applied PEP8 formatting and style fixes across the codebase, including standardizing all tensor concatenation
calls to `torch.cat` and removing non-standard import aliases.


### <font size="4">Version 0.2.0</font>
*2024, January 4th*

First and foremost, this update moves `motornet` from `tensorflow` to `pytorch`. There has been systematic requests for 
a `pytorch` implementation of this package, and over time it is becoming clear that this will enable better integration
with existing research efforts from the scientific community that this package is aiming to help. As a consequence,
many API changes and change in the code structure were made, as the logical structure of `pytorch` is fundamentally
different than that of `tensorflow`. These changes are further detailed below.


- Renamed the `motornet.plants` package to `effector` and the `motornet.plants.Plant` class to `Effector`, as 'plant' 
is a specific engineering term and may be overly arcanic to a more general audience. Generally, the swap from "plant" 
to "effector" has been enacted consistently in the text and code.

- `Task` object essentially perform computations pertaining to `environment` objects in typical simulation software for
machine learning. Therefore, the `motornet.tasks` module has been renamed `motornet.environment` and the `Task` base 
class has been renamed to `Environment`. This is also now a subclass of [gymnasium](https://gymnasium.farama.org)'s
[gymnasium.Env](https://gymnasium.farama.org/api/env/#gymnasium-env) class, and it shares its API convention. The 
motivation behind these changes is that `gymnasium` is a popular interfacing package for simulation environments
in machine learning, and standardizing `motornet`'s API according to `gymnasium` will enable wider cross-compatibility,
as well as facilitate familiarization efforts from a lot of researchers already accustomed to 
`gymnasium`'s API. Users are strongly encouraged to check the updated tutorial notebooks on `motornet`'s GitHub
repository and on the online documentation website for more detailed explanation of the new `Environment` API, if they
are not alredy familiar with `gymnasium`. Generally, the swap from "task" to "environment" has been enacted consistently
in the text and code.

- `Pytorch` does not require the creation of end-to-end `model` objects as `tensorflow` does. Consequently, `motornet`
pipelines only require setting up an `Effector` and wrapping it up in an `Environment` object, without having to create 
a `Network` object at all. Feedback delays and Gaussian noise are now handled directly by the `Environment` class.

- Removed all sub-packages in `motornet`. The `pytorch` implementation allows users to create their own loss and network 
objects the way they typically would for any project beyond `motornet`, removing the need for a complex sub-packaging 
structure differentiating between set of modules falling under the  `nets` or `effectors` category. Therefore, 
`motornet` now only contains modules. For instance, the `motornet.effector.muscle.Muscle` class is now directly 
accessible as `motornet.muscle.Muscle`.

- The `motornet.plotor.plot_pos_over_time()` function now takes cartesian position as argument rather than full 
cartesian states that include positions and velocities. In practice, the velocities were always discarded by that 
function so we removed this step to allow for a more transparent and intuitive function syntax.

- The `muscle_type` argument of the `motornet.effector.Effector` class has been renamed to `muscle` for conciseness.

- The term `excitation` is now replaced by `action` to better match the terminology in place in continuous control 
machine learning. Note that `action` and `activation` are not the same variables.

- Added a `motornet.effector.muscles.MujocoHillMuscle` class to the `muscle` module. This object instantiates MuJoCo's
Hill-type muscle as described in
[the MuJoCo documentation](https://mujoco.readthedocs.io/en/stable/modeling.html#muscle-actuators).

- The `motornet.utils.parallelizer.py` file has been removed, as the means of streamlining model training pipelines 
usually boils down to personal preference.

- Users can now seed their `Environment` and `Effector` classes. Seeding is an important aspect of reproducible 
programming, and is usually considered a "best practice". Since the `Environment` and `Effector` classes are the only
classes that make use of a random generator, these are the only classes that currently require seeding in `motornet`.

- All `motornet` objects now inherit from the `torch.nn.Module` class. Amongst other things, this allows easy device 
assignment for model parameters, using `pytorch`'s usual `.to(device)` method.

- Renamed the `muscles`, `skeletons`, `effectors`, and `environments` modules to `muscle`, `skeleton`, `effector`, and
`environment` for conciseness.



### <font size="4">Version 0.1.5</font>
*2023, February 19th*

- Fixed a typo for a parameter value in the `mn.plants.muscles.RigidTendonHillMuscleThelen` class, from 0.66 to 0.6.
This parameter was epsilon_0^M in equation 3 of the main reference (Thelen, 2003).

- Random noise is now correctly applied to gated recurrent units in the `mn.nets.layers.GRUNetwork.forward_pass()`
method. Specifically, it is now applied before the non-linearity is applied rather than after.


### <font size="4">Version 0.1.4</font>
*2022, November 8th*
- Added an attribute alias object at `mn.utils.Alias` that allows users declare transparent aliases to object 
attributes.

- Declared an alias `state_names` for `output_names` in the mn.nets.Network base class.

- Fixed the first `state_name` of `ReluMuscle` class from `excitation/activation` to `activation`, as excitation
and activation are actually distinct variables. See that class' documentation for details. 


### <font size="4">Version 0.1.3</font>
*2022, October 30th*
- Fixed a bug which would prevent some new custom models from compiling due to mismatched sequence duration.

- Added a `ClippedPositionLoss`, which penalizes positional error unless the radial distance to the desired position is
less than a user-defined radius (target size) around said desired position (see documentation in the `mn.nets.losses`
module for more details).

- The `plot_pos_over_time` function in the `mn.utils.plotor` module can now take colormaps as a keyword argument (see
documentation for details.)

- Removed a numpy.ndarray from `CenterOutReach` attributes to allow for JSON serialization when saving models.

- Added a warning in `Task` base class to inform users when their task contains a numpy.ndarray as attribute. This is 
to make them aware that it might raise an error when saving models due to numpy.ndarray not being JSON serializable.

- Fixed an error in `mn.plants.Plant.get_muscle_cfg()` which occured when the method is called and  the `add_muscle` 
method was not called before. 


### <font size="4">Version 0.1.2</font>
*2022, July 31st*
- Some optional arguments, and associated attributes for `mn.plants.skeletons.TwoDofArm` at initialization are now 
case-insensitive.

- Removed a typo that resulted in printing of some state shapes in `mn.plants.skeletons.TwoDofArm.path2cartesian()` 
method.


### <font size="4">Version 0.1.1</font>
*2022, June 4th*
- Fixed `setup.py` to allow for solving the `tensorflow` dependency on M1-chip equipped Apple devices. Instead of asking
for `tensorflow` as a requirement upon `pip install` calls, `motornet` will now ask for the M1-compatible version of 
TensorFlow on machines equipped with M1 chips, which is [tensorflow-macos](https://pypi.org/project/tensorflow-macos/).


### <font size="4">Version 0.1.0</font>
*2022, June 3th*
- Initial release.