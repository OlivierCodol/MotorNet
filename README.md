# MotorNet

This repository contains **MotorNet**, a python package that allows training recurrent neural networks to control for
biomechanically realistic effectors. This toolbox is designed to meet several goals:

- No hard dependencies beyond typically available python packages. This should make it easier to run this package on remote computing units.
- Provide users with a variety of muscle types to chose from.
- Flexibility in creating new, arbitrary muscle wrappings around the skeleton, to enable fast exploration of
different potential effectors and how to control them. The moment arm are calculated online according to the 
geometry of the skeleton and the (user-defined) paths of the muscles.
- User-friendly API, to allow for easier familiarization and intuitive usage. We want to enable a focus on ideas, not implementation.
The toolbox focuses on subclassing to allow users to implement their custom task designs, custom plants, and custom controller networks.
- Open-source, to allow for user contribution and community-driven incremental progress.

## State of the project

The package is feature complete, and is used by several people to progress in their research.
We are currently in beta phase, meaning the toolbox is publicly available but we are still on the lookout for potential
bugs and fixes to apply. Please feel free to log an issue if you think you found a bug, we appreciate any contribution. 
Stay tuned for more!

An [online documentation](motornet.org) is available. Feel free to 
check it out.

## How to Install

### Install from source

To install the latest development release, you can install directly from GitHub's repository. This will install 
version 0.2.0, which relies on PyTorch instead of TensorFlow. Please see the 
[changelog](https://oliviercodol.github.io/MotorNet/build/documentation/changelog.html) for more details on the
difference between the current development release and the PyPI release.

```
pip install git+https://github.com/OlivierCodol/MotorNet.git@pytorch
```

**Please see the staged changes at the bottom of this file to see the changes currently implemented on this branch.**

### Install with `pip`

NOTE: The current PyPI version of `motornet` is 0.1.5, which relies on TensorFlow

First, please make sure that the latest `pip` version is installed in your working environment.

```
python3 -m pip install -U pip
```

Then you can install `motornet` using a simple `pip install` command.
```
python3 -m pip install motornet
```

### Install with Anaconda

Installation via Anaconda is currently not supported.


## Requirements

There is no third-party software required to run MotorNet. However, some freely available python dependencies are 
required.

If you are running the current development release (version 0.2.0), the requirements are as follows.

- [PyTorch](https://pytorch.org/docs/stable/torch.html): MotorNet relies on PyTorch to create tensors and build the 
graph.
- [NumPy](https://numpy.org/): For array and matrix computations when not using tensors.
- [Gymnasium](https://numpy.org/): `motornet` environments are child classes of `gymnasium` environments.
- [Matplotlib](https://matplotlib.org/): For plotting utilities, mainly in the 
[plotor](https://github.com/OlivierCodol/MotorNet/blob/master/motornet/plotor.py) module.


If you are running the current PyPI release (version 0.1.5), which relies on TensorFlow, the requirements are as follows.

- [TensorFlow](https://www.tensorflow.org/): MotorNet is first and foremost built on TensorFlow. However, the standard
TensorFlow toolbox is not compatible with recent Apple machines equipped with M1 silicon chips, and users must rely on 
an adapted version called [tensorflow-macos](https://pypi.org/project/tensorflow-macos/). When installing MotorNet, the 
`setup.py` routine will automatically check the machine's OS platform and hardware to assess whether to solve for the 
`tensorflow` or `tensorflow-macos` dependency. 
- [NumPy](https://numpy.org/): For array and matrix computations when not using tensors.
- [Matplotlib](https://matplotlib.org/): For plotting utilities, mainly in the 
[plotor](https://github.com/OlivierCodol/MotorNet/blob/master/motornet/utils/plotor.py) module.
- [IPython](https://ipython.org/): Mainly for
[callbacks](https://github.com/OlivierCodol/MotorNet/blob/master/motornet/nets/callbacks.py) that output training 
metrics during model training.
- [joblib](https://joblib.readthedocs.io/en/latest/): For parallelization routines in the 
[parallelizer](https://github.com/OlivierCodol/MotorNet/blob/master/motornet/utils/parallelizer.py) script.


## Tutorials

There are several tutorials available to get you started, available in the
[repository](https://github.com/OlivierCodol/MotorNet)'s
[<em>examples</em>](https://github.com/OlivierCodol/MotorNet/tree/master/examples) folder, as well as on the 
[documentation website](motornet.org). Hopefully they will give a sense
of how the  API is supposed to work.

Tutorials and API documentation for version 0.1.5 are still available on the website and GitHib repository for those
who wish to consult them. They will remain available for the foreseeable future.

## Changelog

See [here](https://oliviercodol.github.io/MotorNet/build/documentation/changelog.html) for a curated log of update 
contents [**Note: This will redirect to the main branch's changelog**].


### <font size="4">Version 0.2.0 - Staged changes</font>

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
