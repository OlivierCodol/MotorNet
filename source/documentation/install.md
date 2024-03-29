# How to Install

## Install from source

To install the latest development release, you can install directly from GitHub's repository.
```
pip install git+https://github.com/OlivierCodol/MotorNet.git
```


## Install with `pip`

First, please make sure that the latest `pip` version is installed in your working environment.

```
python3 -m pip install -U pip
```

Then you can install `motornet` using a simple `pip install` command.
```
python3 -m pip install motornet
```

## Install with Anaconda

Installation via Anaconda is not currently supported.

<h1>Requirements</h1>

There is no third-party software required to run MotorNet. However, some freely available python dependencies are required.

- [PyTorch](https://pytorch.org/docs/stable/torch.html): MotorNet relies on PyTorch to create tensors and build the 
graph.
- [NumPy](https://numpy.org/): For array and matrix computations when not using tensors.
- [Gymnasium](https://numpy.org/): `motornet` environments inherit from `gymnasium` environments.
- [Matplotlib](https://matplotlib.org/): For plotting utilities, mainly in the 
`plotor.py` module.

If you are running a version earlier than 0.2.0, which relies on TensorFlow, the requirements are as follow.

- [TensorFlow](https://www.tensorflow.org/): MotorNet is first and foremost built on TensorFlow. However, the standard
TensorFlow toolbox is not compatible with recent Apple machines equipped with M1 silicon chips, and users must rely on 
an adapted version called [tensorflow-macos](https://pypi.org/project/tensorflow-macos/). When installing MotorNet, the 
`setup.py` routine will automatically check the machine's OS platform and hardware to assess whether to solve for the 
`tensorflow` or `tensorflow-macos` dependency. 
- [NumPy](https://numpy.org/): For array and matrix computations when not using tensors.
- [Matplotlib](https://matplotlib.org/): For plotting utilities, mainly in the 
`plotor.py` module.
- [IPython](https://ipython.org/): Mainly for
[callbacks](https://github.com/OlivierCodol/MotorNet/blob/master/motornet/nets/callbacks.py) that output training 
metrics during model training.
- [joblib](https://joblib.readthedocs.io/en/latest/): For parallelization routines in the 
[parallelizer](https://github.com/OlivierCodol/MotorNet/blob/master/motornet/utils/parallelizer.py) script.


<h1>Tutorials</h1>

There are several tutorials available to get you started, available in the
[repository](https://github.com/OlivierCodol/MotorNet)'s
[examples](https://github.com/OlivierCodol/MotorNet/tree/master/examples) folder, as well as on the 
[documentation website](https://oliviercodol.github.io/MotorNet/build/html/index.html). Hopefully they will give a sense
of how the  API is supposed to work.


<div class="admonition note">
<p class="admonition-title">Note</p>
<p>Tutorials and API documentation for version 0.1.5 are still available on the website and GitHib repository for those
who wish to consult them. They will remain available for the foreseeable future.</p>
</div>
