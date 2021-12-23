# MotorNet

This repository contains MotorNet, a python package that allows training Recurrent Neural Networks to control for
biomechanically realistic effectors. This toolbox is designed to meet several goals:

- No hard dependencies beyond typically available python packages. This should make it easier to run this package in remote computing units.
- Provide users with a variety of muscle types to chose from.
- Flexibility in creating new, arbitrary muscle wrappings around the skeleton, to enable fast exploration of
different potential effectors and how to control them. The moment arm are calculated online according to the 
geometry of the skeleton and the (user-defined) paths of the muscles.
- User-friendly API, to allow for easier familiarization and intuitive usage. We want to enable a focus on ideas, not implementation.
Currently, we focus on subclassing to allow users to implement their custom task designs, custom plants, and custom controllers.

## State of the project

The package is technically functional, and is used by several people to progress in their research.
However we are still in testing and developpment (pre-alpha), and we are currently focusing on documenting,
commenting, and cleaning the code. For now, the package is available "as-is" but we hope to soon move to an
alpha and beta stage. Please feel free to log an issue if you think you found a bug, we appreciate any contribution. 
Stay tuned for more!

## Dependencies

There are no dependencies outside of Python 3. The packages required for MotorNet to function are:
- Tensorflow (successfully tested with v2.7)
- NumPy (successfully tested with v1.21)

## Tutorials

There are a couple of tutorials available to get you started (see <em>tutorials</em> folder). Hopefully they will give a sense of how the 
API is supposed to work. As indicated above, more furnished documentation is on the way, hopefully very soon!

