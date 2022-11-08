# Changelog

## Development releases


<!--
### <font size="4">Version 0.1.3</font>
*YYYY, Month DDth*
-->


<!--
### <font size="4">Version 0.1.3</font>
*YYYY, Month DDth*
- Added an attribute alias object at `mn.utils.Alias that allows users declare transparent aliases to object attributes.

- Declared an alias `state_names` for `output_names` in the mn.nets.Network base class.

- Fixed the first `state_name` of `ReluMuscle` class from `excitation/activation` to `activation`, as excitation
and activation are actually distinct variables. See that class' documentation for details. 
-->


### <font size="4">Version 0.1.3</font>
*2022, October 30th*
- Fixed a bug which would prevent some new custom models from compiling due to mismatched sequence duration.

- Added a `ClippedPositionLoss`, which penalizes positional error unless the radial distance to the desired position is
less than a user-defined radius (target size) around said desired position (see documentation in the `mn.nets.losses`
module for more details.)

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