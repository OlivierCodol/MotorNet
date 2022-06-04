# Changelog

## Development releases

### <font size="4">Version 0.1.1</font>
*2022, June 4th*
- Fixed `setup.py` to allow for solving the `tensorflow` dependency on M1-chip equipped Apple devices. Instead of asking
for `tensorflow` as a requirement upon `pip install` calls, `motornet` will now ask for the M1-compatible version of 
TensorFlow on machines equipped with M1 chips, which is [tensorflow-macos](https://pypi.org/project/tensorflow-macos/).


### <font size="4">Version 0.1.0</font>
*2022, June 3th*
- Initial release.