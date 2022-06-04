from setuptools import setup
import platform

system = platform.system().lower()
processor = platform.machine()
tensorflow_version = "tensorflow-macos" if system == "darwin" and processor == "arm64" else "tensorflow"

install_req = [
    tensorflow_version,
    "numpy",
    "matplotlib",
    "ipython",
    "joblib",
]

if __name__ == "__main__":
    setup(
        install_requires=install_req,
    )
