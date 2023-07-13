from setuptools import setup

install_req = [
    "numpy",
    "torch",
    "gymnasium",
]

if __name__ == "__main__":
    setup(
        install_requires=install_req,
    )
