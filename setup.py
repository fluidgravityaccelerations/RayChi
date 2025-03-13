gifrom setuptools import setup, find_packages

setup(
    name="RayChi",
    version="0.1.9",
    description="A Taichi-Powered Ray Tracer",
    author="Fluid Gravity Accelerated Solutions",
    author_email="fluidgravityaccelerations@gmail.com",
    packages=find_packages(),
    install_requires=[
        "taichi",
        "numpy",
        "matplotlib",
    ],
    entry_points={
        "console_scripts": [
            "raychi=raychi.cli:main",
        ],
    },
)
