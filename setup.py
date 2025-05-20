from setuptools import find_packages, setup

setup(
    name="dimer_models",
    version="0.0",
    description="Dimers and Bipartite Lattices",
    long_description="",
    author="Peru D'Ornellas",
    author_email="peru.dornellas@gmail.com",
    license="Apache Software License",
    home_page="",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.2",
        "scipy",
        "matplotlib",
        "flake8",
        "koala",
        "pytest",
        "pytest-cov",
        "pytest-xdist",
        "mpire",
    ],
)
