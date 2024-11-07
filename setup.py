# +
from setuptools import setup

setup(
    name="skfin",
    version="0.1",
    description="a machine-learning library for portfolio management and trading",
    author="Sylvain Champonnois",
    author_email="sylvain.champonnois@gmail.com",
    license="Apache License",
    packages=["skfin"],
    include_package_data=True,
    platforms=["linux", "unix"],
    python_requires=">=3.6",
)

