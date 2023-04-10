# +
from setuptools import setup

if __name__ == "__main__":
    setup(
        name="skfin",
        description="a machine-learning library for portfolio management and trading",
        author="Sylvain Champonnois",
        author_email="sylvain.champonnois@gmail.com",
        license="Apache License",
        packages=["skfin"],
        include_package_data=True,
        platforms=["linux", "unix"],
        python_requires=">=3.6",
    )

