from setuptools import find_packages, setup


with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name="hw1",
    packages=find_packages(),
    version="1.0.3",
    description="1st homework in mlops",
    author="Makezh",
    install_requires=required,
    license="MIT",
)