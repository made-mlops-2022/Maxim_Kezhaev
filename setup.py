from setuptools import find_packages, setup


with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name="hw1",
    packages=find_packages(),
    version="0.1.0",
    description="1st homework in mlops",
    author="Makezh",
    entry_points={
        "console_scripts": [
            "ml_example_train = ml_example.train_pipeline:train_pipeline_command"
        ]
    },
    install_requires=required,
    license="MIT",
)