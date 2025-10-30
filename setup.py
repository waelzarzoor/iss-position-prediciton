from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = [line.strip() for line in f.readlines()]

info = {
    "name": "modules",
    "version": "1.0.0",
    "maintainer": "Dawid Mazur",
    "maintainer_email": "dawid.mazur@icloud.com",
    "url": "https://github.com/Dawidemm/iss-position-prediction",
    "license": "MIT License",
    "packages": find_packages(),
    "description": "Python project to predict position of the ISS.",
    "long_description": open("README.md").read(),
    "long_description_content_type": "text/markdown",
    "install_requires": requirements,
    "include_package_data": True,
}

classifiers = [
    "Programming Language :: Python :: 3.10",
    "Topic :: Scientific/Engineering :: Engineering",
]

setup(classifiers=classifiers, **(info))