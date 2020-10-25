from setuptools import setup, find_packages

from reqs_triage.__version__ import __version__

setup(
    name="reqs_triage",
    description="Utilities for ingesting legislative data",
    author="Matt Robinson",
    author_email="mrobinson23@gwu.edu",
    packages=find_packages(),
    version=__version__,
    entry_points={"console_scripts": "reqs_triage=reqs_triage.cli:main"},
)
