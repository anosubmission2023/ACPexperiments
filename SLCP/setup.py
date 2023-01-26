import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "SLCP",
    version = "0.0.0",
    author = "NA",
    description = ("Install SLCP to run ACP experiements"),
    packages=['SLCP', 'SLCP.all_experiments', 'SLCP.cqr', 'SLCP.datasets',
              'SLCP.nonconformist'],
    long_description=read('README.md'),
)