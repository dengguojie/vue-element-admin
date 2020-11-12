import os
import sys
import sysconfig
from setuptools import setup, find_packages, Extension
from setuptools.dist import Distribution

__version__ = "0.1"


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False


with open("MANIFEST.in", "w") as fo:
    fo.write("include libs/libmodel_run_tool.so\n")

setup_kwargs = {
    "include_package_data": True
}

setup(
    name="op_test_frame",
    version=__version__,
    zip_safe=False,
    packages=find_packages(),
    install_requires=[],
    distclass=BinaryDistribution,
    **{
        "include_package_data": True,
        "data_files": [('', ['LICENSE']),
                       ('lib', ['op_test_frame/libs/libmodel_run_tool.so'])]
    }
)
