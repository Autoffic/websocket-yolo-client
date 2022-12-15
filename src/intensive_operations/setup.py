from setuptools import setup, Extension, Command
from Cython.Build import cythonize
from pathlib import Path
import sys, os

import distutils.command.build

FILE = Path(__file__).resolve()
INTENSIVE_OPERATIONS = FILE.parents[0]  # intensive_operations directory
if str(INTENSIVE_OPERATIONS) not in sys.path:
    sys.path.append(str(INTENSIVE_OPERATIONS))  # add intensive_operations to PATH
INTENSIVE_OPERATIONS = Path(os.path.relpath(INTENSIVE_OPERATIONS, Path.cwd()))  # relative

# Override build command
class BuildCommand(distutils.command.build.build):
    def initialize_options(self):
        cython_build_dir = str(Path(str(INTENSIVE_OPERATIONS) + "/cython_build_dir").resolve())  # (WIP) for setting different output dir

        if not os.path.exists(cython_build_dir):
            os.mkdir(cython_build_dir)

        distutils.command.build.build.initialize_options(self)
        self.build_base = cython_build_dir

USE_CYTHON = True

ext = '.py' if USE_CYTHON else '.c'

extensions = [Extension("centroidtracker", [str(Path(str(INTENSIVE_OPERATIONS) + "/centroidtracker"+ext).resolve())]),
              Extension("checkpassingvehicle", [str(Path(str(INTENSIVE_OPERATIONS) + "/checkpassingvehicle"+ext).resolve())]),
              Extension("filterrois", [str(Path(str(INTENSIVE_OPERATIONS) + "/filterrois"+ext).resolve())])]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

setup(
    name='Intensive operation with cython',
    ext_modules=cythonize(extensions,
        annotate=True,
        ),
    zip_safe=False
)