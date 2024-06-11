# Copyright 2016 United Kingdom Atomic Energy Authority (UKAEA)
#
# This file is part of IDLBridge.
#
# IDLBridge is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# IDLBridge is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with IDLBridge. If not, see <http://www.gnu.org/licenses/>.

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import sys
import numpy
import os
import os.path as path
import platform

# set the code version
__version__ = "1.1.1"

force = False
profile = False

if "--force" in sys.argv:
    force = True
    del sys.argv[sys.argv.index("--force")]

if "--profile" in sys.argv:
    profile = True
    del sys.argv[sys.argv.index("--profile")]

# configure IDL paths
bits, _ = platform.architecture()
if bits == "32bit":
    machine = "x86"
elif bits == "64bit":
    machine = "x86_64"
else:
    raise Exception("Platform type could not be determined.")

idl_library_path = os.environ["IDL_DIR"] + "/bin/bin.linux." + machine
idl_include_path = os.environ["IDL_DIR"] + "/external/include"

# configure include and library paths
include_dirs = [".", numpy.get_include(), idl_include_path]
library_dirs = [idl_library_path]
libraries = ["idl"]
extra_compile_args = []
extra_link_args = ["-Wl,-rpath,.", "-Wl,-rpath,{}".format(idl_library_path)]

setup_path = path.dirname(path.abspath(__file__))

# build extension list
extensions = []
for root, dirs, files in os.walk(setup_path):
    for file in files:
        if path.splitext(file)[1] == ".pyx":
            pyx_file = path.relpath(path.join(root, file), setup_path)
            module = path.splitext(pyx_file)[0].replace("/", ".")
            extensions.append(Extension(module,
                                        [pyx_file],
                                        include_dirs=include_dirs,
                                        libraries=libraries,
                                        library_dirs=library_dirs,
                                        extra_compile_args=extra_compile_args,
                                        extra_link_args=extra_link_args))

if profile:
    directives = {"profile": True}
else:
    directives = {}

setup(
    packages=["idlbridge"],
    ext_modules=cythonize(extensions, force=force, compiler_directives=directives)
)
