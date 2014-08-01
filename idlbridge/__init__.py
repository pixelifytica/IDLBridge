# Copyright (c) 2014, Culham Centre For Fusion Energy
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of the Culham Centre For Fusion Energy nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE CULHAM CENTRE FOR FUSION ENERGY BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Allows users to call IDL routines from inside Python.

This package wraps the IDL callable library and provides a simple user interface
for passing data between IDL and Python. IDL functions and procedures can be
exposed in Python and be called like native Python functions.
"""

import ctypes as _ctypes
from . import _core
from ._version import __version__

__author__ = 'Dr. Alex Meakins'
__responsible_officer__ = 'Dr. Alex Meakins'

# By default the core (IDL) library is opened by Python with RTLD_LOCAL
# preventing subsequently loaded IDL DLM libraries from seeing the IDL_*
# symbols. To solve this we reload the library with RTLD_GLOBAL prior to
# its first use.
_ctypes.CDLL(_core.__file__, mode=_ctypes.RTLD_GLOBAL)

# module level bridge and functions
__bridge__ = _core.IDLBridge()


def execute(command):
    """
    Executes a string as an IDL command.

    :param command: A string defining the idl command e.g. "d = 5.1"
    :return: None
    """

    global __bridge__
    __bridge__.execute(command)


def get(variable):
    """
    Returns a Python copy of the specified IDL variable.

    :param variable: A string containing the variable name.
    :return: The IDL variable data converted to an appropriate Python type.
    """

    global __bridge__
    return __bridge__.get(variable)


def put(variable, data):
    """
    Sets an IDL variable with Python data.

    :param variable: A string containing the variable name.
    :param data: A Python object containing data to send.
    """

    global __bridge__
    __bridge__.put(variable, data)


def delete(variable):
    """
    Deletes the specified IDL variable.

    :param variable: A string specifying the name of the IDL variable to delete.
    :return: None
    """

    global __bridge__
    __bridge__.delete(variable)


def export_function(name):
    """
    Wraps an IDL function in an object that behaves like a Python function.

    For example, to gain access to the IDL "sin" function type:

        sin = idl.export_function("sin")

    Use "sin" like an ordinary Python function:

        v = sin(0.5)

    Keyword arguments are specified using the normal Python syntax. To provide
    an IDL "/keyword", simply set the keyword equal to True in Python.

        my_idl_function(1.2, 3.4, my_keyword=True)

    :param None: A string specifying the name of the IDL function to wrap.
    :return: An IDLFunction object.
    """

    global __bridge__
    return __bridge__.export_function(name)


def export_procedure(name):
    """
    Wraps an IDL procedure in an object that behaves like a Python function.

    For example, to gain access to the IDL "plot" procedure type:

        plot = idl.export_procedure("plot")

    Use "plot" like an ordinary Python function:

        plot([1,2,3], [4,5,6])

    Keyword arguments are specified using the normal Python syntax. To provide
    an IDL "/keyword", simply set the keyword equal to True in Python.

        my_idl_procedure(1.2, 3.4, my_keyword=True)

    :param None: A string specifying the name of the IDL procedure to wrap.
    :return: An IDLProcedure object.
    """

    global __bridge__
    return __bridge__.export_procedure(name)


