"""
Allows users to call IDL routines from inside Python.

This package wraps the IDL callable library and provides a simple user interface
for passing data between IDL and Python. IDL functions and procedures can be
exposed in Python and be called like native Python functions.
"""

import ctypes as _ctypes
from . import _core

__author__ = 'Dr. Alex Meakins'
__responsible_officer__ = 'Dr. Alex Meakins'
__version__ = "1.1.0"

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

# todo: update docstring
def export_function(name, return_arguments=None):
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
    return __bridge__.export_function(name, return_arguments)

# todo: update docstring
def export_procedure(name, return_arguments=None):
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
    return __bridge__.export_procedure(name, return_arguments)


