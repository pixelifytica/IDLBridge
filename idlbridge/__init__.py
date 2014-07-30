import ctypes
from . import _core

__author__ = 'Dr Alex Meakins'
__responsible_officer__ = 'Dr Alex Meakins'

# By default the core (IDL) library is opened by Python with RTLD_LOCAL,
# preventing subsequently loaded IDL DLM libraries from seeing the IDL_*
# symbols. To solve this we reload the library with RTLD_GLOBAL prior to
# its first use.
ctypes.CDLL(_core.__file__, mode=ctypes.RTLD_GLOBAL)

# module level bridge and functions
__bridge__ = _core.IDLBridge()


def execute(command):

    global __bridge__
    __bridge__.execute(command)


def get(variable):

    global __bridge__
    return __bridge__.get(variable)


def put(variable, data):

    global __bridge__
    __bridge__.put(variable, data)


def delete(variable):

    global __bridge__
    __bridge__.delete(variable)


def export_function(name):

    global __bridge__
    return __bridge__.export_function(name)


def export_procedure(name):

    global __bridge__
    return __bridge__.export_procedure(name)


