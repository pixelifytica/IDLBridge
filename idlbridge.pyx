# TODO: add license

# include IDL callable interface
include "idl_export.pxi"


class IDLLibraryError(Exception):
    """
    An IDL Library Error Exception.

    This exception is thrown when there is a problem interacting with the IDL
    callable library.
    """

    pass


class IDLTypeError(Exception):
    """
    An IDL Type Error Exception.

    This exception is thrown when a data type is to the IDL library that it is
    unable to handle.
    """

    pass

# global variables used to track the usage of the IDL library
__references__ = 0
__shutdown__ = False


cdef _initialise_idl(bint quiet=False):

    cdef IDL_INIT_DATA init_data
    global __shutdown__

    if not __shutdown__:

        # initialise the IDL library
        init_data.options = IDL_INIT_NOCMDLINE

        if quiet:

            init_data.options |= IDL_INIT_QUIET

        v = IDL_Initialize(&init_data)

    else:

        # library has already been cleaned up, it can not be reinitialised
        raise IDLLibraryError("The IDL library has been shutdown, it can not be restarted in this session.")


cdef _cleanup_idl():

    global __shutdown__

    if not __shutdown__:

        IDL_Cleanup(True)


cdef _register():

    global __references__
    global __shutdown__

    if __references__ == 0:

        _initialise_idl()

    __references__ += 1


cdef _deregister():

    global __references__
    global __shutdown__

    __references__ -= 1

    if __references__ <= 0:

        _cleanup_idl()


cdef class _IDLCallable:

    pass


cdef class IDLFunction(_IDLCallable):

    pass


cdef class IDLProcedure(_IDLCallable):

    pass


cdef class IDLBridge:
    """

    """

    def __init__(self):

        # register this bridge with the IDL library
        _register()

    def __dealloc__(self):

        # de-register this bridge with the IDL library
        _deregister()

    cpdef object execute(self, unicode command):

        byte_string = command.encode("UTF8")
        IDL_ExecuteStr(byte_string)

    cpdef object get(self, unicode variable):

        return None

    cpdef object put(self, unicode variable, object data):

        pass

    cpdef object delete( self, unicode variable):

        pass

    def export_function(self, name):

        return IDLFunction(name, idl_bridge=self)

    def export_procedure(self, name):

        return IDLProcedure(name, idl_bridge=self)


# module level bridge and functions
cdef IDLBridge __module_bridge__ = IDLBridge()

def execute(commands):

    global __module_bridge__
    __module_bridge__.execute(commands)

def get(variable):

    global __module_bridge__
    return __module_bridge__.get(variable)

def put(variable, data):

    global __module_bridge__
    __module_bridge__.put(variable, data)

def delete(variable):

    global __module_bridge__
    __module_bridge__.delete(variable)

def export_function(name):

    global __module_bridge__
    return __module_bridge__.export_function(name)

def export_procedure(name):

    global __module_bridge__
    return __module_bridge__.export_procedure(name)



