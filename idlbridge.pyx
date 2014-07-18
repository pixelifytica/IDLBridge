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

    cpdef object export_function(self, unicode name):

        return IDLFunction(name, idl_bridge=self)

    cpdef object export_procedure(self, unicode name):

        return IDLProcedure(name, idl_bridge=self)


class _IDLCallable:

    def __init__(self, name, idl_bridge=IDLBridge()):

        self.name = name
        self._idl = idl_bridge
        self._id = idl_bridge._id

    def _process_arguments(self, arguments, temporary_variables):

        # parse arguments
        argument_variables = []
        for index, argument in enumerate(arguments):

            # transfer argument to idl
            tempvar = "_idlbridge_arg{index}".format(index=index)
            self._idl.put(tempvar, argument)

            # record temporary variable for later cleanup
            argument_variables.append(tempvar)

        # add argument temporaries to list of temporary variables
        temporary_variables += argument_variables

        # build and return argument command string fragment
        return ", ".join(argument_variables)

    def _process_keywords(self, keywords, temporary_variables):

        keyword_strings = []
        for index, (key, argument) in enumerate(keywords.items()):

            # transfer argument
            tempvar = "idlbridge_kw{index}".format(index=index)
            self._idl.put(tempvar, argument)

            # generate key string for command
            keyword_strings.append("{key}={var}".format(key=key, var=tempvar))

            # record variable for later cleanup
            temporary_variables.append(tempvar)

        # build and return keyword command string fragment
        return ", ".join(keyword_strings)


class IDLFunction(_IDLCallable):

    def __call__(self, *arguments, **keywords):

        temporary_variables = []

        # pass arguments to idl and assemble the relevant command string fragments
        argument_fragment = self._process_arguments(arguments, temporary_variables)
        keyword_fragment = self._process_keywords(keywords, temporary_variables)

        # assemble command string
        return_variable = "_idlbridge_return".format(id=self._id)
        command = "{rtnvar} = {name}({arg}".format(rtnvar=return_variable, name=self.name, arg=argument_fragment)

        if argument_fragment and keyword_fragment:

            # need an extra comma to separate arguments from keywords
            command += ", "

        command += "{key})".format(key=keyword_fragment)

        # execute command and obtain returned data
        self._idl.execute(command)
        data = self._idl.get(return_variable)

        # clean up
        for variable in temporary_variables:

            self._idl.delete(variable)

        self._idl.delete(return_variable)

        return data


class IDLProcedure(_IDLCallable):

    def __call__(self, *arguments, **keywords):

        temporary_variables = []

        # pass arguments to idl and assemble the relevant command string fragments
        argument_fragment = self._process_arguments(arguments, temporary_variables)
        keyword_fragment = self._process_keywords(keywords, temporary_variables)

        # assemble command string
        if argument_fragment or keyword_fragment:

            command = "{name}, {arg}".format(name=self.name, arg=argument_fragment)

            if argument_fragment and keyword_fragment:

                # need an extra comma to separate arguments from keywords
                command += ", "

            command += keyword_fragment

        else:

            command = self.name

        # execute command
        self._idl.execute(command)

        # clean up
        for variable in temporary_variables:

            self._idl.delete(variable)


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



