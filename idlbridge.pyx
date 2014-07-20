# TODO: add license

from numpy cimport ndarray

# include IDL callable interface
include "idl_export.pxi"


class IDLLibraryError(Exception):
    """
    An IDL Library Error Exception.

    This exception is thrown when there is a problem interacting with the IDL
    callable library.
    """

    pass


class IDLTypeError(TypeError):
    """
    An IDL Type Error Exception.

    This exception is thrown when a data type is passed to the IDL library that
    it is unable to handle.
    """

    pass


class IDLValueError(ValueError):
    """
    An IDL Value Error Exception.

    This exception is thrown when a value is passed to the IDL library that it
    isunable to handle.
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

    cpdef object execute(self, str command):

        byte_string = command.encode("UTF8")
        IDL_ExecuteStr(byte_string)

    cpdef object get(self, str variable):

        cdef IDL_VPTR vptr

        # convert unicode string to c compatible byte string
        byte_string = variable.encode("UTF8")

        # request variable from IDL
        vptr = IDL_FindNamedVariable(byte_string, False)

        if vptr == NULL or vptr.type == IDL_TYP_UNDEF:

            raise IDLValueError("Variable {} not found.".format(variable.upper()))

        # identify variable type and translate to python
        if vptr.flags & IDL_V_ARR:

            if vptr.flags & IDL_V_STRUCT:

                return self._get_structure(vptr)

            else:

                return self._get_array(vptr)

        elif vptr.type == IDL_TYP_BYTE:

            return vptr.value.c

        elif vptr.type == IDL_TYP_INT:

            return vptr.value.i

        elif vptr.type == IDL_TYP_UINT:

            return vptr.value.ui

        elif vptr.type == IDL_TYP_LONG:

            return vptr.value.l

        elif vptr.type == IDL_TYP_ULONG:

            return vptr.value.ul

        elif vptr.type == IDL_TYP_LONG64:

            return vptr.value.l64

        elif vptr.type ==  IDL_TYP_ULONG64:

            return vptr.value.ul64

        elif vptr.type == IDL_TYP_FLOAT:

            return vptr.value.f

        elif vptr.type == IDL_TYP_DOUBLE:

            return vptr.value.d

        elif vptr.type == IDL_TYP_COMPLEX:

            return complex(vptr.value.cmp.r, vptr.value.cmp.i)

        elif vptr.type == IDL_TYP_DCOMPLEX:

            return complex(vptr.value.dcmp.r, vptr.value.dcmp.i)

        elif vptr.type == IDL_TYP_STRING:

            return self._get_string(vptr)

        elif vptr.type == IDL_TYP_PTR:

            raise NotImplementedError("Pointer types are not supported.")

        elif vptr.type == IDL_TYP_OBJREF:

            raise NotImplementedError("Object types are not supported.")

        else:

            raise IDLTypeError("Unrecognised IDL type.")

    cdef inline ndarray _get_array(self, IDL_VPTR vptr):

        # TODO: implement me
        raise NotImplementedError("Not currently implemented.")

    cdef inline dict _get_structure(self, IDL_VPTR vptr):

        raise NotImplementedError("Not currently implemented.")

    cdef inline str _get_string(self, IDL_VPTR vptr):

        # The string pointer in the IDL string structure is invalid when the string length is zero.
        if vptr.value.str.slen == 0:

            return ""

        else:

            return vptr.value.str.s.decode("UTF8")

    cpdef object put(self, str variable, object data):

        pass

    cpdef object delete(self, str variable):

        cdef IDL_VPTR vptr

        byte_string = variable.encode("UTF8")
        vptr = IDL_FindNamedVariable(byte_string, False)

        if vptr == NULL or vptr.type == IDL_TYP_UNDEF:

            raise IDLValueError("Variable {} not found.".format(variable.upper()))

        IDL_Delvar(vptr)

    cpdef object export_function(self, str name):

        return IDLFunction(name, idl_bridge=self)

    cpdef object export_procedure(self, str name):

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
cdef IDLBridge __bridge__ = IDLBridge()

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



