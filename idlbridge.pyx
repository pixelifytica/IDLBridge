# TODO: add license

from libc.string cimport memcpy
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np

# include IDL callable interface
include "idl_export.pxi"

# initialise numpy c-api
np.import_array()


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
    is unable to handle.
    """

    pass


# global variables used to track the usage of the IDL library
__references__ = 0
__shutdown__ = False


cdef _initialise_idl(bint quiet=True):

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
        """
        Executes a string as an IDL command.

        :param command: A string defining the idl command e.g. "d = 5.1"
        :return: None
        """

        byte_string = command.encode("UTF8")
        IDL_ExecuteStr(byte_string)

    cpdef object get(self, str variable):
        """
        Returns a python copy of the specified IDL variable.

        :param variable: A string containing the variable name.
        :return: The IDL variable data converted to python form.
        """

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

        elif vptr.type == IDL_TYP_BYTE: return vptr.value.c
        elif vptr.type == IDL_TYP_INT: return vptr.value.i
        elif vptr.type == IDL_TYP_UINT: return vptr.value.ui
        elif vptr.type == IDL_TYP_LONG: return vptr.value.l
        elif vptr.type == IDL_TYP_ULONG: return vptr.value.ul
        elif vptr.type == IDL_TYP_LONG64: return vptr.value.l64
        elif vptr.type == IDL_TYP_ULONG64: return vptr.value.ul64
        elif vptr.type == IDL_TYP_FLOAT: return vptr.value.f
        elif vptr.type == IDL_TYP_DOUBLE: return vptr.value.d
        elif vptr.type == IDL_TYP_COMPLEX: return complex(vptr.value.cmp.r, vptr.value.cmp.i)
        elif vptr.type == IDL_TYP_DCOMPLEX: return complex(vptr.value.dcmp.r, vptr.value.dcmp.i)
        elif vptr.type == IDL_TYP_STRING: return self._string_idl_to_py(vptr.value.str)
        elif vptr.type == IDL_TYP_PTR:

            raise NotImplementedError("Pointer types are not supported.")

        elif vptr.type == IDL_TYP_OBJREF:

            raise NotImplementedError("Object types are not supported.")

        else:

            raise IDLTypeError("Unrecognised IDL type.")

    cdef inline np.ndarray _get_array(self, IDL_VPTR vptr):
        """
        Converts an IDL array to an equivalent numpy array.

        :param vptr: An IDL variable pointer pointing to an IDL array definition.
        :return: A numpy array.
        """

        # All arrays MUST be copied otherwise changes in python or IDL will cause segfaults in one of the other

        # Different array types unfortunately need special handling.
        if vptr.type == IDL_TYP_STRING:

            return self._get_array_string(vptr)

        elif vptr.type == IDL_TYP_STRUCT:

            return self._get_array_structure(vptr)

        else:

            return self._get_array_scalar(vptr)

    cdef inline np.ndarray _get_array_string(self, IDL_VPTR vptr):
        """
        Converts an IDL string array to an equivalent numpy array.

        :param vptr: An IDL variable pointer pointing to an IDL string array definition.
        :return: A numpy array.
        """

        cdef:
            IDL_ARRAY *idl_array
            IDL_STRING idl_string
            list strings
            int index
            np.ndarray array
            np.PyArray_Dims new_dimensions

        # obtain IDL array structure from IDL variable
        idl_array = vptr.value.arr

        # The IDL string array is unpacked to form a flat list of strings, this is then
        # converted to a numpy array before finally being reshaped to the correct dimensions.

        # convert all the IDL strings in the array to python strings
        strings = []
        for index in range(idl_array.n_elts):

            # dereference idl string pointer
            idl_string = (<IDL_STRING *> (idl_array.data + index * idl_array.elt_len))[0]
            strings.append(self._string_idl_to_py(idl_string))

        # convert to a flat numpy array
        numpy_array = np.array(strings)

        # reshape array
        new_dimensions.len = idl_array.n_dim
        new_dimensions.ptr = self._dimensions_idl_to_numpy(idl_array.n_dim, idl_array.dim)
        return np.PyArray_Newshape(numpy_array, &new_dimensions, np.NPY_CORDER)

    cdef inline np.ndarray _get_array_structure(self, IDL_VPTR vptr):
        """
        To be written.

        :param vptr:
        :return:
        """

        return NotImplemented

    cdef inline np.ndarray _get_array_scalar(self, IDL_VPTR vptr):
        """
        Converts an IDL scalar array to an equivalent numpy array.

        :param vptr: An IDL variable pointer pointing to an IDL scalar array definition.
        :return: A numpy array.
        """

        cdef:
            IDL_ARRAY *idl_array
            int num_dimensions
            np.npy_intp dimensions[IDL_MAX_ARRAY_DIM]
            np.ndarray numpy_array
            int index

        # obtain IDL array structure from IDL variable
        idl_array = vptr.value.arr

        # obtain numpy data type
        numpy_type = self._type_idl_to_numpy(vptr.type)

        # IDL defines its dimensions in the opposite order to numpy
        numpy_dimensions = self._dimensions_idl_to_numpy(idl_array.n_dim, idl_array.dim)

        # generate an empty numpy array and copy IDL array data
        numpy_array = np.PyArray_SimpleNew(idl_array.n_dim, numpy_dimensions, numpy_type)
        memcpy(numpy_array.data, <void *> idl_array.data, idl_array.arr_len)

        return numpy_array

    cdef inline dict _get_structure(self, IDL_VPTR vptr):
        """
        To be written.

        :param vptr:
        :return:
        """

        raise NotImplementedError("Not currently implemented.")

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

    cdef inline str _string_idl_to_py(self, IDL_STRING string):
        """
        Converts an IDL string object to a python string.

        :param string: An IDL string structure.
        :return: A python string object.
        """

        # The string pointer in the IDL string structure is invalid when the string length is zero.
        if string.slen == 0:

            return ""

        else:

            return string.s.decode("UTF8")

    cdef inline int _type_idl_to_numpy(self, int type):
        """
        Maps IDL type values to numpy type values.

        :param type: An IDL type value.
        :return: A numpy type value.
        """

        if type == IDL_TYP_INT: return np.NPY_INT16
        elif type == IDL_TYP_LONG: return np.NPY_INT32
        elif type == IDL_TYP_LONG64: return np.NPY_INT64
        elif type == IDL_TYP_BYTE: return np.NPY_UINT8
        elif type == IDL_TYP_UINT: return np.NPY_UINT16
        elif type == IDL_TYP_ULONG: return np.NPY_UINT32
        elif type == IDL_TYP_ULONG64: return np.NPY_UINT64
        elif type == IDL_TYP_FLOAT: return np.NPY_FLOAT32
        elif type == IDL_TYP_DOUBLE: return np.NPY_FLOAT64
        elif type == IDL_TYP_COMPLEX: return np.NPY_COMPLEX64
        elif type == IDL_TYP_DCOMPLEX: return np.NPY_COMPLEX128
        elif type == IDL_TYP_STRING: return np.NPY_STRING

        else:

            raise IDLTypeError("No matching Numpy data type defined for given IDL type.")

    cdef inline int _type_numpy_to_idl(self, int type):
        """
        Maps numpy type values to IDL type values.

        :param type: A numpy type value.
        :return: An IDL type value.
        """

        if type == np.NPY_INT8:

            raise IDLTypeError("IDL does not support signed bytes.")

        elif type == np.NPY_INT16: return IDL_TYP_INT
        elif type == np.NPY_INT32: return IDL_TYP_LONG
        elif type == np.NPY_INT64: return IDL_TYP_LONG64
        elif type == np.NPY_UINT8: return IDL_TYP_BYTE
        elif type == np.NPY_UINT16: return IDL_TYP_UINT
        elif type == np.NPY_UINT32: return IDL_TYP_ULONG
        elif type == np.NPY_UINT64: return IDL_TYP_ULONG64
        elif type == np.NPY_FLOAT32: return IDL_TYP_FLOAT
        elif type == np.NPY_FLOAT64: return IDL_TYP_DOUBLE
        elif type == np.NPY_COMPLEX64: return IDL_TYP_COMPLEX
        elif type == np.NPY_COMPLEX128: return IDL_TYP_DCOMPLEX
        elif type == np.NPY_STRING: return IDL_TYP_STRING

        else:

            raise IDLTypeError("No matching IDL data type defined for given Numpy type.")

    cdef inline np.npy_intp *_dimensions_idl_to_numpy(self, int dimension_count, IDL_ARRAY_DIM idl_dimensions):
        """
        Converts an IDL dimension array to a numpy dimension array.

        IDL defines dimensions in the opposite order to numpy.
        This method inverts the dimension order.

        :param type: An IDL dimension array.
        :return: A numpy dimension array.
        """

        cdef:
            np.npy_intp numpy_dimensions[IDL_MAX_ARRAY_DIM]
            int index

        # IDL defines its dimensions with the opposite order to numpy, invert the order
        for index in range(dimension_count):

            numpy_dimensions[index] = idl_dimensions[dimension_count - (index + 1)]

        return numpy_dimensions

    cdef inline IDL_MEMINT *_dimensions_numpy_to_idl(self, int dimension_count, np.npy_intp *numpy_dimensions):
        """
        Converts a numpy dimension array to an IDL dimension array.

        IDL defines dimensions in the opposite order to numpy.
        This method inverts the dimension order.

        :param type: A numpy dimension array.
        :return: An IDL dimension array.
        """

        cdef:
            IDL_ARRAY_DIM idl_dimensions
            int index

        # IDL defines its dimensions with the opposite order to numpy, invert the order
        for index in range(dimension_count):

            idl_dimensions[index] = numpy_dimensions[dimension_count - (index + 1)]

        return idl_dimensions


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



