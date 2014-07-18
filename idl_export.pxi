"""
Cython wrapper for required parts of idl_export.h (IDL v8.x)
"""

cdef extern from "idl_export.h":

    # data types
    ctypedef unsigned char UCHAR

    IF UNAME_MACHINE == "i686" or UNAME_MACHINE == "x86":

        # IDL data types - 32bit systems (x86)
        ctypedef short IDL_INT
        ctypedef unsigned short IDL_UINT
        ctypedef long IDL_LONG
        ctypedef unsigned long IDL_ULONG
        ctypedef long long IDL_LONG64
        ctypedef unsigned long long IDL_ULONG64

    ELSE:

        # IDL data types - 64bit systems (x86_64)
        ctypedef short IDL_INT
        ctypedef unsigned short IDL_UINT
        ctypedef int IDL_LONG
        ctypedef unsigned int IDL_ULONG
        ctypedef long long IDL_LONG64
        ctypedef unsigned long long IDL_ULONG64

    # IDL_VARIABLE type values
    DEF IDL_TYP_UNDEF = 0
    DEF IDL_TYP_BYTE = 1
    DEF IDL_TYP_INT = 2
    DEF IDL_TYP_LONG = 3
    DEF IDL_TYP_FLOAT = 4
    DEF IDL_TYP_DOUBLE = 5
    DEF IDL_TYP_COMPLEX = 6
    DEF IDL_TYP_STRING = 7
    DEF IDL_TYP_STRUCT = 8
    DEF IDL_TYP_DCOMPLEX = 9
    DEF IDL_TYP_PTR = 10
    DEF IDL_TYP_OBJREF = 11
    DEF IDL_TYP_UINT = 12
    DEF IDL_TYP_ULONG = 13
    DEF IDL_TYP_LONG64 = 14
    DEF IDL_TYP_ULONG64 = 15

    # IDL_VARIABLE flags
    DEF IDL_V_CONST = 1
    DEF DL_V_CONST = 2
    DEF IDL_V_ARR = 4
    DEF IDL_V_FILE = 8
    DEF IDL_V_DYNAMIC = 16
    DEF IDL_V_STRUCT = 32
    DEF IDL_V_NOT_SCALAR = (IDL_V_ARR | IDL_V_FILE | IDL_V_STRUCT)

    # init defines
    DEF IDL_INIT_QUIET = 64
    DEF IDL_INIT_NOCMDLINE = (1 << 12)

    # data structures
    ctypedef int IDL_INIT_DATA_OPTIONS_T

    ctypedef struct IDL_CLARGS:

        int argc
        char **argv

    ctypedef struct IDL_INIT_DATA:

        IDL_INIT_DATA_OPTIONS_T options
        IDL_CLARGS clargs
        void *hwnd

    # functions
    int IDL_Initialize(IDL_INIT_DATA *init_data) nogil

    int IDL_Cleanup(int just_cleanup) nogil

    int IDL_ExecuteStr(char *cmd) nogil



    # STUFF FOR REFERENCE

    # Array definitions

    # Maximum # of array dimensions
    # DEF IDL_MAX_ARRAY_DIM = 8

    # ctypedef int IDL_ARRAY_DIM[IDL_MAX_ARRAY_DIM]
    # ctypedef struct IDL_ARRAY:
    #     int elt_len                 # Length of element in char units
    #     int arr_len		            # Length of entire array (char)
    #     int n_elts		            # total # of elements
    #     char *data			        # ^ to beginning of array data
    #     char n_dim			        # # of dimensions used by array
    #     char flags			        # Array block flags
    #     short file_unit		        # # of assoc file if file var
    #     IDL_ARRAY_DIM dim		    # dimensions
    #     #IDL_ARRAY_FREE_CB free_cb	# Free callback
    #     #IDL_FILEINT offset		    # Offset to base of data for file var
    #     #IDL_MEMINT data_guard	    # Guard longword

    # String definitions
    # ctypedef struct IDL_STRING:
    #     int slen
    #     char *s

    # Structure definitions
    # ctypedef struct _idl_structure:
    #     int ntags

    # ctypedef _idl_structure *IDL_StructDefPtr

    # ctypedef struct IDL_SREF:
    #     IDL_ARRAY *arr
    #     _idl_structure *sdef

    # int IDL_StructNumTags(IDL_StructDefPtr sdef)

    # ctypedef union IDL_ALLTYPES:
    #     UCHAR c
    #     IDL_INT i
    #     IDL_LONG l
    #     float f
    #     double d
    #
    #     IDL_STRING str
    #     IDL_ARRAY *arr
    #     IDL_SREF s

    # ctypedef struct IDL_VARIABLE:
    #     char type
    #     char flags
    #     IDL_ALLTYPES value

    # ctypedef IDL_VARIABLE *IDL_VPTR

    # int IDL_Initialize(IDL_INIT_DATA *init_data)
    #
    # int IDL_Cleanup(int just_cleanup)
    #
    # int IDL_ExecuteStr(char* cmd)
    #
    # void IDL_StrStore(IDL_STRING *s, char *fs)
    #
    # IDL_VPTR IDL_ImportArray(int n_dim, int dim[], int type, UCHAR *data, void* free_cb)
    #
    # IDL_VPTR IDL_Gettmp()
    #
    # int IDL_ExecuteStr(CLIENT *pClient, char * pszCommand)
