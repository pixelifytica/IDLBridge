# TODO: add license

from unittest import TestCase
import idlbridge as idl
import numpy as np


class TestIDLBridge(TestCase):

    def assertArrayEqual(self, first, second, message):

        # check data types are the same
        self.assertEqual(first.dtype, second.dtype, message + " Data types are different.")

        # check all elements are equal
        self.assertTrue((first == second).any(), message + " Elements are not the same.")

#    def assertDictEqualExtended(self, first, second, message):

        # # the basic dict() == dict() fails on numpy arrays so we need to build a special test
        #
        # # check keys are equal
        # self.assertEqual(first.keys(), second.keys(), message)
        #
        # # check each item
        # for key, first_item in first.items():
        #
        #     second_item = second[key]
        #
        #     if isinstance(first_item, dict):
        #
        #         self.assertIsInstance(second_item, dict)
        #         self.assertDictEqual(first_item, second_item, message)
        #
        #     elif isinstance(first_item, np.ndarray):



    def test_execute(self):

        # This function is difficult to test as it requires the other functions.
        # Here we just test the command executes without a failure.
        # Basically if this fails evey test is going to fail!
        idl.execute("d = 128")

    # TODO: add test for null -> None

    def test_get_scalar_byte(self):

        # unsigned byte
        idl.execute("test_get_byte = byte(15)")
        self.assertEqual(15, idl.get("test_get_byte"), "Failed to get unsigned byte.")

    def test_get_scalar_int(self):

        # signed integer
        idl.execute("test_get_int = fix(-3485)")
        self.assertEqual(-3485, idl.get("test_get_int"), "Failed to get signed integer.")

    def test_get_scalar_uint(self):

        # unsigned integer
        idl.execute("test_get_uint = uint(36640)")
        self.assertEqual(36640, idl.get("test_get_uint"), "Failed to get unsigned integer.")

    def test_get_scalar_long(self):

        # signed long
        idl.execute("test_get_long = long(-1633485)")
        self.assertEqual(-1633485, idl.get("test_get_long"), "Failed to get signed long.")

    def test_get_scalar_ulong(self):

        # unsigned long
        idl.execute("test_get_ulong = ulong(3267987588)")
        self.assertEqual(3267987588, idl.get("test_get_ulong"), "Failed to get unsigned long.")

    def test_get_scalar_long64(self):

        # signed long64
        idl.execute("test_get_long64 = long64(-16000000096790)")
        self.assertEqual(-16000000096790, idl.get("test_get_long64"), "Failed to get signed long64.")

    def test_get_scalar_ulong64(self):

        # unsigned long64
        idl.execute("test_get_ulong64 = ulong64(18446728073709454827)")
        self.assertEqual(18446728073709454827, idl.get("test_get_ulong64"), "Failed to get unsigned long64.")

    def test_get_scalar_float(self):

        # float
        idl.execute("test_get_float = float(-2.0)")
        self.assertEqual(-2.0, idl.get("test_get_float"), "Failed to get float.")

    def test_get_scalar_double(self):

        # double
        idl.execute("test_get_double = double(1.0)")
        self.assertEqual(1.0, idl.get("test_get_double"), "Failed to get double.")

    def test_get_scalar_complex_float(self):

        # float complex
        idl.execute("test_get_fcomplex = complex(1.0, 2.0)")
        self.assertEqual(complex(1.0, 2.0), idl.get("test_get_fcomplex"), "Failed to get float complex.")

    def test_get_scalar_complex_double(self):

        # double complex
        idl.execute("test_get_dcomplex = dcomplex(1.0, 2.0)")
        self.assertEqual(complex(1.0, 2.0), idl.get("test_get_dcomplex"), "Failed to get double complex.")

    def test_get_array_byte(self):

        # 1D array, byte
        idl.execute("test_get_array_1d_byte = byte([1,2,3,4,5])")
        v = idl.get("test_get_array_1d_byte")
        r = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
        self.assertArrayEqual(v, r, "Failed to get byte array.")

    def test_get_array_int(self):

        # 1D array, int
        idl.execute("test_get_array_1d_int = fix([1,2,3,4,5])")
        v = idl.get("test_get_array_1d_int")
        r = np.array([1, 2, 3, 4, 5], dtype=np.int16)
        self.assertArrayEqual(v, r, "Failed to get int array.")

    def test_get_array_uint(self):

        # 1D array, uint
        idl.execute("test_get_array_1d_uint = uint([1,2,3,4,5])")
        v = idl.get("test_get_array_1d_uint")
        r = np.array([1, 2, 3, 4, 5], dtype=np.uint16)
        self.assertArrayEqual(v, r, "Failed to get uint array.")

    def test_get_array_long(self):

        # 1D array, long
        idl.execute("test_get_array_1d_long = long([1,2,3,4,5])")
        v = idl.get("test_get_array_1d_long")
        r = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        self.assertArrayEqual(v, r, "Failed to get long array.")

    def test_get_array_ulong(self):

        # 1D array, ulong
        idl.execute("test_get_array_1d_ulong = ulong([1,2,3,4,5])")
        v = idl.get("test_get_array_1d_ulong")
        r = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
        self.assertArrayEqual(v, r, "Failed to get ulong array.")

    def test_get_array_long64(self):

        # 1D array, long64
        idl.execute("test_get_array_1d_long64 = long64([1,2,3,4,5])")
        v = idl.get("test_get_array_1d_long64")
        r = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        self.assertArrayEqual(v, r, "Failed to get long64 array.")

    def test_get_array_ulong64(self):

        # 1D array, ulong64
        idl.execute("test_get_array_1d_ulong64 = ulong64([1,2,3,4,5])")
        v = idl.get("test_get_array_1d_ulong64")
        r = np.array([1, 2, 3, 4, 5], dtype=np.uint64)
        self.assertArrayEqual(v, r, "Failed to get ulong64 array.")

    def test_get_array_float(self):

        # 1D array, float
        idl.execute("test_get_array_1d_float = float([1,2,3,4,5])")
        v = idl.get("test_get_array_1d_float")
        r = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        self.assertArrayEqual(v, r, "Failed to get float array.")

    def test_get_array_double(self):

        # 1D array, double
        idl.execute("test_get_array_1d_double = double([1,2,3,4,5])")
        v = idl.get("test_get_array_1d_double")
        r = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        self.assertArrayEqual(v, r, "Failed to get double array.")

    def test_get_array_complex_float(self):

        # 1D array, complex float
        idl.execute("test_get_array_1d_complex = complex([1,2,3,4,5], [6,7,8,9,10])")
        v = idl.get("test_get_array_1d_complex")
        r = np.array([1+6j, 2+7j, 3+8j, 4+9j, 5+10j], dtype=np.complex64)
        self.assertArrayEqual(v, r, "Failed to get complex float array.")

    def test_get_array_complex_double(self):

        # 1D array, complex double
        idl.execute("test_get_array_1d_dcomplex = dcomplex([1,2,3,4,5], [6,7,8,9,10])")
        v = idl.get("test_get_array_1d_dcomplex")
        r = np.array([1+6j, 2+7j, 3+8j, 4+9j, 5+10j], dtype=np.complex128)
        self.assertArrayEqual(v, r, "Failed to get complex double array.")

    def test_get_array_string(self):

        # 1D array, string
        idl.execute("test_get_array_1d_string = [\"dog\", \"\", \"cat\", \"fish\"]")
        v = idl.get("test_get_array_1d_string")
        r = np.array(["dog", "", "cat", "fish"])
        self.assertArrayEqual(v, r, "Failed to get string array.")

    def test_get_array_multidimensional(self):

        # ND array
        idl.execute("test_get_array_nd = indgen(8,7,6,5,4,3,2,1,/long)")
        v = np.arange(8*7*6*5*4*3*2*1, dtype=np.int32).reshape((1, 2, 3, 4, 5, 6, 7, 8))
        r = idl.get("test_get_array_nd")
        self.assertArrayEqual(v, r, "Failed to get multidimensional array.")

    def test_get_structure_basic(self):

        idl.execute("test_get_structure_basic = {a:byte(1), b:fix(2), "
                    + "c:uint(3), d:long(4), e:ulong(5), f:long64(6), "
                    + "g:ulong64(7), h:float(1.0), i:double(2.0), "
                    + "j:complex(1.0, 2.0), k:dcomplex(1.0, 2.0),"
                    + "l:\"test\"}")

        idl.execute("print, test_get_structure_basic")
        r = idl.get("test_get_structure_basic")

        v = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6,
             "g": 7, "h": 1.0, "i": 2.0, "j": 1+2j, "k": 1+2j,
             "l": "test"}

        self.assertEqual(v, r, "Failed to get basic structure.")

    def test_get_structure_nested(self):

        idl.execute("test_get_structure_nested = {s:{a:1, b:2}, a:[1, 2, 3, 4, 5]}")
        r = idl.get("test_get_structure_nested")

        v = {"s": {"a": 1, "b": 2}, "a": np.array([1, 2, 3, 4, 5], dtype=np.int16)}

        self.assertEqual(v, r, "Failed to get nested structure.")

    def test_get_string(self):

        # null string
        idl.execute("test_get_string_null = \"\"")
        self.assertEqual("", idl.get("test_get_string_null"), "Failed to get null string.")

        # normal string
        idl.execute("test_get_string_normal = \"this is a test string\"")
        self.assertEqual("this is a test string", idl.get("test_get_string_normal"), "Failed to get normal string.")

    def test_get_pointer(self):

        pass

    def test_put_scalar_int(self):

        idl.put("test_put_int", -574)
        self.assertEqual(-574, idl.get("test_put_int"), "Failed to put int.")

    def test_put_scalar_float(self):

        idl.put("test_put_float", 2.0)
        self.assertEqual(2.0, idl.get("test_put_float"), "Failed to put float.")

    def test_put_scalar_complex(self):

        idl.put("test_put_complex", 2.0+1.0j)
        self.assertEqual(2.0+1.0j, idl.get("test_put_complex"), "Failed to put complex.")

    def test_put_string(self):

        idl.put("test_put_string", "test")
        self.assertEqual("test", idl.get("test_put_string"), "Failed to put string.")

    def test_put_array_uint8(self):

        pass

    def test_put_array_int16(self):

        pass

    def test_put_array_uint16(self):

        pass

    def test_put_array_int32(self):

        pass

    def test_put_array_uint32(self):

        pass

    def test_put_array_int64(self):

        pass

    def test_put_array_uint64(self):

        pass

    def test_put_array_float32(self):

        pass

    def test_put_array_float64(self):

        pass

    def test_put_array_complex64(self):

        pass

    def test_put_array_complex128(self):

        pass

    def test_put_array_string(self):

        pass

    def test_put_array_multidimensional(self):

        pass

    def test_put_structure_basic(self):

        pass

    def test_put_structure_nested(self):

        pass

    def test_delete(self):

        pass

    def test_function(self):

        pass

    def test_procedure(self):

        pass



