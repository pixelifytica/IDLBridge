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

    def test_execute(self):

        # This function is difficult to test as it requires the other functions.
        # Here we just test the command executes without a failure.
        # Basically if this fails evey test is going to fail!
        idl.execute("d = 128")

    # TODO: add test for null -> None

    def test_get_scalar_byte(self):

        # unsigned byte
        idl.execute("test_byte = byte(15)")
        self.assertEqual(15, idl.get("test_byte"), "Failed to get unsigned byte.")

    def test_get_scalar_int(self):

        # signed integer
        idl.execute("test_int = fix(-3485)")
        self.assertEqual(-3485, idl.get("test_int"), "Failed to get signed integer.")

    def test_get_scalar_uint(self):

        # unsigned integer
        idl.execute("test_uint = uint(36640)")
        self.assertEqual(36640, idl.get("test_uint"), "Failed to get unsigned integer.")

    def test_get_scalar_long(self):

        # signed long
        idl.execute("test_long = long(-1633485)")
        self.assertEqual(-1633485, idl.get("test_long"), "Failed to get signed long.")

    def test_get_scalar_ulong(self):

        # unsigned long
        idl.execute("test_ulong = ulong(3267987588)")
        self.assertEqual(3267987588, idl.get("test_ulong"), "Failed to get unsigned long.")

    def test_get_scalar_long64(self):

        # signed long64
        idl.execute("test_long64 = long64(-16000000096790)")
        self.assertEqual(-16000000096790, idl.get("test_long64"), "Failed to get signed long64.")

    def test_get_scalar_ulong64(self):

        # unsigned long64
        idl.execute("test_ulong64 = ulong64(18446728073709454827)")
        self.assertEqual(18446728073709454827, idl.get("test_ulong64"), "Failed to get unsigned long64.")

    def test_get_scalar_float(self):

        # float
        idl.execute("test_float = float(-2.0)")
        self.assertEqual(-2.0, idl.get("test_float"), "Failed to get float.")

    def test_get_scalar_double(self):

        # double
        idl.execute("test_double = double(1.0)")
        self.assertEqual(1.0, idl.get("test_double"), "Failed to get double.")

    def test_get_scalar_complex_float(self):

        # float complex
        idl.execute("test_fcomplex = complex(1.0, 2.0)")
        self.assertEqual(complex(1.0, 2.0), idl.get("test_fcomplex"), "Failed to get float complex.")

    def test_get_scalar_complex_double(self):

        # double complex
        idl.execute("test_dcomplex = dcomplex(1.0, 2.0)")
        self.assertEqual(complex(1.0, 2.0), idl.get("test_dcomplex"), "Failed to get double complex.")

    def test_get_array_byte(self):

        # 1D array, byte
        idl.execute("test_array_1d_byte = byte([1,2,3,4,5])")
        v = idl.get("test_array_1d_byte")
        r = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
        self.assertArrayEqual(v, r, "Failed to get byte array.")

    def test_get_array_int(self):

        # 1D array, int
        idl.execute("test_array_1d_int = fix([1,2,3,4,5])")
        v = idl.get("test_array_1d_int")
        r = np.array([1, 2, 3, 4, 5], dtype=np.int16)
        self.assertArrayEqual(v, r, "Failed to get int array.")

    def test_get_array_uint(self):

        # 1D array, uint
        idl.execute("test_array_1d_uint = uint([1,2,3,4,5])")
        v = idl.get("test_array_1d_uint")
        r = np.array([1, 2, 3, 4, 5], dtype=np.uint16)
        self.assertArrayEqual(v, r, "Failed to get uint array.")

    def test_get_array_long(self):

        # 1D array, long
        idl.execute("test_array_1d_long = long([1,2,3,4,5])")
        v = idl.get("test_array_1d_long")
        r = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        self.assertArrayEqual(v, r, "Failed to get long array.")

    def test_get_array_ulong(self):

        # 1D array, ulong
        idl.execute("test_array_1d_ulong = ulong([1,2,3,4,5])")
        v = idl.get("test_array_1d_ulong")
        r = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
        self.assertArrayEqual(v, r, "Failed to get ulong array.")

    def test_get_array_long64(self):

        # 1D array, long64
        idl.execute("test_array_1d_long64 = long64([1,2,3,4,5])")
        v = idl.get("test_array_1d_long64")
        r = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        self.assertArrayEqual(v, r, "Failed to get long64 array.")

    def test_get_array_ulong64(self):

        # 1D array, ulong64
        idl.execute("test_array_1d_ulong64 = ulong64([1,2,3,4,5])")
        v = idl.get("test_array_1d_ulong64")
        r = np.array([1, 2, 3, 4, 5], dtype=np.uint64)
        self.assertArrayEqual(v, r, "Failed to get ulong64 array.")

    def test_get_array_float(self):

        # 1D array, float
        idl.execute("test_array_1d_float = float([1,2,3,4,5])")
        v = idl.get("test_array_1d_float")
        r = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        self.assertArrayEqual(v, r, "Failed to get float array.")

    def test_get_array_double(self):

        # 1D array, double
        idl.execute("test_array_1d_double = double([1,2,3,4,5])")
        v = idl.get("test_array_1d_double")
        r = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        self.assertArrayEqual(v, r, "Failed to get double array.")

    def test_get_array_complex_float(self):

        # 1D array, complex float
        idl.execute("test_array_1d_complex = complex([1,2,3,4,5], [6,7,8,9,10])")
        v = idl.get("test_array_1d_complex")
        r = np.array([1+6j, 2+7j, 3+8j, 4+9j, 5+10j], dtype=np.complex64)
        self.assertArrayEqual(v, r, "Failed to get complex float array.")

    def test_get_array_complex_double(self):

        # 1D array, complex double
        idl.execute("test_array_1d_dcomplex = dcomplex([1,2,3,4,5], [6,7,8,9,10])")
        v = idl.get("test_array_1d_dcomplex")
        r = np.array([1+6j, 2+7j, 3+8j, 4+9j, 5+10j], dtype=np.complex128)
        self.assertArrayEqual(v, r, "Failed to get complex double array.")

    def test_get_array_string(self):

        # 1D array, string
        idl.execute("test_array_1d_string = [\"dog\", \"\", \"cat\", \"fish\"]")
        v = idl.get("test_array_1d_string")
        r = np.array(["dog", "", "cat", "fish"])
        self.assertArrayEqual(v, r, "Failed to get string array.")

    def test_get_array_multidimensional(self):

        # ND array
        idl.execute("test_array_nd = indgen(8,7,6,5,4,3,2,1, /long)")
        self.assertTrue(False, "Test needs to be written.")

    def test_get_structure(self):

        pass

    def test_get_string(self):

        # null string
        idl.execute("test_string_null = \"\"")
        self.assertEqual("", idl.get("test_string_null"), "Failed to get null string.")

        # normal string
        idl.execute("test_string_normal = \"this is a test string\"")
        self.assertEqual("this is a test string", idl.get("test_string_normal"), "Failed to get normal string.")

    def test_get_pointer(self):

        pass

    def test_put_scalar(self):

        pass

    def test_put_array(self):

        pass

    def test_put_structure(self):

        pass

    def test_delete(self):

        pass

    def test_function(self):

        pass

    def test_procedure(self):

        pass



