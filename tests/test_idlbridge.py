# TODO: add license

from unittest import TestCase
import idlbridge as idl

class test_idlbridge(TestCase):

    def test_execute(self):

        # This fuction is difficult to test as it requires the other functions.
        # Here we just test the command executes without a failure.
        idl.execute("d = 128")

    def test_get_scalar(self):

        # unsigned byte
        idl.execute("test_byte = byte(15)")
        self.assertEqual(15, idl.get("test_byte"), "Failed to get unsigned byte.")

        # signed integer
        idl.execute("test_int = fix(-3485)")
        self.assertEqual(-3485, idl.get("test_int"), "Failed to get signed integer.")

        # unsigned integer
        idl.execute("test_uint = uint(36640)")
        self.assertEqual(36640, idl.get("test_uint"), "Failed to get unsigned integer.")

        # signed long
        idl.execute("test_long = long(-1633485)")
        self.assertEqual(-1633485, idl.get("test_long"), "Failed to get signed long.")

        # unsigned long
        idl.execute("test_ulong = ulong(3267987588)")
        self.assertEqual(3267987588, idl.get("test_ulong"), "Failed to get unsigned long.")

        # signed long64
        idl.execute("test_long64 = long64(-16000000096790)")
        self.assertEqual(-16000000096790, idl.get("test_long64"), "Failed to get signed long64.")

        # unsigned long64
        idl.execute("test_ulong64 = ulong64(18446728073709454827)")
        self.assertEqual(18446728073709454827, idl.get("test_ulong64"), "Failed to get unsigned long64.")

        # float
        idl.execute("test_float = float(-2.0)")
        self.assertEqual(-2.0, idl.get("test_float"), "Failed to get float.")

        # double
        idl.execute("test_double = double(1.0)")
        self.assertEqual(1.0, idl.get("test_double"), "Failed to get double.")

        # float complex
        idl.execute("test_fcomplex = complex(1.0, 2.0)")
        self.assertEqual(complex(1.0, 2.0), idl.get("test_fcomplex"), "Failed to get float complex.")

        # double complex
        idl.execute("test_dcomplex = dcomplex(1.0, 2.0)")
        self.assertEqual(complex(1.0, 2.0), idl.get("test_dcomplex"), "Failed to get double complex.")

    def test_get_array(self):

        pass

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



