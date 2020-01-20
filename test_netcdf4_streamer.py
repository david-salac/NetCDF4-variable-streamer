import unittest
import tempfile
import os
import numpy as np
import netCDF4

from NetCDF4_streamer import NetCDF4Streamer, NetCDF4StreamerVariable


class TestNetCDF4Streamer(unittest.TestCase):
    def setUp(self) -> None:
        # Create file:
        self.output_dir = tempfile.mkdtemp()
        self.file = NetCDF4Streamer(os.path.join(self.output_dir, "test.nc"),
                                    mode="w")
        # Create dimensions inside file:
        self.file.createDimension("d1", 500)
        self.file.createDimension("d2", 20)
        self.file.createDimension("d3", 600)

        # Create variable:
        self.variable: NetCDF4StreamerVariable = \
            self.file.createStreamerVariable("var", "f8", ("d1", "d2", "d3"),
                                             chunk_dimension="d2",
                                             chunk_size_mb=3)

    def tearDown(self) -> None:
        os.remove(os.path.join(self.output_dir, "test.nc"))


class TestNetCDF4StreamerSingleValue(TestNetCDF4Streamer):
    def test_single_value(self):
        for i in range(1, 20 + 1):
            data_set = np.ones((500, 600)) * i
            self.variable.streamNumpyData(data_set, singe_entity=True)
        self.variable.flush()
        self.file.close()

        # Test the file
        nc = netCDF4.Dataset(os.path.join(self.output_dir, "test.nc"))
        computed = nc['var'][:]
        expected = np.ones((500, 20, 600))
        for i in range(1, 20 + 1):
            expected[:, i - 1, :] *= i

        self.assertTrue(np.allclose(computed, expected))
        nc.close()


class TestNetCDF4StreamerValuesSet(TestNetCDF4Streamer):
    def test_values_set(self):
        stream = np.ones((500, 20, 600))
        for i in range(1, 20 + 1):
            stream[:, i - 1, :] *= i
        self.variable.streamNumpyData(stream, singe_entity=False)
        self.variable.flush()
        self.file.close()

        # Test the file
        nc = netCDF4.Dataset(os.path.join(self.output_dir, "test.nc"))
        computed = nc['var'][:]
        expected = np.ones((500, 20, 600))
        for i in range(1, 20 + 1):
            expected[:, i - 1, :] *= i

        self.assertTrue(np.allclose(computed, expected))
        nc.close()


class TestNetCDF4StreamerValuesSetAndSingle(TestNetCDF4Streamer):
    def test_values_set_and_single_value(self):
        stream = np.ones((500, 20, 600))
        self.variable.streamNumpyData(stream[:, 0, :].reshape(500, 600),
                                      singe_entity=True)

        for i in range(2, 20 + 1):
            stream[:, i - 1, :] *= i
        self.variable.streamNumpyData(stream[:, 1:, :],
                                      singe_entity=False)
        self.variable.flush()
        self.file.close()

        # Test the file
        nc = netCDF4.Dataset(os.path.join(self.output_dir, "test.nc"))
        computed = nc['var'][:]
        expected = np.ones((500, 20, 600))
        for i in range(1, 20 + 1):
            expected[:, i - 1, :] *= i

        self.assertTrue(np.allclose(computed, expected))
        nc.close()
