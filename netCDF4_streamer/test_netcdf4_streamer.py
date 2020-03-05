import unittest
import tempfile
import os
import numpy as np
import netCDF4

from netCDF4_streamer import NetCDF4Streamer, NetCDF4StreamerVariable


class TestNetCDF4Streamer(unittest.TestCase):
    """Base class that create some simple file."""
    def setUp(self) -> None:
        """Create the file and streamed variable."""
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
        """Delete created variable."""
        os.remove(os.path.join(self.output_dir, "test.nc"))


class TestNetCDF4StreamerSingleValue(TestNetCDF4Streamer):
    def test_single_value(self):
        """Write the single value(s) to the variable in a stream."""
        for i in range(1, 20 + 1):
            data_set = np.ones((500, 600)) * i
            self.variable.streamNumpyData(data_set, single_entity=True)
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

    def test_single_value_permutate(self):
        """Test the permutation"""
        self.variable.axes_order = ("d3", "d1", "d2")
        for i in range(1, 20 + 1):
            # The order is now swapped (transposed)
            data_set = np.ones((600, 500)) * i
            self.variable.streamNumpyData(data_set, single_entity=True)
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
        """Write the whole blob of values to the variable."""
        stream = np.ones((500, 20, 600))
        for i in range(1, 20 + 1):
            stream[:, i - 1, :] *= i
        self.variable.streamNumpyData(stream, single_entity=False)
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

    def test_values_set_permutate(self):
        """Write the whole blob of values to the variable."""
        self.variable.axes_order = ("d3", "d1", "d2")

        stream = np.ones((500, 20, 600))
        for i in range(1, 20 + 1):
            stream[:, i - 1, :] *= i
        stream = np.moveaxis(stream, [0, 1, 2], [1, 2, 0])

        self.variable.streamNumpyData(stream, single_entity=False)
        self.variable.flush()
        self.file.close()

        # Test the file
        nc = netCDF4.Dataset(os.path.join(self.output_dir, "test.nc"))
        computed = np.array(nc['var'][:])

        expected = np.ones((500, 20, 600))
        for i in range(1, 20 + 1):
            expected[:, i - 1, :] *= i

        self.assertTrue(np.allclose(computed, expected))
        nc.close()


class TestNetCDF4StreamerValuesSetAndSingle(TestNetCDF4Streamer):
    def test_values_set_and_single_value(self):
        """Write both blob and single value to the variable."""
        stream = np.ones((500, 20, 600))
        self.variable.streamNumpyData(stream[:, 0, :].reshape(500, 600),
                                      single_entity=True)

        for i in range(2, 20 + 1):
            stream[:, i - 1, :] *= i
        self.variable.streamNumpyData(stream[:, 1:, :],
                                      single_entity=False)
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

    def test_values_set_and_single_value_permutation(self):
        self.variable.axes_order = ("d3", "d1", "d2")
        stream = np.ones((500, 20, 600))

        for i in range(2, 20 + 1):
            stream[:, i - 1, :] *= i

        self.variable.streamNumpyData(stream[:, 0, :].transpose(),
                                      single_entity=True)

        stream = np.moveaxis(stream, [0, 1, 2], [1, 2, 0])
        self.variable.streamNumpyData(stream[:, :, 1:],
                                      single_entity=False)
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


class TestNetCDF4StreamerYielding(unittest.TestCase):
    def setUp(self) -> None:
        """Set-up the NetCDF4 variable for reading"""
        # Create a file:
        self.output_dir = tempfile.mkdtemp()
        file_pointer = netCDF4.Dataset(
            os.path.join(self.output_dir, "testRead.nc"),
            "w"
        )

        # Create dimensions
        file_pointer.createDimension("d1", 50)
        file_pointer.createDimension("d2", 60)
        file_pointer.createDimension("d3", 70)

        # Create variable
        var = file_pointer.createVariable("test", "f8", ("d1", "d2", "d3"),
                                          fill_value=0)
        # Fill variable with values
        self.values = np.random.random((50, 60, 70))
        var[:, :, :] = self.values

        # Save and close
        file_pointer.close()

        # Open a testing file for reading
        self.nc = NetCDF4Streamer(
            os.path.join(self.output_dir, "testRead.nc"),
            "r"
        )
        self.var = self.nc.openStreamerVariable("test",
                                                chunk_dimension="d2",
                                                chunk_size_mb=1)

    def tearDown(self) -> None:
        """Delete created variable."""
        os.remove(os.path.join(self.output_dir, "testRead.nc"))


class TestNetCDF4StreamerYieldingSingleValue(TestNetCDF4StreamerYielding):
    def test_single_value(self):
        """Test the reading by the single value"""
        # Read by dimension 'd2'
        idx = 0
        for line in self.var.yieldNumpyData(single_entity=True):
            self.assertTrue(np.allclose(self.values[:, idx, :], line))
            idx += 1

    def test_single_value_permutation(self):
        # Read by dimension 'd2'
        idx = 0
        self.var.axes_order = ("d3", "d1", "d2")
        for line in self.var.yieldNumpyData(single_entity=True):
            self.assertTrue(np.allclose(self.values[:, idx, :].transpose(),
                                        line))
            idx += 1


class TestNetCDF4StreamerYieldingValuesSet(TestNetCDF4StreamerYielding):
    def test_values_set(self):
        """Test of the reading by the blob."""
        # Read by dimension 'd2'
        chunk_size = self.var.chunk_size
        idx_start = 0
        idx_end = 0
        idx = 0
        for set_values in self.var.yieldNumpyData(single_entity=False):
            idx_start = idx_end
            if idx == self.values.shape[1] - 1:
                idx_end += self.values.shape[1] % chunk_size
            else:
                idx_end += chunk_size
            self.assertTrue(
                np.allclose(
                    self.values[:, idx_start:idx_end, :], set_values
                )
            )

    def test_values_set_permutation(self):
        # Read by dimension 'd2'
        self.var.axes_order = ("d3", "d1", "d2")
        chunk_size = self.var.chunk_size
        idx_start = 0
        idx_end = 0
        idx = 0
        for set_values in self.var.yieldNumpyData(single_entity=False):
            idx_start = idx_end
            if idx == self.values.shape[1] - 1:
                idx_end += self.values.shape[1] % chunk_size
            else:
                idx_end += chunk_size
            self.assertTrue(
                np.allclose(
                    np.moveaxis(self.values[:, idx_start:idx_end, :],
                                [0, 1, 2], [1, 2, 0]),
                    set_values
                )
            )


if __name__ == "__main__":
    unittest.main()
