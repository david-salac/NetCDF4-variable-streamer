import unittest
import tempfile
import os
import numpy as np
import netCDF4

from netCDF4_enhancement import NetCDF4Streamer, NetCDF4StreamerVariable


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
                    self.values[:, idx_start:idx_end, :].transpose((2, 0, 1)),
                    set_values
                )
            )


class TestReadingWritingDimensionOrderFree(unittest.TestCase):
    def setUp(self) -> None:
        """Set-up the NetCDF4 variable for reading"""
        # Create a file:
        self.output_dir = tempfile.mkdtemp()
        file_pointer = netCDF4.Dataset(
            os.path.join(self.output_dir, "testRead.nc"),
            "w"
        )

        # Create dimensions
        self.shape = (5, 6, 7, 8)
        self.dims = ("d1", "d2", "d3", 'd4')
        for i in range(len(self.shape)):
            file_pointer.createDimension(self.dims[i], self.shape[i])
        self.value = np.random.random(self.shape) * 100

        # Create variable
        file_pointer.createVariable(
            "test", "f8", self.dims
        )
        file_pointer['test'][:] = self.value
        file_pointer.close()

        # Open a testing file for reading
        self.nc = NetCDF4Streamer(
            os.path.join(self.output_dir, "testRead.nc"),
            "a"
        )
        self.var = self.nc.openStreamerVariable("test")

    def _get_val_at(self, d1: int, d2: int, d3: int, d4: int):
        return self.value[d1, d2, d3, d4]

    def tearDown(self) -> None:
        """Delete created variable."""
        os.remove(os.path.join(self.output_dir, "testRead.nc"))

    def test_read(self):
        """Test the reading."""
        # A) Fill variable with values
        axes_order = [self.dims[1], self.dims[3],
                      self.dims[2], self.dims[0]]
        expected = self.value.transpose((1, 3, 2, 0))
        self.var.axes_order = axes_order

        for d2 in range(self.shape[1]):
            computed = self.var[d2, :, :, :]
            exp_val = expected[d2, :, :, :]
            self.assertTupleEqual(computed.shape, exp_val.shape)
            self.assertTrue(np.allclose(exp_val, computed))

        # B) Fill variable with values
        axes_order = [self.dims[1], self.dims[3],
                      self.dims[2], self.dims[0]]
        expected = self.value.transpose((1, 3, 2, 0))
        self.var.axes_order = axes_order

        for d3 in range(self.shape[2]):
            computed = self.var[:, :, d3, :]
            exp_val = expected[:, :, d3, :]
            self.assertTupleEqual(computed.shape, exp_val.shape)
            self.assertTrue(np.allclose(exp_val, computed))

        # C) Fill variable with values
        axes_order = [self.dims[2], self.dims[3],
                      self.dims[1], self.dims[0]]
        expected = self.value.transpose((2, 3, 1, 0))
        self.var.axes_order = axes_order

        for d1 in range(self.shape[0]):
            for d3 in range(self.shape[2]):
                computed = self.var[d3, :, d1, :]
                exp_val = expected[d3, :, d1, :]
                self.assertTupleEqual(computed.shape, exp_val.shape)
                self.assertTrue(np.allclose(exp_val, computed))

        # D) Fill variable with values
        axes_order = [self.dims[2], self.dims[3],
                      self.dims[1], self.dims[0]]
        expected = self.value.transpose((2, 3, 1, 0))
        self.var.axes_order = axes_order

        for d1 in range(self.shape[0]):
            for d3 in range(self.shape[2]):
                computed = self.var[:, :, :, :]
                exp_val = expected[:, :, :, :]
                self.assertTupleEqual(computed.shape, exp_val.shape)
                self.assertTrue(np.allclose(exp_val, computed))

        # E) Test single value
        axes_order = [self.dims[2], self.dims[3],
                      self.dims[1], self.dims[0]]
        expected = self.value.transpose((2, 3, 1, 0))
        self.var.axes_order = axes_order
        self.assertAlmostEqual(self.var[1, 1, 2, 1], expected[1, 1, 2, 1])
        self.assertTrue(isinstance(self.var[1, 1, 2, 1], float))

    def test_sel(self):
        """Test the sel method using kwarg values."""
        # A) Fill variable with values
        axes_order = [self.dims[1], self.dims[3],
                      self.dims[2], self.dims[0]]
        expected = self.value.transpose((1, 3, 2, 0))
        self.var.axes_order = axes_order

        for d1 in range(self.shape[0]):
            for d2 in range(self.shape[1]):
                for d3 in range(self.shape[2]):
                    for d4 in range(self.shape[3]):
                        self.assertAlmostEqual(self.var.sel(
                            d1=d1, d2=d2, d3=d3, d4=d4),
                            expected[d2, d4, d3, d1]
                        )

    def test_write(self):
        """Test the writing."""
        # A) Fill variable with single values
        axes_order = [self.dims[1], self.dims[3],
                      self.dims[2], self.dims[0]]
        self.var.axes_order = axes_order
        self.var[1, 2, 3, 4] = 3
        self.assertEqual(self.var.variable[4, 1, 3, 2], 3)
        self.assertEqual(self.var[1, 2, 3, 4], 3)
        self.var[3, 1, 5, 0] = 3
        self.assertEqual(self.var.variable[4, 1, 3, 2], 3)
        self.assertEqual(self.var[1, 2, 3, 4], 3)

        # B) Fill variable with values
        axes_order = [self.dims[1], self.dims[3],
                      self.dims[2], self.dims[0]]
        expected = (np.random.random(self.shape) * 100).transpose((1, 3, 2, 0))
        self.var.axes_order = axes_order

        for d3 in range(self.shape[2]):
            exp_val = expected[:, :, d3, :]
            self.var[:, :, d3, :] = exp_val
            computed = self.var[:, :, d3, :]

            self.assertTupleEqual(computed.shape, exp_val.shape)
            self.assertTrue(np.allclose(exp_val, computed))

        # C) Fill variable with values
        axes_order = [self.dims[2], self.dims[3],
                      self.dims[1], self.dims[0]]
        expected = (np.random.random(self.shape) * 100).transpose((2, 3, 1, 0))
        self.var.axes_order = axes_order

        for d1 in range(self.shape[0]):
            for d3 in range(self.shape[2]):
                exp_val = expected[d3, :, d1, :]
                self.var[d3, :, d1, :] = exp_val
                computed = self.var[d3, :, d1, :]
                self.assertTupleEqual(computed.shape, exp_val.shape)
                self.assertTrue(np.allclose(exp_val, computed))

        # D) Fill variable with values
        axes_order = [self.dims[2], self.dims[3],
                      self.dims[1], self.dims[0]]
        expected = (np.random.random(self.shape) * 100).transpose((2, 3, 1, 0))
        self.var.axes_order = axes_order

        for d1 in range(self.shape[0]):
            for d3 in range(self.shape[2]):
                exp_val = expected[:, :, :, :]
                self.var[:, :, :, :] = exp_val
                computed = self.var[:, :, :, :]
                self.assertTupleEqual(computed.shape, exp_val.shape)
                self.assertTrue(np.allclose(exp_val, computed))
