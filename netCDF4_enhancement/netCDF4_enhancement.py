from typing import Optional, Union, Tuple

import netCDF4
import numpy as np


class NetCDF4StreamerVariable(object):
    """Encapsulate the variable for streaming.

    Attributes:
        write_mode (bool): If True variable streams to file, if False
            yields from the file.
        variable (netCDF4.Variable): The 'real' netCDF4.Variable that is used
            for streaming of the data.
        lengths (tuple): The tuple of integers containing the dimension sizes.
        pos (int): The position of the dimension for streaming (relative
            to other indices). For example we have 4 dimensions and we do
            stream by the fourth (then pos = 3 since we index from zero).
            The value of the pos is in range from 0 to len(lengths).
        chunk_size (int): Actual size of the chunk in number of items.

        p_chunk (int): The position in the in-memory writing chunk.
        p_variable (int): The position in the variable to write.
        _chunk (np.ndarray): Numpy in-memory chunk.

        _permutation (tuple): The permutation definition if different input
            or output axes order is required. It is a tuple of 4 values, first
            value is the default dimension position in blob version, second
            is the permutation (target), third is the default dimension
            position in the one-entity version, fourth is the permutation
            (target for the single entity).

    Warning:
        Once the writing job is finished the method 'flush' has to be called.
    """
    def __init__(self,
                 *,
                 var: netCDF4.Variable,
                 lengths: tuple,
                 dimension: str,
                 chunk_size_mb: float = 512,
                 write_mode: bool = False):
        self.write_mode: bool = write_mode
        # Define variable
        self.variable: netCDF4.Variable = var
        # Define the dimensions sizes
        self.lengths: tuple = lengths

        # Find the index position of the chunked dimension
        self.pos: int = [dim.name for dim in var.get_dims()].index(dimension)
        # The pointer to the top or the in-memory chunk
        self.p_chunk: int = 0
        # The pointer to the top of the NetCDF4 file variable
        self.p_variable: int = 0

        # The chunks (in-memory chunk) size in the number of items
        # Chunk size is computed by the production of the dimension lengths
        #   and the size of double type (which is 8 bytes)
        self.chunk_size = chunk_size_mb * 1_000_000
        self.chunk_size //= ((8 * np.prod(lengths)) // lengths[self.pos])
        if self.chunk_size < 1:
            raise ValueError("The chunk size is not sufficient")

        self._permutation: tuple = None

        if write_mode:
            # Create the in-memory chunk
            chunk_shape = list(lengths)
            chunk_shape[self.pos] = self.chunk_size
            self._chunk = np.empty(tuple(chunk_shape))

    def streamNumpyData(self,
                        data: np.ndarray,
                        single_entity: bool = True) -> None:
        """Stream the data to the NetCDF4 variable.

        Args:
            data (np.ndarray): The line or the blob of the data to be streamed
                to the variable.
            single_entity (bool): If True data are streamed as a single "line"
                (one entity), if False the whole blob is streamed. Number of
                dimensions is of -1 smaller for True value.
        """
        if not self.write_mode:
            raise ValueError("Variable is in the read mode!")
        if len(data.shape) != (len(self.lengths) - int(single_entity)):
            raise ValueError("Number of dimensions does not match!")

        data = self._permutate(data, single_entity)

        if single_entity:
            # The (tuple) of indices for the writing:
            work_idx = [slice(0, dim_size) for dim_size in self.lengths]
            work_idx[self.pos] = self.p_chunk
            self._chunk[tuple(work_idx)] = data
            self.p_chunk += 1
            # If the chunk exceeds the limit:
            if self.p_chunk == self.chunk_size:
                self.flush()
        else:
            # No need for chunking (everything is in memory)
            # Flush the cache
            self.flush()
            self._write_chunk(data, data.shape[self.pos])

    def yieldNumpyData(self,
                       single_entity: bool = True) -> None:
        """Yields the data from the NetCDF4 variable.

        Args:
            single_entity (bool): If True data are yields as a single line,
                if False the whole blob is yield. Number of dimensions
                is of -1 smaller for True value.
        Yields:
            np.ndarray: The single value or the blob of values from variable
        """
        if self.write_mode:
            raise ValueError("Variable is in the write mode!")
        # The index for the reading (tuple of indices):
        work_idx = [slice(0, dim_size) for dim_size in self.lengths]
        if single_entity:
            # Stream one entity (dimension length is len(dim) - 1)
            for read_idx in range(self.lengths[self.pos]):
                work_idx[self.pos] = read_idx
                # Yield the single value
                yield self._permutate(self.variable[tuple(work_idx)],
                                      single_entity,
                                      True)
        else:
            # Stream the whole chunk:
            # Starting and ending index for reading
            chunk_idx_start = 0
            chunk_idx_end = 0

            # Compute number of chunks
            nr_chunks = self.lengths[self.pos] // self.chunk_size
            # Treating the reminder
            nr_chunks += int((self.lengths[self.pos] % self.chunk_size) > 0)

            # Stream whole chunk
            for chunk_pos in range(nr_chunks):
                chunk_idx_start = chunk_idx_end
                if chunk_pos == nr_chunks - 1:
                    # Treating the last chunk:
                    chunk_idx_end += self.lengths[self.pos] % self.chunk_size
                else:
                    # Treating the standard chunk
                    chunk_idx_end += self.chunk_size

                # Determine the index for chunking:
                work_idx[self.pos] = slice(chunk_idx_start, chunk_idx_end)
                # Yield the chunk of values
                yield self._permutate(self.variable[tuple(work_idx)],
                                      single_entity,
                                      True)

    def flush(self):
        """Store the results. Has to be called at the end.
        """
        if self.write_mode:
            self._write_chunk(self._chunk, self.p_chunk)
            self.p_chunk = 0

    def _write_chunk(self, chunk: np.ndarray, p_chunk):
        """Auxiliary method for streaming of the data to the variable.

        Makes streaming of the data blob easier.

        Args:
            chunk (np.ndarray): The chunk to be streamed.
            p_chunk (int): The pointer to the top of the chunk.
        """
        if p_chunk < 1:
            return
        file_index = [slice(0, dim_size) for dim_size in self.lengths]
        file_index[self.pos] = slice(self.p_variable,
                                     self.p_variable + p_chunk)

        chunk_index = [slice(0, dim_size) for dim_size in self.lengths]
        chunk_index[self.pos] = slice(0, p_chunk)

        # Write the data to the file
        self.variable[tuple(file_index)] = chunk[tuple(chunk_index)]

        self.p_variable += p_chunk

    def get_axes_order(self):
        """Get the permutation prescription for the blob (target).
        """
        temp = ["" for _ in self._permutation[1]]
        # Permutate
        for i, pr in enumerate(self._permutation[1]):
            temp[pr] = self.variable.dimensions[i]
        return tuple(temp)

    def set_axes_order(self, new_order: tuple):
        """Set the permutation (accepts values for the blob permutation
            and recompute for the single entity)

        Args:
            new_order (tuple): The n values (= number of dimension) defining
                desired dimension order (target).
        """
        if new_order is None:
            self._permutation = None
        # Permutation for the whole array (for the blob output/input)
        src_ord = list([i for i in range(0, len(self.variable.dimensions))])
        per_all = tuple([new_order.index(source)
                         for source in self.variable.dimensions])
        skip_dim_name = self.variable.dimensions[self.pos]
        dims_single_entity = tuple([dim for dim in self.variable.dimensions
                                    if dim != skip_dim_name])
        line_source = list([i for i
                            in range(0, len(self.variable.dimensions) - 1)])
        per_line = tuple([dims_single_entity.index(source)
                          for source in new_order
                          if source != skip_dim_name])
        self._permutation = (src_ord, per_all, line_source, per_line)

    axes_order = property(get_axes_order, set_axes_order)

    # Property defining axes order
    axes_order = property(get_axes_order, set_axes_order)

    def _permutate(self,
                   data_set: np.ndarray,
                   single_entity: bool,
                   read: bool = False) -> np.ndarray:
        """Permutate the dimension order of the input or output as
            required by user

        Args:
            data_set (np.ndarray): Numpy array with original dimension order.
            single_entity (bool): If the single entity is being permutated.
            read(bool): Reading inverts in the opposite direction.

        Returns:
            np.ndarray: New array with required order of dimensions.
        """
        if self._permutation is None:
            return data_set

        if single_entity:
            if read:
                return np.moveaxis(data_set,
                                   self._permutation[2],
                                   self._permutation[3])

            return np.moveaxis(data_set,
                               self._permutation[3],
                               self._permutation[2])

        if read:
            return np.moveaxis(data_set,
                               self._permutation[0],
                               self._permutation[1])

        return np.moveaxis(data_set,
                           self._permutation[1],
                           self._permutation[0])

    def __del__(self):
        self.flush()

    @property
    def dimensions(self) -> Tuple[str]:
        """Returns the ordered list of dimensions in the physical order of the
            variable.

        Returns:
            Tuple[str]: Tuple of variables (in the physical order of the
                variable on the disk).
        """
        return tuple(self.variable.dimensions)

    @property
    def netcdf4_variable(self) -> netCDF4.Variable:
        """Return the variable as netCDF4.Variable object.

        Returns:
            netCDF4.Variable: variable as netCDF4.Variable object.
        """
        return self.variable

    def sel(self, *args, **kwargs) -> Union[np.ndarray, float]:
        """Select the variable at given location.

        Location can be defined either by the positional argument (following
            the order defined by the 'axes_order' property, or by the
            keyword logic with key equals to the dimension name and
            value equals to the selected position.

        Args:
            args: Define position by the positional arguments following
                the dimension order defined by the 'axes_order' property.
            kwargs: Defines position by the kwargs logic, key is the
                name of dimension, value is the index on that dimension.
        Returns:
            Union[np.ndarray, float]: Either np.ndarray or single float value.
        """
        if len(args) > 0 and len(kwargs) > 0:
            raise ValueError("Only one selecting logic can be chosen. Either "
                             "use position logic, or kwargs logic.")
        # What dimension order is required
        expected_dim_order = self.axes_order
        # Define indices
        indices = [i for i in self.axes_order]
        for i, pos_idx in enumerate(args):
            indices[i] = pos_idx
        for key, value in kwargs.items():
            indices[expected_dim_order.index(key)] = value
        # Current dimension order of the variable
        current_dim_order = self.variable.dimensions
        # Permutation prescription
        permutation_formula = [current_dim_order.index(dim)
                               for dim in expected_dim_order]
        permutation_formula_back = [expected_dim_order.index(dim) for dim in
                                    current_dim_order]

        # Prepare the indices values
        dimension_sizes = []
        squeeze_axis = []
        for i, idx in enumerate(indices):
            if isinstance(idx, int):
                dimension_sizes.append(slice(idx, idx + 1, 1))
                squeeze_axis.append(i)
            elif isinstance(idx, slice):
                start = idx.start
                end = idx.stop
                step = idx.step
                if start is None:
                    start = 0
                if start < 0:
                    start += self.lengths[permutation_formula[i]]
                if end is None:
                    end = self.lengths[permutation_formula[i]]
                if end < 0:
                    end += self.lengths[permutation_formula[i]]
                if step is None:
                    step = 1
                if step < 0:
                    step += self.lengths[permutation_formula[i]]
                dimension_sizes.append(slice(start, end, step))
            else:
                raise AttributeError("Incorrect index type!")

        # Does the permutation
        file_index = tuple(
            dimension_sizes[i] for i in permutation_formula_back)
        # Get the value
        values = np.array(self.variable[file_index])
        # Swap dimensions:
        values = values.transpose(tuple(permutation_formula))
        # Squeeze dimensions:
        values = values.squeeze(tuple(squeeze_axis))
        if len(values.shape) == 0:
            values = float(values)
        # Return the value in expected logic
        return values

    def __getitem__(self, indices) -> Union[float, np.ndarray]:
        """Get the item on the position defined by argument indices.
            Reflects the order of axis in axes_order property.

        Warning:
            Uses np.transpose function if the dimension order is changed.
            This can lead to a memory leaks for a massive arrays (use
            streaming logic in this case).

        Args:
            indices (tuple): the position of the item (or slice)

        Returns:
            Union[float, np.ndarray]: value at position defined by indices.
        """
        return self.sel(*indices)

    def __setitem__(self, indices, value: Union[float, np.ndarray]):
        """Set the item on the position defined by argument indices to the
            value defined by the parameter 'value'. Reflects the order of axis
            in axes_order property.

        Args:
            indices (tuple): the position of the item (or slice).
            value (np.ndarray): new value to be set.
        """
        # What dimension order is required
        expected_dim_order = self.axes_order
        # Current dimension order of the variable
        current_dim_order = self.variable.dimensions
        # Permutation prescription
        permutation_formula = [current_dim_order.index(dim)
                               for dim in expected_dim_order]
        permutation_formula_back = [expected_dim_order.index(dim) for dim in
                                    current_dim_order]

        # Prepare the indices values
        dimension_sizes = []
        squeeze_axis = []
        for i, idx in enumerate(indices):
            if isinstance(idx, int):
                dimension_sizes.append(slice(idx, idx + 1, 1))
                squeeze_axis.append(i)
            elif isinstance(idx, slice):
                start = idx.start
                end = idx.stop
                step = idx.step
                if start is None:
                    start = 0
                if start < 0:
                    start += self.lengths[permutation_formula[i]]
                if end is None:
                    end = self.lengths[permutation_formula[i]]
                if end < 0:
                    end += self.lengths[permutation_formula[i]]
                if step is None:
                    step = 1
                if step < 0:
                    step += self.lengths[permutation_formula[i]]
                dimension_sizes.append(slice(start, end, step))
            else:
                raise AttributeError("Incorrect index type!")

        # Does the permutation
        file_index = tuple(
            dimension_sizes[i] for i in permutation_formula_back)
        # Get the value
        values = np.expand_dims(value, (tuple(squeeze_axis)))
        # Swap dimensions:
        values = values.transpose(tuple(permutation_formula_back))
        # Return the value in expected logic
        self.variable[file_index] = values


class NetCDF4Streamer(netCDF4.Dataset):
    """Extension of the native NetCDF4 driver. Allows to A) create the variable
        for streaming of data in some reasonable chunk size; B) create variable
        from that user can read data (or write data into it) without takeing
        dimension order into an account.

    Usage:
        To create the file write:
            >>> fh = NetCDF4Streamer("PATH_TO_NC.nc", "w")
        To create a dimension write:
            >>> fh.createDimension("d1", 500)
        To create a 'streamed' variable for writing (using in-memory chunk of
            size 3 MB):
            >>> fh.createStreamerVariable("var", "f8", ("d1", )),
            >>>                           chunk_dimension="d1",
            >>>                           chunk_size_mb=3)
        To open a 'streamed' ('yield') variable for reading (using in-memory
            chunk of size 3 MB):
            >>> fh.openStreamerVariable("var",  chunk_dimension="d1",
            >>>                         chunk_size_mb=3)
        To read (or similary write) from variable without reflecting its
        dimension order (say that some_variable has dimensions d1, d2, d3):
            >>> variable = fh.openStreamerVariable('some_variable')
            >>> variable.axes_order = ('d3', 'd2', 'd1')  # Preferable order
            >>> variable[1, :, 5] = data  # Stream data in the required order

    """

    def createStreamerVariable(self,
                               varname,
                               datatype,
                               dimensions=(),
                               zlib=False,
                               complevel=4,
                               shuffle=True,
                               fletcher32=False,
                               contiguous=False,
                               chunksizes=None,
                               endian='native',
                               least_significant_digit=None,
                               fill_value=None,
                               *,
                               chunk_dimension: str = None,
                               chunk_size_mb: float = 512) \
            -> NetCDF4StreamerVariable:
        """Create the variable as a NetCDF4StreamerVariable object.
            Most of the parameters follows the logic defined in the native
            NetCDF4 driver (netCDF4 package in Python).

        Args:
            varname (str): name of the variable to be open. Required!
            chunk_dimension (Optional[str]): define dimension by that data are
                streamed to the variable, if None the first dimension
                is chosen. Default value is None.
            chunk_size_mb (float): Size of the memory chunk to that data
                are streamed first (before they are streamed to the variable
                on the disk). Unit is megabyte. Default size is 512 MB.
        Returns:
            NetCDF4StreamerVariable: The NetCDF4StreamerVariable for
                streaming to/from the variable or reading/writing in
                dimension order relaxed way.
        """

        variable = self.createVariable(varname,
                                       datatype,
                                       dimensions,
                                       zlib,
                                       complevel,
                                       shuffle,
                                       fletcher32,
                                       contiguous,
                                       chunksizes,
                                       endian,
                                       least_significant_digit,
                                       fill_value)

        # Create an array with sizes of each dimension in variable:
        dim_lengths = []
        for dim in dimensions:
            dim_lengths.append(self.dimensions[dim].size)

        if chunk_dimension is None:
            # If the chunk_dimension argument is None, use the first
            #   dimension to streaming by.
            chunk_dimension = self[varname].dimensions[0]

        # Return the variable streamer object:
        return NetCDF4StreamerVariable(var=variable,
                                       lengths=tuple(dim_lengths),
                                       dimension=chunk_dimension,
                                       chunk_size_mb=chunk_size_mb,
                                       write_mode=True)

    def openStreamerVariable(
            self,
            varname: str,
            *,
            chunk_dimension: Optional[str] = None,
            chunk_size_mb: float = 512) -> NetCDF4StreamerVariable:
        """Open the variable as a NetCDF4StreamerVariable object.

        Args:
            varname (str): name of the variable to be open. Required!
            chunk_dimension (Optional[str]): define dimension by that data are
                streamed to the variable, if None the first dimension
                is chosen. Default value is None.
            chunk_size_mb (float): Size of the memory chunk to that data
                are streamed first (before they are streamed to the variable
                on the disk). Unit is megabyte. Default size is 512 MB.
        Returns:
            NetCDF4StreamerVariable: The NetCDF4StreamerVariable for
                streaming to/from the variable or reading/writing in
                dimension order relaxed way.
        """
        # Select the variable
        variable = self[varname]

        # Create an array with sizes of each dimension in variable:
        dim_lengths = []
        for dim in variable.dimensions:
            dim_lengths.append(self.dimensions[dim].size)

        if chunk_dimension is None:
            # If the chunk_dimension argument is None, use the first
            #   dimension to streaming by.
            chunk_dimension = self[varname].dimensions[0]

        # Return the variable streamer object:
        return NetCDF4StreamerVariable(var=variable,
                                       lengths=tuple(dim_lengths),
                                       dimension=chunk_dimension,
                                       chunk_size_mb=chunk_size_mb,
                                       write_mode=False)
