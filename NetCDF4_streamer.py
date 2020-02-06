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
        pos (int): The position (index) of the dimension for streaming.
        chunk_size (int): Actual size of the chunk in number of items.

        p_chunk (int): The position in the in-memory writing chunk.
        p_variable (int): The position in the variable to write.
        _chunk (np.ndarray): Numpy in-memory chunk.

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

        self.pos: int = [dim.name for dim in var.get_dims()].index(dimension)
        self.p_chunk: int = 0
        self.p_variable: int = 0

        # The chunks size in the number of items
        self.chunk_size = chunk_size_mb * 1_000_000
        self.chunk_size //= (8 * np.prod(lengths) // lengths[self.pos])
        if self.chunk_size < 1:
            raise ValueError("The chunk size is not sufficient")

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
            single_entity (bool): If True data are streamed as a single line,
                if False the whole blob is streamed. Number of dimensions
                is of -1 smaller for True value.
        """
        if not(self.write_mode):
            raise ValueError("Variable is in the read mode!")
        if len(data.shape) != (len(self.lengths) - int(single_entity)):
            raise ValueError("Number of dimensions does not match!")

        if single_entity:
            # The index for the writing:
            pos = [slice(0, dim_size) for dim_size in self.lengths]
            pos[self.pos] = self.p_chunk
            self._chunk[tuple(pos)] = data
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
        # The index for the reading:
        pos = [slice(0, dim_size) for dim_size in self.lengths]
        if single_entity:
            # Stream one entity (dimension length is len(dim) - 1)
            for read_idx in range(self.lengths[self.pos]):
                pos[self.pos] = read_idx
                # Yield the single value
                yield self.variable[tuple(pos)]
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
                pos[self.pos] = slice(chunk_idx_start, chunk_idx_end)
                # Yield the chunk of values
                yield self.variable[tuple(pos)]

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

    def __del__(self):
        self.flush()


class NetCDF4Streamer(netCDF4.Dataset):
    """Extension of the native NetCDF4 driver. Allows to create the variable
        for streaming of data in some reasonable chunk size.

    Usage:
        To create the file write:
            >>> fh = NetCDF4Streamer("PATH_TO_NC.nc", "w")
        To create a dimension write:
            >>> fh.createDimension("d1", 500)
        To create a 'streamed' variable for writing (using in-memory chunk of size 3 MB):
            >>> fh.createStreamerVariable("var", "f8", ("d1", )), chunk_dimension="d1", chunk_size_mb=3)
        To open a 'streamed' ('yield') variable for reading (using in-memory chunk of size 3 MB):
            >>> fh.openStreamerVariable("var",  chunk_dimension="d1", chunk_size_mb=3)
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
                               chunk_dimension: str,
                               chunk_size_mb: float = 512) \
            -> NetCDF4StreamerVariable:

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

        # Return the variable streamer object:
        return NetCDF4StreamerVariable(var=variable,
                                       lengths=tuple(dim_lengths),
                                       dimension=chunk_dimension,
                                       chunk_size_mb=chunk_size_mb,
                                       write_mode=True)

    def openStreamerVariable(self,
                             varname: str,
                             *,
                             chunk_dimension: str,
                             chunk_size_mb: float = 512):
        # Select the variable
        variable = self[varname]

        # Create an array with sizes of each dimension in variable:
        dim_lengths = []
        for dim in variable.dimensions:
            dim_lengths.append(self.dimensions[dim].size)

        # Return the variable streamer object:
        return NetCDF4StreamerVariable(var=variable,
                                       lengths=tuple(dim_lengths),
                                       dimension=chunk_dimension,
                                       chunk_size_mb=chunk_size_mb,
                                       write_mode=False)
