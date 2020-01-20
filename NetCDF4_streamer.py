import netCDF4
import numpy as np


class NetCDF4StreamerVariable(object):
    def __init__(self, *,
                 var: netCDF4.Variable,
                 lengths: tuple,
                 dimension: str,
                 chunk_size_mb: float = 512):
        self.variable: netCDF4.Variable = var
        self.lengths: tuple = lengths

        self.pos: int = [dim.name for dim in var.get_dims()].index(dimension)
        self.p_chunk: int = 0
        self.p_variable: int = 0

        self.chunk_size = chunk_size_mb * 1_000_000
        self.chunk_size //= (8 * np.prod(lengths) // lengths[self.pos])
        if self.chunk_size < 1:
            raise ValueError("The chunk size is not sufficient")

        # Create the in-memory chunk
        chunk_shape = list(lengths)
        chunk_shape[self.pos] = self.chunk_size
        self._chunk = np.empty(tuple(chunk_shape))

    def streamNumpyData(self,
                        data: np.ndarray,
                        singe_entity: bool = True) -> None:
        if len(data.shape) != (len(self.lengths) - int(singe_entity)):
            raise ValueError("Number of dimensions does not match!")

        if singe_entity:
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

    def flush(self):
        self._write_chunk(self._chunk, self.p_chunk)
        self.p_chunk = 0

    def _write_chunk(self, chunk: np.ndarray, p_chunk):
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

        dim_lengths = []
        for dim in dimensions:
            dim_lengths.append(self.dimensions[dim].size)

        return NetCDF4StreamerVariable(var=variable,
                                       lengths=tuple(dim_lengths),
                                       dimension=chunk_dimension,
                                       chunk_size_mb=chunk_size_mb)
