# Simple chunk streamer to the NetCDF4 variable
A simple extension of default NetCDF4 driver for streaming to the variable in
some reasonable size chunks. 

## Purpose of the streamer
The typical task that occurs during data processing is to swap computed data
to the disk (due to shortage of memory). This can be done by using Dask
and similar tools that are notoriously known for the inefficiency (the 
developer has to struggle with the finding of the proper chunk size etc.). 
Another way is to stream directly to the NetCDF4 file. Unfortunately, default 
NetCDF4 driver available in Python does not support any simple way to do so 
(and neither does GDAL or xarray).

The presented utility offers the simplest way how to treat this issue. It 
allows the developer directly stream to the desired variable inside the NetCDF4 
file (by selected dimension). The main class simply extends (inherits) from the 
`netCDF4.Dataset` class and add the possibility to create a variable for 
streaming using method `createStreamerVariable`.    

## How to use the script?
Suppose that user wants to create a variable called 'var' with dimensions
called (d1, d2, d3) with sizes (500, 20, 600) and swapped (stream) to the
variable using the dimension 'd2' as a pivotal one. 

Following example shows the way how to deal with this issue. There are two
options what to stream: stream line by line (entity by entity) or stream the 
whole blob of data.

Once the job is finished, variable has to be closed using the `flush` method.
```
from NetCDF4_streamer import NetCDF4Streamer

# Create a file handler:
fh = NetCDF4Streamer("PATH_TO_NC.nc", "w")
# Define dimensions with names: ("d1", "d2", "d3"), sizes (500, 20, 600)
fh.createDimension("d1", 500)
fh.createDimension("d2", 20)
fh.createDimension("d3", 600)

# Suppose that user wants to have dimensions (d1, d2, d3) and stream
# to the variable using the dimension "d2" and the in-memory "swap" (chunk)
# of size 3 MB.
# Create a streamed variable called 'var' (following expected API):
variable = fh.createStreamerVariable("var", "f8", ("d1", "d2", "d3")),
                                     chunk_dimension="d2",
                                     chunk_size_mb=3)

# Writing the data to the variable:

# A) Write the single value (single entity):
data_set = np.random.random((500, 600))
variable.streamNumpyData(data_set, singe_entity=True)
variable.flush()  # MUST BE CALLED EXPLICITELY WHEN FINISHED

# B) Writing the chunk of values:
data_set = np.random.random((500, 19, 600))
variable.streamNumpyData(data_set, singe_entity=False)
variable.flush()  # MUST BE CALLED EXPLICITELY WHEN FINISHED

# ...

# Best practice is to close file at the end (not required)
fh.close()
```

### Accessing the NetCDF4 variable inside streamed variable
It can be helpful to access the `netCDF4.Variable` object inside the
streamed variable (e. g. for defining of some attributes). To do so
follow the logic (example define the attribute `description`):
```
# ...
variable = fh.createStreamerVariable("var", "f8", ("d1", "d2", "d3")),
                                     chunk_dimension="d2",
                                     chunk_size_mb=3)
# ...
# Access the netCDF4.Variable object:
netCDF4_variable_object: netCDF4.Variable = variable.variable
# Define the description attribute:
netCDF4_variable_object.description = "Streamed variable!"
# ...
```
