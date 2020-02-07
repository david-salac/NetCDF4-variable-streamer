# Simple chunk streamer to (and from) the NetCDF4 variable
A simple extension of default NetCDF4 driver for streaming to the variable and
reading from the variable in some reasonable size chunks.

## Purpose of the streamer
The typical task that occurs during data processing is to swap computed data
to the disk (due to shortage of memory). This can be done by using Dask
and similar tools that are notoriously known for the inefficiency (the 
developer has to struggle with the finding of the proper chunk size etc.). 
Another way is to stream directly to the NetCDF4 file. Unfortunately, default 
NetCDF4 driver available in Python does not support any simple way to do so 
(and neither does GDAL or xarray).

The same rationale is valid also for the reading of the data from the large 
NetCDF4 file where the size of the variable can exceed the size of the 
available memory.

The presented utility offers the simplest way how to treat this issue. It 
allows the developer directly stream to the desired variable inside the NetCDF4 
file (by selected dimension). It also allows to read (yield) from the
NetCDF4 variable the chunks of data. The main class simply extends (inherits) 
from the `netCDF4.Dataset` class and add the possibility to create a variable 
for streaming using method `createStreamerVariable`.    

## How to use the script for writing?
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
variable.streamNumpyData(data_set, single_entity=True)
variable.flush()  # MUST BE CALLED EXPLICITELY WHEN FINISHED

# B) Writing the chunk of values:
data_set = np.random.random((500, 19, 600))
variable.streamNumpyData(data_set, single_entity=False)
variable.flush()  # MUST BE CALLED EXPLICITELY WHEN FINISHED

# ...

# Best practice is to close file at the end (not required)
fh.close()
```

## How to use the script for reading in chunks?
The purpose of this part is to read from the NetCDF4 variable in some chunks 
of the defined size. Logic is the opposite of the streaming logic described 
above.

Suppose that we have a variable described above and we want to read chunks by 
slicing the dimension d2:
```
from NetCDF4_streamer import NetCDF4Streamer

fh = NetCDF4Streamer("PATH_TO_NC.nc", "r")

variable = fh.openStreamerVariable("var", 
                                   chunk_dimension="d2",
                                   chunk_size_mb=3)

# ...
# For reading the file "line by line"
for line in variable.yieldNumpyData(True):
    # Work with the line

# ...
# For reading the file by blobs (chunks)
for blob in variable.yieldNumpyData(False):
    # Work with the blob
# ...

```

## Accessing the NetCDF4 variable inside streamed variable
It can be helpful to access the `netCDF4.Variable` object inside the
streamed variable (e. g. for defining of some attributes). To do so
follow the logic (example define the attribute `description`):
```
# ...
wrapped_variable = fh.createStreamerVariable("var", "f8", ("d1", "d2", "d3")),
                                             chunk_dimension="d2",
                                             chunk_size_mb=3)

# ...
# Access the netCDF4.Variable object:
netCDF4_variable_object: netCDF4.Variable = wraped_variable.variable
# Define the description attribute (in writing/append mode only):
netCDF4_variable_object.description = "Streamed variable!"
# ...
```

## Reading / writing data in the not-matching dimension order
If the dimension order of data set to be stream to the file or 
the expected dimension order for file to be read from the file do not
match to the actual order of dimension in the file, you can use the 
`axes_order` property to treat this issue.

For example:
```
variable.axes_order = ("d2", "d3", "d1")
```
After calling this command, you can stream to the variable using proposed
dimension order or read in this order.

If you want to reset to the default dimension order, just write:
```
variable.axes_order = None
```

# Install as a package
If you want to install application as a Python package using PIP, use:
```
python setup.py install
```
