# Enhancement of the netCDF4 Python driver
Author: David Salac <http://www.github.com/david-salac>

Extends the default NetCDF4 driver (package netCDF4) by providing helpful
functionality like reading and writing to variable in some chunks
(previously pre-stored in memory - Dask like style of chunking)
or dealing with variables regardless of its dimension order.
Principally directly extends netCDF4 Dataset class with new functionality.
Covers most of the functionality that is often the only reason why developers
choose to use stogy libraries like xarray or Dask.

## Purpose of the streamer (use cases)
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

Another quite typical situation occurs when a user does not exactly know
the order of dimensions in the variable before you write into it (or
read from it). Or similarly, when the application is in development
and order can be changed at some point. This issue is efficiently
solved in xarray library. But installing xarray involves many drawbacks
like having a massive library in a system (that is not useful otherwise)
or risking memory leaks (often related even to the correct usage of xarray).
This library directly extends the native netCDF4 drive. Accordingly, it
is minimalistic and solves exactly this typical problem.

## How to use the script for streaming to the variable?
Suppose that user wants to create a variable called `'var'` with dimensions
called `(d1, d2, d3)` with sizes `(500, 20, 600)` and swapped (stream) to the
variable using the dimension 'd2' as a pivotal one. 

Following example shows the way how to deal with this issue. There are two
options what to stream: stream line by line (entity by entity) or stream the 
whole blob of data.

Once the job is finished, variable must be closed using the `flush` method.
```
import netCDF4_enhancement as ne

# Create a file handler:
fh = ne.NetCDF4Streamer("PATH_TO_NC.nc", "w")
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

## How to use the script for reading in chunks from the variable?
The purpose of this part is to read from the NetCDF4 variable in some chunks 
of the defined size. Logic is the opposite of the streaming logic described 
above.

Suppose that we have a variable described above and we want to read chunks by 
slicing the dimension d2:
```
import netCDF4_enhancement as ne

fh = ne.NetCDF4Streamer("PATH_TO_NC.nc", "r")

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
netCDF4_variable_object: netCDF4.Variable = wraped_variable.netcdf4_variable
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

For reading/writing data afterwards, you can use standard approach:
```
# Using slices
variable[:, :, :, :] = new_value_to_be_written  # <- for writing
value_inside = variable[:, :, :, :]  # <- for reading

# Slices and concrete indices:
variable[:, :, 34, :] = new_value_to_be_written  # <- for writing
value_inside = variable[:, :, 34, :]  # <- for reading

# Alternative method for reading value is using .sel method:
#   - First option expect arguments ordered by the value of 'axes_order'
value_inside = variable.sel(1, 2, 3, 4)
#   - Another approach is bit more universal, keys are names of dimensions
value_inside = variable.sel(d1=1, d2=2, d3=3, d4=4)
```
This overloaded operator for accessing the values does always return either
`numpy.ndarray` object or the `float` value (never masked array).

## API documentation
Documentation of the basic functionality.
### NetCDF4Streamer class
Class `NetCDF4Streamer` is the child of `netCDF4.Dataset` class. Accordingly
inherits all the methods.

There are following new methods:
* `createStreamerVariable`: has the same interface as 
`netCDF4.Dataset.createVariable`. 
Creates new variable inside NetCDF4 file.
Returns `NetCDF4StreamerVariable` object. 
Method adds two new optional (keyword type) parameters:
    - `chunk_dimension (Optional[str])`: define dimension by that data are
    streamed to the variable, if `None` the first dimension is chosen. 
    Default value is `None`. _It is pertinent only if you want to use
    streaming to the variable._
    - `chunk_size_mb (float)`: Size of the memory chunk to that data
    are streamed first (before they are streamed to the variable
    on the disk). Unit is megabyte. Default size is 512 MB.
    _It is pertinent only if you want to use streaming to the variable._
    
* `openStreamerVariable`: has the same logic as 
`netCDF4.Dataset['variable_name']'`. 
Opens an existing variable inside NetCDF4 file.
Returns `NetCDF4StreamerVariable` object. 
Method adds two new optional (keyword type) parameters:
    - `chunk_dimension (Optional[str])`: define dimension by that data are
    streamed to the variable, if `None` the first dimension is chosen. 
    Default value is `None`. _It is pertinent only if you want to use
    streaming to the variable._
    - `chunk_size_mb (float)`: Size of the memory chunk to that data
    are streamed first (before they are streamed to the variable
    on the disk). Unit is megabyte. Default size is 512 MB.
    _It is pertinent only if you want to use streaming to the variable._

### NetCDF4StreamerVariable class
Represent the crucial class in the logic. 

There are following methods and properties:
* `streamNumpyData`: Method that stream the data to the NetCDF4 variable.
With parameters:
    - `data (np.ndarray)`: The line or the blob of the data to be streamed
        to the variable.
    - `single_entity (bool)`: If True data are streamed as a single "line"
        (one entity), if False the whole blob is streamed. Number of
        dimensions is of -1 smaller for True value.
* `yieldNumpyData`: Method that yields the data from the NetCDF4 variable.
With parameters:
    - `single_entity (bool)`: If True data are streamed as a single "line"
        (one entity), if False the whole blob is streamed. Number of
        dimensions is of -1 smaller for True value.
* `flush`: Important method that store temporal results of streaming into
the physical variable.
* `axes_order`: Property that gets or sets dimension order that is required.
Default order can be re-seted using `obj.axes_order = None`. Returns/accepts
the tuple of dimension names (string).
* `dimensions`: Return the physical dimension order (tuple of dimension
names as string values).
* `__getitem__`: Operator for getting value in the position reflecting order
of dimensions defined by the `axes_order` property (similar logic to the
normal `netCDF4` variable). Always return either `float` value or
`np.ndarray` value (never the mask array).
* `__setitem__`: Operator for setting value in the position reflecting order
of dimensions defined by the `axes_order` property (similar logic to the
normal `netCDF4` variable).
* `netcdf4_variable`: Return the variable as a `netCDF4.Variable` object.
* `sel`: Return the value at given position.
Location can be defined either by the positional argument (following
the order defined by the 'axes_order' property, or by the
keyword logic with key equals to the dimension name and
value equals to the selected position.

## Software User Manual (SUM), how to use it?
### Installation
To install the most actual package, use the command:
```
git clone https://github.com/david-salac/NetCDF4-variable-streamer
cd NetCDF4-variable-streamer/
python setup.py install
```
or simply install using PIP:
```
pip install netCDF4-enhancement
```
#### Running of the unit-tests
For running package unit-tests, use command:
```
python setup.py test
```
In order to run package unit-tests you need to clone package first.
