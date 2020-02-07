from setuptools import setup

setup(
   name='netCDF4_streamer',
   version='0.1.0',
   description='A simple extension of default NetCDF4 driver for streaming '
               'to the variable and reading from the variable in some '
               'reasonable size chunks.',
   author='David Salac',
   author_email='info@davidsalac.eu',
   packages=['netCDF4_streamer'],
   install_requires=['numpy', 'netCDF4'],
)
