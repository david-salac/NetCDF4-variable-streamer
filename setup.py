import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='netCDF4-enhancement',
    version='0.1.2',
    author='David Salac',
    author_email='info@davidsalac.eu',
    description='Extends the default NetCDF4 driver by providing helpful'
                ' functionality like reading and writing to variable in'
                ' some chunks or dealing with variables regardless of'
                ' its dimension order. Principally directly extends netCDF4'
                ' Dataset class with new functionality. Covers most of the'
                ' functionality that is often the only reason why developers'
                ' choose to use stogy libraries like xarray.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/david-salac/NetCDF4-variable-streamer",
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'netCDF4'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
