from distutils.core import setup

setup(
    name='RLE',
    version='0.1',
    packages=['rle',],
    license='ISC',
    author="Dmytro S Lituiev",
    author_email="d.lituiev@gmail.com",
    description="run length encoding",
    long_description=open('README.md').read(),
    install_requires=['numpy', 'numba'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: ISC",
        "Operating System :: OS Independent",
    ],
)
