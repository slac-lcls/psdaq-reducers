# psdaq-reducers
Repository for holding interface codes between the LCLS2 DAQ and various
third-party data reducers

## Building
Run the ./build_all.sh script using the
[![lcls2](https://github.com/slac-lcls/lcls2)] environment.  Results are placed
in ./install.  Optionally choose the build type.  Most developers can eliminate
all the arguments to build_all.sh.

The following environment variables are needed before building the GPU portion
of the [![lcls2](https://github.com/slac-lcls/lcls2)] repo on a host with a GPU:
```bash
export LC_DIR=~/git/psdaq-reducers/install/cmake
export PFPL_DIR=~/git/psdaq-reducers/install/cmake
export cuSZp_DIR=~/git/psdaq-reducers/install/cmake
export cuSZ_DIR=~/git/psdaq-reducers/install/lib64/cmake/CUSZ
```
