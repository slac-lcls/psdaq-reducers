# psdaq-reducers
Repository for holding interface codes between the LCLS2 DAQ and various
third-party data reducers.

## Building
Run the ./build_all.sh script using the
![lcls2](https://github.com/slac-lcls/lcls2) environment.  Results are placed
in ./install.  Optionally choose the build type:
```bash
./build_all.sh -c {Release, Debug, RelWithDebInfo}
```

Source the ./setup_env.sh script to set up environment variables that are
needed before building the GPU portion of the
![lcls2](https://github.com/slac-lcls/lcls2) repo on a host with a GPU:
```bash
source ./setup_env.sh
```
