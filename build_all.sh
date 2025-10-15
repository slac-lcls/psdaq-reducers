#!/bin/bash

# NB: This script may needed to be run multiple times after starting from a
#     clean slate (e.g., after a -f (force-clean)) as cmake doesn't initially
#     seem to generate proper install/cmake/*Targets.cmake files.  This results
#     in dependent builds not being able to find this package's header files.

set -e

export PACKAGE_PREFIX=`pwd`
export INSTDIR=`pwd`/install

cuda_arch="86"
cmake_option="RelWithDebInfo"
force_clean=0

while getopts "a:c:f" opt; do
  case $opt in
    a) cuda_arch="$OPTARG"
       ;;
    c) cmake_option="$OPTARG"
       ;;
    f) force_clean=1
       ;;
    \?) echo "Usage: ${BASH_SOURCE[0]} [-a <CUDA Architecture>] [-c {Release, Debug, RelWithDebInfo}] [-f]"
        echo "  -a  CUDA architecture [$cuda_arch]"
        echo "  -c  Build type [$cmake_option]"
        echo "  -f  Force clean before building"
        exit 1
        ;;
  esac
done

echo "CMAKE_BUILD_TYPE:" $cmake_option

if [ $force_clean == 1 ]; then
    echo "force_clean"
    for entry in `ls -d */build`; do
        if [ -d $entry ]; then
            echo "rm -rf $entry"
            rm -rf $entry
        fi
    done
    echo "rm -rf $INSTDIR"
    rm -rf $INSTDIR
fi

function cmake_build() {
    cd $1
    shift
    mkdir -p build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=$INSTDIR -DCMAKE_PREFIX_PATH=$PACKAGE_PREFIX -DCMAKE_BUILD_TYPE=$cmake_option $@ ..
    make -j 4 install
    cd ../..
}

cmake_build lc -DCMAKE_CUDA_ARCHITECTURES=$cuda_arch
cmake_build pfpl -DCMAKE_CUDA_ARCHITECTURES=$cuda_arch
cmake_build cuSZ -DPSZ_BACKEND=cuda -DPSZ_BUILD_EXAMPLES=on -DCMAKE_CUDA_ARCHITECTURES=$cuda_arch
cmake_build cuSZp -DCMAKE_CUDA_ARCHITECTURES=$cuda_arch
