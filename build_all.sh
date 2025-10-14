#!/bin/bash

set -e

export PACKAGE_PREFIX=`pwd`
export INSTDIR=`pwd`/install

cmake_option="RelWithDebInfo"
force_clean=0

while getopts "c:f" opt; do
  case $opt in
    c) cmake_option="$OPTARG"
    ;;
    f) force_clean=1                       # Force clean is required building between rhel6&7
    ;;
    \?) echo "Usage: ${BASH_SOURCE[0]} [-c {Release, Debug, RelWithDebInfo}] [-f]"
        echo "  -c  Build type"
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

cmake_build lc -DCMAKE_CUDA_ARCHITECTURES="86"
cmake_build pfpl -DCMAKE_CUDA_ARCHITECTURES="86"
cmake_build cuSZ -DPSZ_BACKEND=cuda -DPSZ_BUILD_EXAMPLES=on -DCMAKE_CUDA_ARCHITECTURES="86"
cmake_build cuSZp -DCMAKE_CUDA_ARCHITECTURES="86"
