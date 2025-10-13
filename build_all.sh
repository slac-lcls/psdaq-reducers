#!/bin/bash

set -e

export PACKAGE_PREFIX=`pwd`
export INSTDIR=`pwd`/install

cmake_option="RelWithDebInfo"
force_clean=0

while getopts "c:p:s:b:fdam" opt; do
  case $opt in
    c) cmake_option="$OPTARG"
    ;;
    f) force_clean=1                       # Force clean is required building between rhel6&7
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
        exit 1
    ;;
  esac
done

echo "CMAKE_BUILD_TYPE:" $cmake_option

if [ $force_clean == 1 ]; then
    echo "force_clean"
    if [ -d app/build ]; then
        rm -rf app/build
    fi
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

cmake_build lc
cmake_build pfpl
#cmake_build_cuSZ
#cmake_build_cuSZp

