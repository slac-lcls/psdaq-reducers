#!/bin/bash

# NB: This script may needed to be run multiple times after starting from a
#     clean slate (e.g., after a -f (force-clean)) as cmake doesn't initially
#     seem to generate proper install/cmake/*Targets.cmake files.  This results
#     in dependent builds not being able to find this package's header files.

set -e

export PACKAGE_PREFIX=`pwd`
export INSTDIR=`pwd`/install

cuda_arch="86;90"
cmake_option="RelWithDebInfo"
force_clean=0
tpp_build=0

while getopts "a:c:ft" opt; do
  case $opt in
    a) cuda_arch="$OPTARG"
       ;;
    c) cmake_option="$OPTARG"
       ;;
    f) force_clean=1
       ;;
    t) tpp_build=1
       ;;
    \?) echo "Usage: ${BASH_SOURCE[0]} [-a <CUDA Architecture>] [-c {Release, Debug, RelWithDebInfo}] [-f]"
        echo "  -a  CUDA architecture [$cuda_arch]"
        echo "  -c  Build type [$cmake_option]"
        echo "  -f  Force clean before building"
        echo "  -t  Don't build third party packages by default"
        exit 1
        ;;
  esac
done

echo "Applying patches..."
for d in ./patches/* ; do
    for p in "$d/"* ; do
        echo "Applying $p"
        p=$(readlink -f "$p")
        if ! pushd $(basename "$d") > /dev/null ; then
            break
        fi

        # Allow errors for this
        set +e
        O=$(patch -p1 -N < "$p")
        if [ $? -ne 0 ]; then
            # Patch always gives us an error code, even for previously applied patches,
            # so we have to parse stdout :(
            if ! echo "$O" | grep "Reversed " ; then
                echo "Patch ${p} failed to apply"
                exit 1
            fi
        fi
        set -e

        popd > /dev/null
    done
done

echo "CMAKE_BUILD_TYPE:" $cmake_option
echo "CMAKE_CUDA_ARCHITECTURES:" $cuda_arch

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
    if [ -d $1 ]; then
        cd $1
        shift
        mkdir -p build
        cd build
        cmake -DCMAKE_INSTALL_PREFIX=$INSTDIR -DCMAKE_PREFIX_PATH=$PACKAGE_PREFIX -DCMAKE_BUILD_TYPE=$cmake_option $@ ..
        make -j 4 install
        cd ../..
    else
        echo "Reducer $1 not found"
    fi
}

function cmake_build_3rd() {
    if [ -d $1 ]; then
        cd $1
        shift
        if [ -d $1 ]; then
            tpp=$1
            cd $1
            shift
            mkdir -p build
            cd build
            cmake -DCMAKE_INSTALL_PREFIX=$INSTDIR/$tpp -DCMAKE_PREFIX_PATH=$PACKAGE_PREFIX -DCMAKE_BUILD_TYPE=$cmake_option $@ ..
            make -j 4 install
            cd ../..
        else
            echo "Third party package $2 not found"
        fi
        cd ..
    else
        echo "Reducer $1 not found"
    fi
}

cmake_build lc -DCMAKE_CUDA_ARCHITECTURES=$cuda_arch
cmake_build pfpl -DCMAKE_CUDA_ARCHITECTURES=$cuda_arch
cmake_build sleek -DCMAKE_CUDA_ARCHITECTURES=$cuda_arch
# Is CuSZ from this repo obsolete?
#cmake_build cuSZ -DPSZ_BACKEND=cuda -DPSZ_BUILD_EXAMPLES=on -DCMAKE_CUDA_ARCHITECTURES=$cuda_arch
cmake_build cuSZp -DCMAKE_CUDA_ARCHITECTURES=$cuda_arch
# EIP also builds a cuSZ
if [ $tpp_build == 1 ]; then
    cmake_build_3rd eip EIP -DPSZ_BACKEND=cuda -DPSZ_BUILD_EXAMPLES=on -DCMAKE_CUDA_ARCHITECTURES=$cuda_arch
fi
cmake_build eip -DCMAKE_CUDA_ARCHITECTURES=$cuda_arch
