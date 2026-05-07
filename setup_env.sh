#  Source this script prior to building the LCLS2 DAQ

RELDIR="$( cd "$( dirname $(readlink -f "${BASH_SOURCE[0]:-${(%):-%x}}") )" && pwd )"
export LC_DIR=$RELDIR/install/cmake
export PFPL_DIR=$RELDIR/install/cmake
export SLEEK_DIR=$RELDIR/install/cmake
export CUSZ_DIR=$RELDIR/install/lib64/cmake/CUSZ
export cuSZp_DIR=$RELDIR/install/cmake
