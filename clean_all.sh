#! /bin/bash

echo "Clean release of files generated during build"

REPODIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo REPODIR: $REPODIR

echo rm -rf $REPODIR/install
     rm -rf $REPODIR/install

echo find . -name "*~" -delete
     find . -name "*~" -delete

echo find . -name "*.pyc" -delete
     find . -name "*.pyc" -delete

echo find . -name build -type d -exec rm -rf {} +
     find . -name build -type d -exec rm -rf {} +

echo find . -name __pycache__ -type d -exec rm -rf {} +
     find . -name __pycache__ -type d -exec rm -rf {} +

echo find . -name dist -type d -exec rm -rf {} +
     find . -name dist -type d -exec rm -rf {} +

echo "Cleaning is done"
echo ""
echo ">>>>> [31mbuild_all.sh must be run TWICE after clean_all.sh due to a cmake issue[m <<<<<"
