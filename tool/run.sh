#!/bin/bash
# configure the environment
. tool/environment.sh

if [ "$ConfigurationStatus" != "Success" ]; then
    echo "Exit due to configure failure."
    exit
fi

set -e

mkdir -p build

cd build
cmake ..
make -j

cd ..

./build/fastbev $DEBUG_DATA $DEBUG_MODEL $DEBUG_PRECISION