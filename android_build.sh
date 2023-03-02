#!/usr/bin/env bash
set -e

BUILD_TYPE=Release
ARCH=arm64-v8a
OPENCL=OFF
GPU_VENDOR=none
ADRENO=OFF
MALI=OFF
PFM=OFF
LOG=OFF
REMOVE_OLD_BUILD=OFF
INSTALL_PREFIX=/usr/local

READLINK=readlink
MAKEFILE_TYPE="Ninja"
OS=$(uname -s)

if [ $OS = "Darwin" ];then
    READLINK=greadlink
fi

SRC_DIR=$(dirname $($READLINK -f $0))
echo "the SRC_DIR val is $SRC_DIR"

function usage() {
    echo "$0 args1 ..."
    echo "available args detail:"
    echo "-d : build with Debug mode, default Release mode"
    echo "-m : machine arch(arm64-v8a, armeabi-v7a), default arm64-v8a"
    echo "-g : mobile gpu vendor(adreno, mali)"
    echo "-i : set your custom cmake install prefix, default \"/usr/local\" "
    echo "-p : enable pfm, default off"
    echo "-l : enable logging, default off"
    echo "-r : remove old build dir before make, default off"
    echo "-h : show usage"
    echo "example: $0 -g adreno"
}

while getopts "dhm:g:i:rlp" arg
do
    case $arg in
        d)
            echo "Build with Debug mode"
            BUILD_TYPE=Debug
            ;;
        m)
            echo "build with arch:$OPTARG"
            ARCH=$OPTARG
            ;;
        g)
            echo "build with gpu:$OPTARG"
            GPU_VENDOR=$OPTARG
            ;;
        i)
            echo "set your custom cmake install prefix:$OPTARG"
            INSTALL_PREFIX=$OPTARG
            ;;
        p)
            echo "enable pfm"
            PFM=ON
            ;;
        l)
            echo "enable logging"
            LOG=ON
            ;;
        r)
            echo "config REMOVE_OLD_BUILD=true"
            REMOVE_OLD_BUILD=true
            ;;
        h)
            echo "show usage"
            usage
            exit 0
            ;;
        ?)
            echo "unkonw argument"
            usage
            exit 0
            ;;
    esac
done

function cmake_build() {
    if [ $NDK_ROOT ];then
        echo "NDK_ROOT: $NDK_ROOT"
    else
        echo "Please define env var NDK_ROOT first"
        exit 1
    fi

    BUILD_DIR=$SRC_DIR/build-${ARCH}/
    BUILD_ABI=$1
    BUILD_NATIVE_LEVEL=$2
    if [ $REMOVE_OLD_BUILD = "true" ]; then
        echo "remove old build dir"
        rm -rf ${BUILD_DIR}
    else
        echo "strip remove old build"
    fi

    echo "build dir: $BUILD_DIR"
    echo "build ARCH: $ARCH"
    echo "build ABI: $BUILD_ABI"
    echo "build native level: $BUILD_NATIVE_LEVEL"
    echo "BUILD MAKEFILE_TYPE: $MAKEFILE_TYPE"
    echo "cmake install prefix: $INSTALL_PREFIX"
    echo "create build dir"
    mkdir -p $BUILD_DIR
    cd $BUILD_DIR
    cmake -G "$MAKEFILE_TYPE" \
        "-B$BUILD_DIR" \
        "-S$SRC_DIR" \
        -DCMAKE_TOOLCHAIN_FILE="$NDK_ROOT/build/cmake/android.toolchain.cmake" \
        -DANDROID_NDK="$NDK_ROOT" \
        -DANDROID_ABI=$BUILD_ABI \
        -DANDROID_NATIVE_API_LEVEL=$BUILD_NATIVE_LEVEL \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DMPERF_ENABLE_OPENCL=$OPENCL \
        -DMPERF_ENABLE_ADRENO=$ADRENO \
        -DMPERF_ENABLE_MALI=$MALI \
        -DMPERF_ENABLE_PFM=$PFM \
        -DMPERF_ENABLE_LOGGING=$LOG \
        -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX

    ninja -v
}


API_LEVEL=16
ABI="armeabi-v7a with NEON"
IFS=""
if [ "$ARCH" = "arm64-v8a" ]; then
    API_LEVEL=21
    ABI="arm64-v8a"
elif [ "$ARCH" = "armeabi-v7a" ]; then
    API_LEVEL=16
    ABI="armeabi-v7a with NEON"
else
    echo "ERR IN ARCH CONFIG, ABORT NOW!!"
    exit -1
fi

if [ "$GPU_VENDOR" = "adreno" ]; then
    OPENCL=ON
    ADRENO=ON
elif [ "$GPU_VENDOR" = "mali" ]; then
    OPENCL=ON
    MALI=ON
elif [ "$GPU_VENDOR" = "none" ]; then
    OPENCL=OFF
else
    echo "ERR IN GPU_VENDOR CONFIG, ABORT NOW!!"
    exit -1
fi

cmake_build $ABI $API_LEVEL