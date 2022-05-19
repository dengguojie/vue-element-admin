#!/bin/bash
DIR=$( cd "$( dirname "$0" )" && pwd )

if [[ $1 = "--help" ]] || [[ $1 = "-h" ]]
then
    echo "Usage: ./run.sh input output [options]"
    echo "input                                A test file whose name is '.cvs' can contain a path, such as: ./xxx.csv."
    echo "output                               Output file name, which ends with '.csv' by default."
    echo ""
    echo "[options]"
    echo "--help | -h                          Print this message."
    echo "--dynamic | -d                       This option controls whether to enable the dynamic Shape test process. The default value is true."
    echo "                                     --dynamic=false or -d=false disables the dynamic Shape test process."
    echo "--static | -s                        This option controls whether to enable the static Shape test process. The default value is true."
    echo "                                     --static=false or -s=false disables the static Shape test process."
    echo "--compile-only | --compile | --co    Compiling only. The default value is false."
    echo "--testcase | -t                      Specify test case names, splited by comma."
    echo "                                     --testcase=xxx,xxx,... or -t=xxx,xxx,..."
    echo "--device                             Specify the number of devices to be run."
    echo "--device-blacklist                   Specify the serial number of the device that does nut run."
    echo "                                     Device blacklist should be a list of device id, such as: 1,2,3,4,5,6,7 ."
    echo "--ti | --testcase-index              Specify the index of the test case to be run, such as: --ti=0-1 ."
    echo "--op | --operator                    Specify op_name for testcases."
    echo "--pc | --process-count               Specify process count- for each device, such as: --pc=1 ."
    echo "--tc | --testcase-count              Specify testcase count."
    exit 0
fi

set_env() {
    if [ $UID -eq 0 ]; then
        ASCEND_ROOT=/usr/local/Ascend/latest
        ASCEND_ROOT_OLD=/usr/local/Ascend
    else
        ASCEND_ROOT=~/Ascend/latest
        ASCEND_ROOT_OLD=~/Ascend
    fi

    if [ ! -d ${ASCEND_ROOT}/compiler ] || [ ! -d ${ASCEND_ROOT}/opp ]; then
        ASCEND_ROOT=$ASCEND_ROOT_OLD
    fi

    # Ascend Log Options
    export ASCEND_GLOBAL_LOG_LEVEL=3
    export ASCEND_GLOBAL_EVENT_ENABLE=0
    export ASCEND_SLOG_PRINT_TO_STDOUT=0

    # TF Log level
    export TF_CPP_MIN_LOG_LEVEL=2

    # OS Limitation Options
    ulimit -l 65535
    ulimit -n 655300
    ulimit -s 81920

    # Python Hash Random Seed Options
    export PYTHONHASHSEED=0
    # Path Options
    export PATH=$PATH:$DIR/tbetoolkits/utilities
    export PATH=$PATH:${ASCEND_ROOT}/compiler/ccec_compiler/bin
    # Segfault Tracing
    unset LD_PRELOAD
    export LD_PRELOAD=libSegFault.so
    # segv ill abrt fpe bus stkflt
    unset SEGFAULT_SIGNALS
    export SEGFAULT_SIGNALS=all

    # Environments
    unset LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ASCEND_ROOT}/compiler/lib64/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ASCEND_ROOT}/runtime/lib64/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ASCEND_ROOT}/opp/op_impl/built-in
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ASCEND_ROOT}/opp/op_impl/built-in/ai_core/tbe/op_tiling
    if [ -d "${ASCEND_ROOT_OLD}/fwkacllib" ]; then
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ASCEND_ROOT_OLD}/fwkacllib/lib64
    else
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ASCEND_ROOT}/fwkacllib/lib64
    fi
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ASCEND_ROOT_OLD}/driver/lib64/driver/
    export ASCEND_OPP_PATH=${ASCEND_ROOT}/opp
    export PYTHONPATH=$PYTHONPATH:${ASCEND_ROOT}/opp/op_impl/built-in/ai_core/tbe

    # Remove logs and session dependant artifacts
    rm -rf kernel_meta
    rm -rf tbetoolkits-*.log
    rm -rf ~/ascend/log/plog
}

main() {
    set_env
    # Pass all arguments to test infrastructure
    python3 -u "$DIR/run_tbe_toolkits.py" $@
}

main $@
