#!/bin/bash
DIR=$( cd "$( dirname "$0" )" && pwd )

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
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ASCEND_ROOT_OLD}/fwkacllib/lib64
    export ASCEND_OPP_PATH=${ASCEND_ROOT}/opp
    export PYTHONPATH=$PYTHONPATH:${ASCEND_ROOT}/opp/op_impl/built-in/ai_core/tbe

    # Remove logs and session dependant artifacts
    rm -rf tbetoolkits-*.log
    rm -rf ~/ascend/log/plog
}

main() {
    set_env
    # Pass all arguments to test infrastructure
    python3 -u "$DIR/run_tbe_toolkits.py" $@
}

main $@
