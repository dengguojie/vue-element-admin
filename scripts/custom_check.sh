#!/bin/bash

STATUS_SUCCESS=0
STATUS_FAILED=-1

usage() {
    echo "==================================================================="
    echo "Usage:"
    echo "     bash scripts/custom_check.sh --pr_changed_file=pr_filelist.txt"
    echo "you can get pr_filelist.txt with:"
    echo "     git diff --name-only > pr_filelist.txt"
    echo "==================================================================="
    exit $STATUS_FAILED
}

checkopts() {
    change_files=""
    GETOPT_ARGS=`getopt -o p:h -al pr_changed_file:,help: -- "$@"`
    eval set -- "$GETOPT_ARGS"
    #get opt
    while [ -n "$1" ]
    do
        case "$1" in
            -p|--pr_changed_file) change_files=$2; shift 2;;
            -h|--help) usage;;
            --) break ;;           
        esac
    done
}

show_content() {
    if [[ "$change_files"x == ""x ]]; then
        usage
    fi
    change_filelist_content=`cat $change_files`
    echo "[INFO]changed files:"
    echo "$change_filelist_content"
    echo ""
}

check_file() {
    file_name=$1
    if [[ $file_name =~ "impl/dynamic/" ]] && [ "${file_name##*.}"x = "py"x ]; then
        # check import te
        if [ `grep -c "import te" $file_name` -ne '0' ]; then
            echo "[ERROR]\"import te\" has been abandoned, please use a new interface \"import tbe\""
            grep -nH "import te" $file_name
            exit $STATUS_FAILED
        fi
        # check from te import 
        if [ `grep -c "from te import" $file_name` -ne '0' ]; then
            echo "[ERROR]\"from te import\" has been abandoned, please use a new interface \"from tbe import\""
            grep -nH "from te import" $file_name
            exit $STATUS_FAILED
        fi
    fi

    if [[ $file_name =~ "impl/" ]] && [ "${file_name##*.}"x = "py"x ]; then
        # check print
        if [ `grep -c " print(" $file_name` -ne '0' ]; then
            echo "[ERROR]\"print\" is forbidden, please use raise RuntimeError!"
            grep -nH " print(" $file_name
            exit $STATUS_FAILED
        fi
        # check tikdb.debug_print
        if [ `grep -c ".tikdb.debug_print(" $file_name` -ne '0' ]; then
            echo "[ERROR]\"tik_instance.tikdb.debug_print(\" is forbidden, please use raise RuntimeError!"
            grep -nH ".tikdb.debug_print(" $file_name
            exit $STATUS_FAILED
        fi
    fi
}

main() {
    checkopts "$@"
    show_content
    for line in $change_filelist_content
    do
        check_file $line
    done
}

main "$@"
exit $STATUS_SUCCESS