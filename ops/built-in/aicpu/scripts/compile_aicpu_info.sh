#!/bin/bash
pwd
path1=./cann/ops/built-in/aicpu/op_info_cfg
filename=$(basename $1)
cd $path1/parser
python parser_ini.py ../aicpu_kernel/aicpu_kernel.ini ../aicpu_cust_kernel/aicpu_cust_kernel.ini $filename
echo "----------------convert ini to json ok------------------"
echo `ls -l`
echo `pwd`
mv $filename ../../../../../../out/$2/host/obj/
