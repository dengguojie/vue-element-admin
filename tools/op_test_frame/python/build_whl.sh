#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2010-2018. All rights reserved.
python_cmd=$1
whl_path=$2

echo ${whl_path}
cd ${whl_path}

${python_cmd} setup.py  bdist_wheel

