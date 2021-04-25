#!/usr/local/bin/python3.7
# encoding: utf-8
import time

import os
import sys
import subprocess

st_flag = False

# run st
st_path = os.path.abspath("st/")
msopgen_source_code = os.path.abspath("../msopgen/op_gen")
msopst_source_code = os.path.abspath("../op_test_frame/python/op_test_frame/st")
gen_cmd = ['python3.7', '-m', 'pytest', st_path, '--cov=' + msopgen_source_code, '--cov=' + msopst_source_code, "--cov-report=html:st_report"]
result_opgen = subprocess.Popen(gen_cmd, shell=False,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
while result_opgen.poll() is None:
    line = result_opgen.stdout.readline()
    line = line.strip()
    if line:
        print(line)
if result_opgen.returncode == 0:
    st_flag = True
    print("run st success")
else:
    print("run st failed")

if st_flag:
    sys.exit(0)
else:
    sys.exit(1)
