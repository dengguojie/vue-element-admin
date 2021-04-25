#!/usr/local/bin/python3.7
# encoding: utf-8
import time

import os
import sys
import subprocess

ut_flag = False

# run ut
ut_path = os.path.abspath("ut/")
msopgen_source_code = os.path.abspath("../msopgen/op_gen")
msopst_source_code = os.path.abspath("../op_test_frame/python/op_test_frame/st")
gen_cmd = ['python3.7', '-m', 'pytest', ut_path, '--cov=' + msopgen_source_code, '--cov=' + msopst_source_code, "--cov-report=html:ut_report"]
result_opgen = subprocess.Popen(gen_cmd, shell=False,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
while result_opgen.poll() is None:
    line = result_opgen.stdout.readline()
    line = line.strip()
    if line:
        print(line)
if result_opgen.returncode == 0:
    ut_flag = True
    print("run ut success")
else:
    print("run ut failed")

if ut_flag:
    sys.exit(0)
else:
    sys.exit(1)