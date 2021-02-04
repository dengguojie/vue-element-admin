#!/usr/local/bin/python3.7
# encoding: utf-8
import time

import os
import sys
import subprocess

opgen_flag = False
opst_flag = False

# run msopgen ut
msopgen_ut_path = os.path.abspath("./msopgen/ut/msopgen_ut.py")
gen_cmd = ['python3', msopgen_ut_path]
result_opgen = subprocess.Popen(gen_cmd, shell=False,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
while result_opgen.poll() is None:
    line = result_opgen.stdout.readline()
    line = line.strip()
    if line:
        print(line)
if result_opgen.returncode == 0:
    opgen_flag = True
    print("run msopgen ut success")
else:
    print("run msopgen ut failed")

# run msopst ut
msopst_ut_path = os.path.abspath("./msopst/ut/msopst_ut.py")
st_cmd = ['python3', msopst_ut_path]
result_opst = subprocess.Popen(st_cmd, shell=False,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
while result_opst.poll() is None:
    line = result_opst.stdout.readline()
    line = line.strip()
    if line:
        print(line)
if result_opst.returncode == 0:
    opst_flag = True
    print("run msopst ut success")
else:
    print("run msopst ut failed")


if opgen_flag and opst_flag:
    sys.exit(0)
else:
    sys.exit(1)