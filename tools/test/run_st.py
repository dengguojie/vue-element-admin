#!/usr/local/bin/python3.7
# encoding: utf-8
import time

import os
import sys
import subprocess

opgen_flag = False
opst_flag = False

# run msopgen st
msopgen_st_path = os.path.abspath("./msopgen/st/msopgen_st.py")
gen_cmd = ['python3', msopgen_st_path]
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
    print("run msopgen st success")
else:
    print("run msopgen st failed")

# run msopst st
msopst_st_path = os.path.abspath("./msopst/st/msopst_st.py")
st_cmd = ['python3', msopst_st_path]
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
    print("run msopst st success")
else:
    print("run msopst st failed")


if opgen_flag and opst_flag:
    sys.exit(0)
else:
    sys.exit(1)