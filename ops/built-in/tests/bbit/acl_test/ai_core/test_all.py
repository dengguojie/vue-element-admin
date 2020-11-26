import os
from op_test_frame.st.op_st import run_st

if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    run_st(soc_version="Ascend310", case_dir=cur_dir, out_path="../out", run_mode="gen")
