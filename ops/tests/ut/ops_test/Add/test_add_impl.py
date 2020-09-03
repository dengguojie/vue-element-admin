from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT("Add")

ut_case.add_broadcast_case_simple(["Ascend910", "Ascend310"], "float16", (16, 32), (16, 32))

if __name__ == "__main__":
    ut_case.run("Ascend910")
