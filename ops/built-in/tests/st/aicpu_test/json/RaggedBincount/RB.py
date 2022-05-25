import random

def fuzz_branch_case1():
    data_num = 11

    # input
    splits_value = [x for x in range(0, data_num)]


    return {
        "input_desc": {
            "splits": {"value": splits_value, "shape": [11]},
        }
    }



def fuzz_branch_case2():
    data_num = 101

    # input
    splits_value = [x for x in range(0, data_num)]


    return {
        "input_desc": {
            "splits": {"value": splits_value, "shape": [data_num]},
        }
    }



def fuzz_branch_case3():
    data_num = 257

    # input
    splits_value = [x for x in range(0, data_num)]


    return {
        "input_desc": {
            "splits": {"value": splits_value, "shape": [data_num]},
        }
    }



def fuzz_branch_case4():
    data_num = 1025

    # input
    splits_value = [x for x in range(0, data_num)]


    return {
        "input_desc": {
            "splits": {"value": splits_value, "shape": [data_num]},
        }
    }

def fuzz_branch_case5():
    data_num = 1024 * 4 + 1

    # input
    splits_value = [x for x in range(0, data_num)]


    return {
        "input_desc": {
            "splits": {"value": splits_value, "shape": [data_num]},
        }
    }


def fuzz_branch_case6():
    data_num = 4096 * 4 + 1

    # input
    splits_value = [x for x in range(0, data_num)]


    return {
        "input_desc": {
            "splits": {"value": splits_value, "shape": [data_num]},
        }
    }


def fuzz_branch_case7():
    data_num = 4096 * 4 * 4 + 1

    # input
    splits_value = [x for x in range(0, data_num)]


    return {
        "input_desc": {
            "splits": {"value": splits_value, "shape": [data_num]},
        }
    }


def fuzz_branch_case8():
    data_num = 262144 + 1

    # input
    splits_value = [x for x in range(0, data_num)]


    return {
        "input_desc": {
            "splits": {"value": splits_value, "shape": [data_num]},
        }
    }