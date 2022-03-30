from sch_test_frame.ut import OpUT
from tbe.common.utils import log

ut_case = OpUT("log", "errormgr.test_dynamic_log_impl")

def test_log_info(_):
    try:
        log.info("input=[%s] and output=[%s]"%("input", "output"))
    except RuntimeError as e:
        return False
    return True

def test_log_debug(_):
    try:
        log.debug("input=[%s] and output=[%s]"%("input", "output"))
    except RuntimeError as e:
        return False
    return True

def test_log_warn(_):
    try:
        log.warn("input=[%s] and output=[%s]"%("input", "output"))
    except RuntimeError as e:
        return False
    return True

def test_log_error(_):
    try:
        log.error("input=[%s] and output=[%s]"%("input", "output"))
    except RuntimeError as e:
        return False
    return True

def test_log_event(_):
    try:
        log.event("input=[%s] and output=[%s]"%("input", "output"))
    except RuntimeError as e:
        return False
    return True

case_list = [
    test_log_info,
    test_log_debug,
    test_log_warn,
    test_log_error,
    test_log_event,
]

for item in case_list:
    ut_case.add_cust_test_func(test_func=item)