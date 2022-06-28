# # -*- coding:utf-8 -*-
import warnings

from sch_test_frame.common.register import add_cust_test_func
from sch_test_frame.ut import OpUT
from tbe.dsl.classifier import shape_classifier

warnings.filterwarnings("ignore")
ut_case = OpUT("classifier", "classifier.test_classifier_impl")


@add_cust_test_func(ut_case)
def test_register_classifier_with_func_register(_):

    @shape_classifier.register_classifier("mode_1")
    def mode_1_classifier():
        pass

    classifier_func = shape_classifier._classifiers.get("mode_1")

    return classifier_func is not None


@add_cust_test_func(ut_case)
def test_register_classifier_with_func_exec(_):

    @shape_classifier.register_classifier("mode_2")
    def mode_2_classifier(inputs, params):
        return inputs

    classifier_func = shape_classifier._classifiers.get("mode_2")
    inputs = [{"shape": (-1, -1), "range": [(2, 10), (1, None)]}]
    ret = classifier_func(inputs, None)

    return inputs == ret


@add_cust_test_func(ut_case)
def test_get_classifier(_):

    @shape_classifier.register_classifier("mode_3")
    def mode_3_classifier(inputs):
        return inputs

    classifier_func = shape_classifier._get_classifier("mode_3")

    return classifier_func is not None


if __name__ == "__main__":
    for k, v in ut_case._case_info_map.items():
        # if not k == "test_calc_padding_softmax":
        #     continue

        try:
            ret = v.test_func(None)
        except Exception:
            import traceback
            print(f"\033[93mException: {k}\033[0m")
            print(traceback.format_exc())
            continue

        if ret:
            print(f"\033[92mPASS: {k}\033[0m")
        else:
            print(f"\033[91mFAIL: {k}\033[0m")
