# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Generate Result HTML
"""
import os
import csv
import sys
import pathlib
from datetime import datetime
from typing import List, NoReturn, Tuple


def extract_results(npu_results: Tuple) -> List:
    """Extract testcase results"""
    fail_case: List = []
    succ_case: List = []
    for result in npu_results:
        op_res = (result["testcase_name"],
                  result["network_name"],
                  "lightgreen" if result["precision_status"] == "PASS" else "pink",
                  result["dyn_precision"],
                  result["dyn_perf_us"],
                  result["stc_inputs"],
                  result["stc_outputs"],
                  result["dyn_input_dtypes"],
                  result["dyn_input_formats"])
        if result["precision_status"] == "PASS":
            succ_case.append(op_res)
        else:
            fail_case.append(op_res)
    fail_case = sorted(fail_case, key=lambda x: x[0])
    succ_case = sorted(succ_case, key=lambda x: x[0])
    return fail_case + succ_case


def generate_html(dt_string: str, my_path: str, output: str, operator_results: List) -> NoReturn:
    with open(pathlib.Path(my_path, "html_templates/click_me_for_result_template.html"), encoding="utf-8") as f:
        html_template = f.read()

    html_template = html_template.replace("%time", dt_string)
    tds = "\n".join(tuple(("<tr>" +
                           "<td>%s</td>" +
                           "<td>%s</td>" +
                           "<td style=\"background-color: %s\" class=\"precision\">%s</td>" +
                           "<td class=\"npu_perf\">%s</td>" +
                           "<td>%s</td>" +
                           "<td>%s</td>" +
                           "<td>%s</td>" +
                           "<td>%s</td>" +
                           "</tr>") % op_td for op_td in operator_results))
    html_template = html_template.replace("%op_tds", tds)
    with open(output, mode="w+", encoding="utf-8") as f:
        f.write(html_template)


def main(result_csv: str, output: str) -> NoReturn:
    # My Path
    my_path = os.path.dirname(os.path.realpath(__file__))

    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
    print("current date and time =", dt_string)

    print("Reading results...")
    with open(result_csv, encoding="UTF-8", newline="") as f:
        results = tuple(csv.DictReader(f))

    if not results:
        print("no result exists in csv file...")
        sys.exit(1)

    print("Extracting results...")
    operator_results = extract_results(results)

    print("Generating html elements...")
    generate_html(dt_string, my_path, output, operator_results)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: python3", sys.argv[0], "<result.csv> [output_path]")
        sys.exit(1)
    output = "click_me_for_result.html"
    if len(sys.argv) > 2:
        try:
            true_path = os.path.realpath(sys.argv[2])
        except Exception:
            print("Invalid output file")
            sys.exit(1)
        else:
            output = true_path
    main(sys.argv[1], output)
