#! /urs/bin/env python
# *—* coding=utf-8 *-*

import sys
import os
import re


def get_line_coverage(html_file, threshold):
    print( "the html path is %s: " % html_file)
    if not os.path.exists(html_file):
        print("index.html not exists")
        return False

    with open(html_file, 'r') as index_html:
        index_html_content = index_html.read()

    pattern = r'<span class="pc_cov">(.*?)</span>'
    coverage_data = re.findall(pattern, index_html_content)
    print( "line coverage_data is %s"% coverage_data)
    line_coverage = "0.0%"
    if len(coverage_data) > 0:
        line_coverage = coverage_data[0].replace("%", "")
    print("line coverage is %s" % line_coverage)
    coverage_number = int(line_coverage.split(".")[0])
    threshold_num = int(threshold)
    if coverage_number >= threshold_num:
        print("line coverage is equal or over %d, coverage result is succ!" % threshold_num)
        return True
    else:
        print("line coverage is below %d, coverage result is fail!" % threshold_num)
        return False


def get_inc_line_coverage(html_file, threshold):
    print("the html path is %s: " % html_file)
    if not os.path.exists(html_file):
        print("index.html not exists")
        return False

    with open(html_file, 'r') as index_html:
        index_html_content = index_html.read()

    pattern = r'<li><b>Coverage</b>(.*?)</li>'
    coverage_data = re.findall(pattern, index_html_content)
    if len(coverage_data) == 0:
        print("No lines with coverage information in this diff. build succ,return True")
        return True

    line_coverage = coverage_data[0].replace(": ", "")
    print("line coverage is %s" % line_coverage)
    coverage_number = int(line_coverage.replace("%", "").strip())
    threshold_num = int(threshold)
    if coverage_number >= threshold_num:
        print("line coverage is equal or over %d, coverage result is succ!" % threshold_num)
        return True
    else:
        print("line coverage is below %d, coverage result is fail!" % threshold_num)
        return False


def main():
    print(("parameters length is : ", len(sys.argv)))
    print(("parameters is :", str(sys.argv)))

    html_path = sys.argv[1]
    threshold = sys.argv[2]
    full_coverage = sys.argv[3]

    if "true" == full_coverage:
        html_file = os.path.join(html_path, 'index.html')
        cover_result = get_line_coverage(html_file, threshold)
    else:
        html_file = os.path.join(html_path, 'report.html')
        cover_result = get_inc_line_coverage(html_file, threshold)

    if cover_result:
        sys.exit(0)
    else:
        sys.exit(-1)


if __name__ == '__main__':
    main()
