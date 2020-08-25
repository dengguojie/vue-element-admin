"""
error manager util
"""
import json
import os


def get_error_message(args):
    """
    :param args: dict
        keys in dict must be in accordance with xlsx
    :return: string
            formatted message
    """
    error_code = args.get("errCode")
    with open("{}/errormanager.json".format(
            os.path.dirname(os.path.abspath(__file__)))) as file_content:
        data = json.load(file_content)
        error_dict = {}
        for error_message in data:
            error_dict[error_message['errCode']] = error_message
        error_json = error_dict
    error_stmt = error_json.get(error_code)
    if error_stmt is None:
        return "errCode = {} has not been defined".format(error_code)
    arg_list = error_stmt.get("argList").split(",")
    arg_value = []
    for arg_name in arg_list:
        if arg_name == "op_name":
            arg_value.append("")
        else:
            arg_value.append(args.get(arg_name.strip()))
    msg = error_json.get(error_code).get("errMessage") % tuple(arg_value)
    return msg
