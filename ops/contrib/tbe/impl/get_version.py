import sys


def get_tbe_version():
    """
    C3X: including C30/C31/C32/C33
    C7X: including C73/C75...
    Unsupport: not including C3x and C7x
    :return:
    """
    try:
        from te import tik
        return tik, "C7x"
    except ImportError:
        try:
            import tik
            return tik, "c3x"
        except ImportError:
            sys.stdout.write("\r Unsupport tbe platform!")
            return None, "Unsupport"
