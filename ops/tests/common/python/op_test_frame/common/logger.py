DEBUG = "DEBUG"
INFO = "INFO"
WARN = "WARN"
ERROR = "ERROR"


def log(level, msg):
    print("[Op Test] [%s] %s" % (level, msg))


def log_warn(msg):
    log(WARN, msg)


def log_debug(msg):
    log(DEBUG, msg)


def log_info(msg):
    log(INFO, msg)


def log_err(msg):
    log(ERROR, msg)
