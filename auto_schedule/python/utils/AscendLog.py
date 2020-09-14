import ctypes
import os
import threading


class LogMap:

    class LogMapError(TypeError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.LogMapError("Can't rebind (%s)", name)
        self.__dict__[name] = value


class AscendLog:

    """ Ascend Log Instance
    usage example:

        from te.utils.AscendLog import LOGGER
        LOGGER.set_level(LOGGER.module.cce, LOGGER.level.info,
                    LOGGER.level.event_enable)
        LOGGER.info(LOGGER.module.cce,"this is test .")
    """
    _instance_lock = threading.Lock()

    def __init__(self):
        """ Initialize AscendLog"""
        self.log = None
        self.module = LogMap()
        self.module.cce = 8
        self.module.app = 33
        self.module.aicpu = 36
        self.module.mind_data = 40
        self.module.mind_board = 41
        self.module.mind_engine = 42
        self.module.acl = 48
        self.module.tbe = 57
        self.module.fvr = 58
        self.level = LogMap()
        self.level.debug = 0
        self.level.info = 1
        self.level.warning = 2
        self.level.error = 3
        self.level.null = 4
        self.level.event_enable = 1
        self.level.event_disable = 0
        try:
            self.log = ctypes.cdll.LoadLibrary('libslog.so')
        except OSError as err:
            ld_path = os.getenv('LD_LIBRARY_PATH')
            if ld_path is None:
                print('[Warning]Can not find libslog.so')
                return
            path_list  = ld_path.split(':')
            for path in path_list:
                target_path = os.path.join(path, 'libslog.so')
                if os.path.isfile(target_path):
                    self.log = ctypes.cdll.LoadLibrary(target_path)
                    break
            if self.log is None:
                print('[Warning]Can not find libslog.so')

    def __new__(cls, *args, **kwargs):
        if not hasattr(AscendLog, "_instance"):
            with AscendLog._instance_lock:
                if not hasattr(AscendLog, "_instance"):
                    AscendLog._instance = object.__new__(cls)
        return AscendLog._instance

    def debug(self, module, fmt):
        """ print debug log

        Parameters
        ----------
        module: module id, eg: CCE
        fmt: log content

        Returns:
        ---------
        None
        """
        if self.log is None:
            return
        self.log.DlogDebugInner(ctypes.c_int(module),
                        ctypes.c_char_p(fmt.encode("utf-8")))

    def info(self, module, fmt):
        """ print info log

        Parameters
        ----------
        module: module id, eg: CCE
        fmt: log content

        Returns:
        ----------
        None
        """
        if self.log is None:
            return
        self.log.DlogInfoInner(ctypes.c_int(module),
                       ctypes.c_char_p(fmt.encode("utf-8")))

    def warn(self, module, fmt):
        """ print warning log

        Parameters
        ----------
        module: module id, eg: CCE
        fmt: log content

        Returns:
        ----------
        None
        """
        if self.log is None:
            return
        self.log.DlogWarnInner(ctypes.c_int(module),
                       ctypes.c_char_p(fmt.encode("utf-8")))

    def error(self, module, fmt):
        """ print error log

        Parameters
        ----------
        module: module id, eg: CCE
        fmt: log content

        Returns:
        ----------
        None
        """
        if self.log is None:
            return
        self.log.DlogErrorInner(ctypes.c_int(module),
                        ctypes.c_char_p(fmt.encode("utf-8")))

    def event(self, module, fmt):
        """ print event log

        Parameters
        ----------
        module: module id, eg: CCE
        fmt: log content

        Returns:
        ----------
        None
        """
        if self.log is None:
            return
        self.log.DlogEventInner(ctypes.c_int(module),
                        ctypes.c_char_p(fmt.encode("utf-8")))

    def set_level(self, module, level, event):
        """ set log level

        Parameters
        ----------
        module: module id, eg: CCE
        level: level id, include: debug, info, warning, error, null
        event: event switch, enable: event_enable; disable: event_disable;

        Returns:
        ----------
        None
        """
        if self.log is None:
            return
        self.log.dlog_setlevel(ctypes.c_int(module), ctypes.c_int(level),
                        ctypes.c_int(event))

LOGGER = AscendLog()
