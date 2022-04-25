from enum import Enum
from typing import Dict, Sequence, Optional, NoReturn, List


class DavinciV100PipelineType(Enum):
    PID_PIPELINE_FLOWCTRL = 0
    PID_PIPELINE_SCALAR = 1
    PID_PIPELINE_SCALARLDST = 2
    PID_PIPELINE_MTE1 = 3
    PID_PIPELINE_MTE2 = 4
    PID_PIPELINE_VEC = 5
    PID_PIPELINE_VECTOR = 5
    PID_PIPELINE_CUBE = 6
    PID_PIPELINE_MTE3 = 7
    PID_PIPELINE_ICACHE = 8
    PID_PIPELINE_ICACHEREAD = 9


class DavinciV200PipelineType(Enum):
    PID_PIPELINE_FLOWCTRL = 0
    PID_PIPELINE_SCALAR = 1
    PID_PIPELINE_SCALARLDST = 2
    PID_PIPELINE_MTE1 = 3
    PID_PIPELINE_MTE2 = 4
    PID_PIPELINE_VEC0 = 5
    PID_PIPELINE_VECTOR0 = 5
    PID_PIPELINE_VECTOR = 5
    PID_PIPELINE_CUBE = 6
    PID_PIPELINE_MTE3 = 7
    PID_PIPELINE_ICACHE = 8
    PID_PIPELINE_ICACHEREAD = 9


class DavinciV220PipelineType(Enum):
    PID_PIPELINE_FC = 0
    PID_PIPELINE_SCALAR = 1
    PID_PIPELINE_SCALARLDST = 2
    PID_PIPELINE_MTE1 = 3
    PID_PIPELINE_MTE2 = 4
    PID_PIPELINE_VEC = 5
    PID_PIPELINE_CUBE = 6
    PID_PIPELINE_MTE3 = 7
    PID_PIPELINE_ICACHE = 8
    PID_PIPELINE_ICACHEREAD = 9
    PID_PIPELINE_PUSHQ = 10
    PID_PIPELINE_RVECSU = 11
    PID_PIPELINE_RVECEX = 12
    PID_PIPELINE_RVECLD = 13
    PID_PIPELINE_RVECST = 14
    PID_PIPELINE_FLOWCTRL = 15
    PID_PIPELINE_FLOWCONTROL = 16
    PID_PIPELINE_FIXP = 17


class DavinciV300PipelineType(Enum):
    PID_PIPELINE_FLOWCTRL = 0
    PID_PIPELINE_SCALAR = 1
    PID_PIPELINE_SCALARLDST = 2
    PID_PIPELINE_MTE1 = 3
    PID_PIPELINE_MTE2 = 4
    PID_PIPELINE_VEC = 5
    PID_PIPELINE_CUBE = 6
    PID_PIPELINE_MTE3 = 7
    PID_PIPELINE_ICACHE = 8
    PID_PIPELINE_ICACHEREAD = 9
    PID_PIPELINE_PUSHQ = 10
    PID_PIPELINE_RVECSU = 11
    PID_PIPELINE_RVECEX = 12
    PID_PIPELINE_RVECLD = 13
    PID_PIPELINE_RVECST = 14


class ChromeColorName(Enum):
    BLUE = "rail_response"
    BLACK = "black"
    GREY = "grey"
    WHITE = "white"
    YELLOW = "yellow"
    OLIVE = "olive"
    GREEN = "thread_state_running"
    DARK_GREEN = "rail_load"
    RED = "rail_animation"
    PURPLE = "detailed_memory_dump"
    ORANGE = "thread_state_iowait"
    DARK_PURPLE = "thread_state_uninterruptible"


class ChromeEventType(Enum):
    DURATION_BEGIN = "B"  # Begin
    DURATION_END = "E"  # End
    COMPLETE = "X"  # Complete event = B+E
    INSTANT = "i"
    COUNTER = "C"
    ASYNC_START = "b"  # Nestable start
    ASYNC_INSTANT = "n"  # Nestable instant
    ASYNC_END = "e"  # Nestable end
    FLOW_START = "s"
    FLOW_STEP = "t"
    FLOW_END = "f"
    SAMPLE = "P"
    OBJECT_CREATED = "N"
    OBJECT_SNAPSHOT = "O"
    OBJECT_DESTROY = "D"
    METADATA = "M"
    MEMORY_DUMP_GLOBAL = "V"
    MEMORY_DUMP_PROCESS = "v"
    MARK = "R"
    CLOCK_SYNC = "c"
    CONTEXT_START = "("
    CONTEXT_END = ")"


INSTR_COLOR_MAP: Dict[Enum, ChromeColorName] = {
    DavinciV100PipelineType.PID_PIPELINE_ICACHE: ChromeColorName.BLACK,
    DavinciV100PipelineType.PID_PIPELINE_ICACHEREAD: ChromeColorName.GREY,
    DavinciV100PipelineType.PID_PIPELINE_MTE3: ChromeColorName.BLUE,
    DavinciV100PipelineType.PID_PIPELINE_MTE2: ChromeColorName.RED,
    DavinciV100PipelineType.PID_PIPELINE_MTE1: ChromeColorName.YELLOW,
    DavinciV100PipelineType.PID_PIPELINE_VEC: ChromeColorName.PURPLE,
    DavinciV100PipelineType.PID_PIPELINE_VECTOR: ChromeColorName.PURPLE,
    DavinciV100PipelineType.PID_PIPELINE_CUBE: ChromeColorName.OLIVE,
    DavinciV100PipelineType.PID_PIPELINE_FLOWCTRL: ChromeColorName.ORANGE,
    DavinciV100PipelineType.PID_PIPELINE_SCALAR: ChromeColorName.GREEN,
    DavinciV100PipelineType.PID_PIPELINE_SCALARLDST: ChromeColorName.DARK_GREEN,

    DavinciV200PipelineType.PID_PIPELINE_ICACHE: ChromeColorName.BLACK,
    DavinciV200PipelineType.PID_PIPELINE_ICACHEREAD: ChromeColorName.GREY,
    DavinciV200PipelineType.PID_PIPELINE_MTE3: ChromeColorName.BLUE,
    DavinciV200PipelineType.PID_PIPELINE_MTE2: ChromeColorName.RED,
    DavinciV200PipelineType.PID_PIPELINE_MTE1: ChromeColorName.YELLOW,
    DavinciV200PipelineType.PID_PIPELINE_VEC0: ChromeColorName.PURPLE,
    DavinciV200PipelineType.PID_PIPELINE_VECTOR0: ChromeColorName.PURPLE,
    DavinciV200PipelineType.PID_PIPELINE_CUBE: ChromeColorName.OLIVE,
    DavinciV200PipelineType.PID_PIPELINE_FLOWCTRL: ChromeColorName.ORANGE,
    DavinciV200PipelineType.PID_PIPELINE_SCALAR: ChromeColorName.GREEN,
    DavinciV200PipelineType.PID_PIPELINE_SCALARLDST: ChromeColorName.DARK_GREEN,

    DavinciV220PipelineType.PID_PIPELINE_ICACHE: ChromeColorName.BLACK,
    DavinciV220PipelineType.PID_PIPELINE_ICACHEREAD: ChromeColorName.GREY,
    DavinciV220PipelineType.PID_PIPELINE_MTE3: ChromeColorName.BLUE,
    DavinciV220PipelineType.PID_PIPELINE_MTE2: ChromeColorName.RED,
    DavinciV220PipelineType.PID_PIPELINE_MTE1: ChromeColorName.YELLOW,
    DavinciV220PipelineType.PID_PIPELINE_VEC: ChromeColorName.PURPLE,
    DavinciV220PipelineType.PID_PIPELINE_CUBE: ChromeColorName.OLIVE,
    DavinciV220PipelineType.PID_PIPELINE_FC: ChromeColorName.ORANGE,
    DavinciV220PipelineType.PID_PIPELINE_FIXP: ChromeColorName.BLUE,
    DavinciV220PipelineType.PID_PIPELINE_SCALAR: ChromeColorName.GREEN,
    DavinciV220PipelineType.PID_PIPELINE_SCALARLDST: ChromeColorName.DARK_GREEN,

    DavinciV220PipelineType.PID_PIPELINE_PUSHQ: ChromeColorName.ORANGE,
    DavinciV220PipelineType.PID_PIPELINE_RVECSU: ChromeColorName.ORANGE,
    DavinciV220PipelineType.PID_PIPELINE_RVECEX: ChromeColorName.ORANGE,
    DavinciV220PipelineType.PID_PIPELINE_RVECLD: ChromeColorName.ORANGE,
    DavinciV220PipelineType.PID_PIPELINE_RVECST: ChromeColorName.ORANGE,
    DavinciV220PipelineType.PID_PIPELINE_FLOWCTRL: ChromeColorName.ORANGE,
    DavinciV220PipelineType.PID_PIPELINE_FLOWCONTROL: ChromeColorName.ORANGE,
}


class ChromeInstantEventScope(Enum):
    GLOBAL = "g"
    PROCESS = "p"
    THREAD = "t"


class BaseChromeEvent:
    def __init__(self,
                 name: str,
                 categories: Sequence[str],
                 event_type: int,
                 timestamp: int,
                 pid: int,
                 tid: int,
                 tts: Optional[int] = None,
                 args: Optional[dict] = None,
                 cname: Optional[ChromeColorName] = None):
        self.data = {"name": name,
                     "cat": ','.join(categories),
                     "ph": event_type,
                     "ts": timestamp,
                     "pid": pid,
                     "tid": tid,
                     }
        if tts:
            self.data["tts"] = tts
        if args:
            self.data["args"] = args
        if cname:
            self.data["cname"] = cname.value

    def to_dict(self):
        return self.data

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return self.__str__()

    def set_color(self, cname: ChromeColorName):
        self.data["cname"] = cname.value
        return self


class ChromeDurationEvent(BaseChromeEvent):
    def __init__(self, name,
                 categories: Sequence[str],
                 ts: int,
                 pid: int,
                 tid: int,
                 is_start: bool,
                 args: Optional[Dict[str, str]] = None,
                 cname: Optional[ChromeColorName] = None):
        super().__init__(name,
                         categories,
                         ChromeEventType.DURATION_BEGIN.value if is_start else ChromeEventType.DURATION_END.value,
                         ts,
                         pid,
                         tid,
                         args=args,
                         cname=cname)


class ChromeMetadataEvent(BaseChromeEvent):
    def __init__(self, name: str, pid: int, tid: int, arg_name: str, value: str):
        super().__init__(name, (), ChromeEventType.METADATA.value, 0, pid, tid, args={arg_name: value})


class ChromeInstantEvent(BaseChromeEvent):
    def __init__(self,
                 name,
                 categories: Sequence[str],
                 ts: int,
                 pid: int,
                 tid: int,
                 scope: ChromeInstantEventScope,
                 args: Optional[dict] = None,
                 cname: Optional[ChromeColorName] = None):
        super().__init__(name,
                         categories,
                         ChromeEventType.INSTANT.value,
                         ts,
                         pid,
                         tid,
                         args=args,
                         cname=cname)
        self.data["s"] = scope.value


class ChromeCompleteEvent(BaseChromeEvent):
    def __init__(self, name, categories: Sequence[str], ts: int, pid: int, tid: int, dur: int,
                 args: Optional[Dict[str, str]] = None,
                 cname: Optional[ChromeColorName] = None):
        super().__init__(name, categories, ChromeEventType.COMPLETE.value, ts, pid, tid, args=args,
                         cname=cname)
        self.data["dur"] = dur


class ChromeAsyncEvent(BaseChromeEvent):
    def __init__(self, name,
                 categories: Sequence[str],
                 ts: int,
                 pid: int,
                 tid: int,
                 _id: int,
                 is_start: bool = True,
                 args: Optional[Dict[str, str]] = None,
                 cname: Optional[ChromeColorName] = None):
        super().__init__(name,
                         categories,
                         ChromeEventType.ASYNC_START.value if is_start else ChromeEventType.ASYNC_END.value,
                         ts,
                         pid,
                         tid,
                         args=args,
                         cname=cname)
        self.data["id"] = _id


class ChromeTraceJson:
    def __init__(self, timeunit: str = "ms"):
        self.traceEvents = []
        self.timeunit = timeunit
        self.end_event: Optional[BaseChromeEvent] = None
        self.start_event: Optional[BaseChromeEvent] = None

    def addEvent(self, event: BaseChromeEvent) -> NoReturn:
        self.traceEvents.append(event.to_dict())

    def get_all_events(self) -> List[dict]:
        res = self.traceEvents.copy()
        if self.end_event:
            res.append(self.end_event.to_dict())
        if self.start_event:
            res.append(self.start_event.to_dict())
        return res

    def get(self):
        return {'schemaVersion': 1,
                'displayTimeUnit': self.timeunit,
                'traceEvents': self.get_all_events()}
