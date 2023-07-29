from mlutils.common import *

"输出函数"
"debug print"
def dprint(*values,debug=debug):
    if debug:
        print(*values)

"输出函数"
"debug print"
def dprint(*values,debug=debug):
    if debug:
        print(*values)


def stdformat(x,gap,place="M"):
    """
    gap:占位宽度
    place:L,M，R<-->左，中,右
    """
    "字符串为中文则填充中文空格"
    if isinstance(x,str) and _is_contains_chinese(x):
        return '{0:{1}^{2}}'.format(x,chr(12288),gap)
    if isinstance(x,np.ndarray):
        if x.dtype in [np.float,np.float16,np.float32,np.float64]:
            x=np.round(x,2)
    return format(str(x),"^"+str(gap))
def _is_contains_chinese(x):
    for _char in x:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False
"计数print"
class debug_print_tool():
    def __init__(self,namespace,debug=debug) -> None:
        self.debug=debug
        self.step=0
        self.namesapce=namespace
    def __call__(self,additional_str="") :
        self.step+=1
        if self.debug:
            print(f"{self.namesapce} step {self.step} {additional_str}")

"计时类"
class timer_tool():
    def __init__(self,module_name,_debug=debug) -> None:
        self.module_name=module_name
        self.debug=_debug
        self.time_info_dict={}
        self.start()
        pass
    def start(self):
        self.time=time.time()
    def end(self,message):
        cost_time=time.time()-self.time
        if message in self.time_info_dict.keys():
            self.time_info_dict[message]+=cost_time
        else:
            self.time_info_dict[message]=cost_time

        if self.debug:
            print(f"{self.module_name} {message} cost {cost_time}")
            
    def end_and_start(self,message):
        self.end(message)
        self.start()
    def analyze(self):
        if self.debug:
            total_time=0
            for key in self.time_info_dict:
                total_time+=self.time_info_dict[key]
            print(f"{self.module_name} total time is {total_time}")
            for key in self.time_info_dict:
                print(f"{key} cost {self.time_info_dict[key]} {round(self.time_info_dict[key]/total_time*100,1)}%")