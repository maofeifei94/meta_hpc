from concurrent.futures import thread
from dataclasses import dataclass
import numpy as np
from multiprocessing import Queue as process_Queue
from multiprocessing import Lock as process_Lock
from queue import Queue as thread_Queue
import threading
from threading import Lock as thread_Lock
from multiprocessing import Manager
from multiprocessing import shared_memory
import time
import copy
debug=True
def state_dict_to_np(sd):
    """
    sd:state_dict
    """
    sd_numpy={}
    for key in sd.keys():
        sd_numpy[key]=sd[key].numpy()
    return sd_numpy
def set_state_dict_from_np(sd,sd_numpy):
    for key in sd:
        sd[key].set_value(sd_numpy[key])
    return sd

def cal_shm_dict_size(shm_dict):
    free_num=0
    used_num=0
    for key in shm_dict.keys():
        if 'free' in key:
            free_num+=len(shm_dict[key].keys())
        elif 'used' in key:
            used_num+=len(shm_dict[key].keys())
    return ["total",free_num+used_num,"free",free_num,"used",used_num]


"""
共享内存数据，记录内存块的name，shape，dtype信息
通过调用np_data来获取numpy数组
"""
class SharedMemData():
    def __init__(self,shm_name,shape,dtype) -> None:
        self.shm_name=shm_name
        self.shape=shape
        self.dtype=dtype
        self.mem_retainer=[]#SharedMemory若失去引用会立即回收ndarray失去内存空间报错，因此要用self.mem_retainer引用
    def np_data(self):
        shm=shared_memory.SharedMemory(name=self.shm_name)
        data=np.ndarray(self.shape,self.dtype,shm.buf)
        self.mem_retainer.append(shm)#SharedMemory若失去引用会立即回收,ndarray失去内存空间报错，因此要用self.mem_retainer引用
        return data

"""
多进程数据流架构采用管道-过滤器架构

Task指定filter调用函数：filter_name.func_name(cache,**kwargs)，kwargs为参数（不存储数据）
Cache保存函数所需数据的内存块索引
TaskFlow定义任务流[Task1，Task2，Task3，...]，并用Cache存储沿途需要保存的数据
FlowGenertor为调度中心，生产TaskFlow
ProcessFilter为处理模块
"""
class Task():
    """
    Task:指定filter,执行func(cache,**kwargs),其中kwargs是参数，不包含数据。cache是flow的数据缓存。
    """
    def __init__(self,filter_name,func_name:str,**kwargs):
        self.filter_name=filter_name
        self.func_name=func_name
        self.kwargs=kwargs
    def __repr__(self):
        
        # kwargs_keys=str(tuple(self.kwargs.keys())).replace("'","")[1:-1]
        # kwargs_keys=str(self.kwargs)
        kwargs_str=""
        for key in self.kwargs.keys():
            kwargs_str+=f"{key}={self.kwargs[key]},"
        kwargs_str=kwargs_str[:-1]
        return f"Task-->{self.filter_name}.{self.func_name}({kwargs_str})"

class Cache():
    def __init__(self,share_info,lock,**kwargs) -> None:
        self.share_info=share_info
        self.lock=lock
        self.shm_dict=self._np_to_shm(kwargs)
    def update(self,cache_update:dict):
        assert isinstance(cache_update,dict)
        self.shm_dict.update(self._np_to_shm(cache_update))

    def release(self):
        "在TaskFlow最后一步释放所占用的内存块地址"
        self.lock.acquire()
        self._release_item(self.shm_dict)
        self.lock.release()
        # print("cache release,share_info=",cal_shm_dict_size(self.share_info))
    def _release_item(self,x):
        if isinstance(x,SharedMemData):
            shm=x
            shape_dtype_used=str(shm.shape)+str(shm.dtype)+'used'
            shape_dtype_free=str(shm.shape)+str(shm.dtype)+'free'
            used_copy=self.share_info[shape_dtype_used]
            free_copy=self.share_info[shape_dtype_free]
            used_copy.pop(shm.shm_name)
            free_copy[shm.shm_name]=shm.shm_name
            self.share_info[shape_dtype_used]=used_copy
            self.share_info[shape_dtype_free]=free_copy
        if isinstance(x,dict):
            {self._release_item(x[key]) for key in x.keys()}
    def keys(self):
        return self.shm_dict.keys()
    def np_dict(self):
        return self._shm_to_np(self.shm_dict)
    def __getitem__(self,key):
        "获取key对应的value数据时，自动转换为numpy数组"
        return self._shm_to_np(self.shm_dict[key])
    def __setitem__(self,key,value):
        "存储key时，自动将numpy数组转换成shared_memory"
        self.shm_dict[key]=self._np_to_shm(value)
    def __repr__(self) -> str:
        def str_shm_dict(x):
            if isinstance(x,SharedMemData):
                return x.shm_name
            if isinstance(x,dict):
                return str({key:str_shm_dict(x[key]) for key in x.keys()})
        return str_shm_dict(self.shm_dict)
    def _shm_to_np(self,x):
        if isinstance(x,SharedMemData):
            return x.np_data()
        if isinstance(x,dict):
            return {key:self._shm_to_np(x[key]) for key in x.keys()}
        else:
            return x
    def _np_to_shm(self,x):
        if isinstance(x,np.ndarray):
            return self._ndarray_to_shmdata(x)
        if isinstance(x,dict):
            return {key:self._np_to_shm(x[key]) for key in x.keys()}#递归
        else:
            return x
    def _ndarray_to_shmdata(self,data:np.ndarray) -> SharedMemData:
        """
        将numpy数组转换成shared_memory
        share_info格式为{
            str(shape)+str(dtype)+'used'/'free':{
                name:name
            }
        }
        """
        assert isinstance(data,np.ndarray)
        shape_dtype_used=str(data.shape)+str(data.dtype)+'used'
        shape_dtype_free=str(data.shape)+str(data.dtype)+'free'
        "分配内存块"
        self.lock.acquire()
        if not shape_dtype_used in self.share_info.keys():
            self.share_info[shape_dtype_used]={}
        if not shape_dtype_free in self.share_info.keys():
            self.share_info[shape_dtype_free]={}

        "Manager的dict是一个proxy代理，取出的value是一种复制，修改value并不能更新dict内容"
        used_copy=self.share_info[shape_dtype_used]
        free_copy=self.share_info[shape_dtype_free]
                    
        if len(free_copy.keys())>0:
            "有空闲内存块"
            shm_name=list(free_copy.keys())[0]
            shm=shared_memory.SharedMemory(name=shm_name,size=data.nbytes)
            free_copy.pop(shm_name)
            used_copy.update({shm_name:shm_name})
        else:
            "没有空闲内存块"
            shm=shared_memory.SharedMemory(create=True,size=data.nbytes)
            shm_name=shm.name
            used_copy.update({shm_name:shm_name})
        self.share_info[shape_dtype_used]=used_copy
        self.share_info[shape_dtype_free]=free_copy
        self.lock.release()
        "写内存"
        buf_data=np.ndarray(data.shape,data.dtype,buffer=shm.buf)
        buf_data[:]=data
        return SharedMemData(shm_name,data.shape,data.dtype)
class TaskFlow():
    """
    按顺序执行tasklist的每一个Task
    cache储存task返回的cache_update
    priority:high,mid,low，优先级
    """
    def __init__(
            self,
            tasklist:list,
            cache:Cache,
            priority:str
        ):

        self.tasklist=tasklist
        self.current_index=0
        self.priority=priority

        self.current_task=self.tasklist[self.current_index]
        self.cache=cache

        self.info_list=[]

        self.left_steps=len(self.tasklist)
    def collect_info(self,info):
        self.info_list.append(info)
    def after_filt(self,cache_update):
        if not isinstance(cache_update,dict):
            raise ValueError(f'cache_update={cache_update},type(cache_update)!=dict')
        self.left_steps-=1
        self.current_index+=1
        if self.left_steps<=0:
            "Flow结束，释放所占用的共享内存块"

            if self.priority in ['high']:
                print(self.__repr__())
            # print(self.__repr__())
            self.cache.release()
        else:
            self.current_task=self.tasklist[self.current_index]
            self.cache.update(cache_update)
    def __repr__(self):
        content=f"******TaskFlow at step {self.current_index}/{len(self.tasklist)}******"
        content+="\n cache="+str(tuple(self.cache.keys())).replace("'","")
        for i,task in enumerate(self.tasklist+["finish and free cache"]):
            if i==self.current_index:
                content+=f"\n@step {i}:"+str(task)
            else:
                content+=f"\n step {i}:"+str(task)
            
            if i<=len(self.info_list)-1:
                content+=self.info_list[i]
        content+="\n***********************************"
        return content

class FlowGenerator():
    """
    任务产生器
    起到调节任务比例的作用。
    比如某一个模块训练的比较好了,就根据它的Loss值调节训练间隔,将计算资源分配给训练得不好的模块
    """
    def __init__(self,queue_dict,name) -> None:
        self.queue_dict=queue_dict
        self.name=name
        self.queue=self.queue_dict[self.name]
        self._run()
    def _send(self,flow:TaskFlow):
        """
        根据flow的信息,将数据发送到下一个进程专属的Queue中
        """
        if flow.left_steps>0:
            self.queue_dict[flow.current_task.filter_name].send(flow)
    def _run(self):
        """
        产生任务
        """
        pass

class ProcessFilter():
    """
    Filter以下的_send,_filt,_run函数定义了处理系统的架构
    自定义模块只需要写function函数
    其标准为:func(cache:dict,**kwargs)->cache_update:dict
    cache存储数据，kwargs提供参数
    """
    def __init__(self,queue_dict,name) -> None:
        self.queue_dict=queue_dict
        self.name=name
        self.flow_queue=self.queue_dict[self.name]
        self._run()
    # 功能函数示例
    # def func(self,cache:dict,**kwargs):
    #     cache_update={"filter_name+func_name+target_name":data}
    #     return cache_update
    def _filt(self,flow:TaskFlow):
        """
        step1:处理数据
        step2:更新flow
        step3:flow送到下一个filter
        """
        t1=time.time()
        # print(self.name,flow.current_task.func_name,'start')
        cache_update=getattr(self,flow.current_task.func_name)(flow.cache,**flow.current_task.kwargs)
        # print(self.name,flow.current_task.func_name,'finish')
        cost_time=time.time()-t1
        flow.collect_info(f"cost {cost_time}")
        # fname=flow.current_task.func_name
        flow.after_filt(cache_update)
        if flow.left_steps>0:
            #下一个filter依然是本身，丢入queue时若已满会造成卡死，因此直接运行递归
            if flow.current_task.filter_name==self.name:
                self._filt(flow)
            else:
                self.queue_dict[flow.current_task.filter_name][flow.priority].put(flow)
    def _run(self):
        """
        循环函数,不断地接受Queue的数据并处理
        """
        while 1:
            recv_task=False
            for priority in ['high','mid','low']:
                if self.flow_queue[priority].qsize()>0:
                    recv_task=True
                    flow=self.flow_queue[priority].get()
                    self._filt(flow)
            if not recv_task:
                time.sleep(0.05)



    