from multiprocessing import Queue,Process,Manager
from brain.hyperparam.brain_guess_box_hp import brain_hyperparam
import numpy as np


def run():
    "共享内存调度信息"
    share_info=Manager().dict()
    "进程锁，用来确保share_info安全"
    lock=Manager().Lock()
    def process_flowg(queue_dict,process_n):
        print(f"start {process_n}")
        from brain.brain_guess_box import FlowGenerator,DataFilter
        FlowGenerator(queue_dict,'flowgenerator',share_info,lock)
    def process_envf(queue_dict,process_n):
        print(f"start {process_n}")
        from brain.brain_guess_box import EnvFilter
        EnvFilter(queue_dict,'envfilter')
    def process_ppof(queue_dict,process_n):
        print(f"start {process_n}")
        from brain.brain_guess_box import PPOFilter
        PPOFilter(queue_dict,'ppofilter')
        print("PPOFilter init done")
    def process_dataf(queue_dict,process_n):
        print(f"start {process_n}")
        from brain.brain_guess_box import FlowGenerator,DataFilter
        DataFilter(queue_dict,'datafilter')
    def process_caef(queue_dict,process_n):
        print(f"start {process_n}")
        from brain.brain_guess_box import FlowGenerator,DataFilter
        CaeFilter(queue_dict,'caefilter')
    def process_ecf(queue_dict,process_n):
        print(f"start {process_n}")
        from brain.brain_guess_box import EcFilter
        EcFilter(queue_dict,'ecfilter')
    def process_finishf(queue_dict,process_n):
        print(f"start {process_n}")
        from brain.brain_guess_box import FinishFilter
        FinishFilter(queue_dict,'finishfilter')
    
    def get_queue():
        return Queue(maxsize=brain_hyperparam.queue_max_size)
    queue_dict={
        'flowgenerator':get_queue(),
        'envfilter':{'high':get_queue(),"mid":get_queue(),"low":get_queue()},
        'ppofilter':{'high':get_queue(),"mid":get_queue(),"low":get_queue()},
        'datafilter':{'high':get_queue(),"mid":get_queue(),"low":get_queue()},
        # 'caefilter':{'high':get_queue(),"mid":get_queue(),"low":get_queue()},
        'ecfilter':{'high':get_queue(),"mid":get_queue(),"low":get_queue()},
        'finishfilter':{'high':get_queue(),"mid":get_queue(),"low":get_queue()},
    }
    p_list=[
        Process(target=process_flowg,args=(queue_dict,'process_flowg')),
        Process(target=process_envf,args=(queue_dict,'process_env')),
        Process(target=process_ppof,args=(queue_dict,'process_ppo')),
        Process(target=process_dataf,args=(queue_dict,'process_dataf')),
        # Process(target=process_caef,args=(queue_dict,'process_caef')),
        Process(target=process_ecf,args=(queue_dict,'process_ecf')),
        Process(target=process_finishf,args=(queue_dict,'process_finishf')),
    ]
    # print(p_list)
    [p.start() for p in p_list]
    [p.join() for p in p_list]
    [p.close() for p in p_list]

if __name__=="__main__":
    run()
