
import time
import numpy as np
import os
from matplotlib import pyplot as plt
from mlutils.common import mkdirs
from importlib import reload
import logging
def get_date_str():
    return time.strftime("%Y_%m%d_%H%M%S",time.localtime(time.time()))
class mllogger():
    def __init__(self,log_dir,log_fname="") -> None:
        
        
        mkdirs(log_dir)
        reload(logging)
        logging.basicConfig(filename=f"{log_dir}/{log_fname}_"+time.strftime("%Y_%m%d_%H%M%S",time.localtime(time.time()))+".txt",level=logging.INFO)
        print(f"{log_dir}/{log_fname}_"+time.strftime("%Y_%m%d_%H%M%S",time.localtime(time.time()))+".txt")


    def log_str(self,content):
        logging.info(content)
    def log_dict(self,content_dict):
        "记录损失值，用|隔开，loss=value形式记录"
        content=""
        for key in content_dict.keys():
            content+=f"|{key}={content_dict[key]}"
        logging.info(content)
def find_newest_file(log_dir):
    max_time=0
    newest_f=None
    for f in os.listdir(log_dir):
        if ".txt" not in f:
            continue
        try:
            f_time=int(f.replace("_","").replace(".txt",""))
        except:
            continue
        if f_time>max_time:
            max_time=f_time
            newest_f=f
    return f"{log_dir}/{newest_f}"
def read_log(log_file):
    info_dict={}
    for line in open(log_file,"r").readlines():
        if not "|" in line:
            continue
        line_split=line.split("|")
        for j in range(1,len(line_split)):
            info_parts=line_split[j].split("=")
            info_name=info_parts[0]
            info_value=float(info_parts[1].replace("\n",""))
            if info_name in info_dict.keys():
                info_dict[info_name].append(info_value)
            else:
                info_dict[info_name]=[info_value]
    return info_dict

def norm_list(x):
    return x/np.max(np.abs(x))

def moving_avg_list(reward_list,gamma=0.9):
    avg_reward_list=[]
    for r in reward_list:
        if len(avg_reward_list)==0:
            avg_reward_list.append(r)
        else:
            avg_reward_list.append(avg_reward_list[-1]*gamma+(1-gamma)*r)
    return avg_reward_list
def round_sf_np(x,significant_figure=0):
    """
    有效位数significant_figure
    """
    return np.format_float_positional(x, precision=significant_figure, unique=False, fractional=False, trim='k')
def plot_info_dict(info_dict,loc='lower left'):
    for key in info_dict.keys():
        info_dict[key]=info_dict[key]
        if key in ["reward","v_loss","action_continuous_dist_entropy"]:
            label=key
            prec=3
            label+=" | max"+str(round_sf_np(np.max(moving_avg_list(info_dict[key])),prec))
            label+=" | min"+str(round_sf_np(np.min(moving_avg_list(info_dict[key])),prec))
            label+=" | now"+str(round_sf_np(moving_avg_list(info_dict[key])[-1],prec))
            plt.plot(norm_list(moving_avg_list(info_dict[key])),label=label)
            plt.scatter(list(range(len(info_dict[key]))),info_dict[key]/np.max(np.abs(moving_avg_list(info_dict[key]))),s=1.0,alpha=0.3)
    plt.ylim(-1.1,1.1)
    # plt.legend(loc='upper left')
    plt.legend(loc=loc)
    plt.show()
    