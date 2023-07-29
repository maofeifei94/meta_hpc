
try:
    import paddle
    import paddle.nn as nn
    import paddle.nn.functional as F
    import xlrd
except:
    pass
# import hyperparam as hp
import numpy as np
import time
import cv2


debug=False

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
        np.float
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

# "信息传递类"
# class messager_single_process():
#     def __init__(self) -> None:
#         self.send_pair=[]
#         pass
#     def send(self,send_obj,recv_obg):
#         for pair in self.send_pair:
#             getattr()=getattr(send_obj,pair[0])
#         pass
#     def regist(self,send_obj_var_name,recv_obj_var_name):
#         self.send_pair.append([send_obj_var_name,recv_obj_var_name])
#         pass
"图像处理类"
def concat_imglist(img_list,target_grid=None,scale=[1,1]):
    """
    target_grid:输出的图像网格，比如[4,3]代表高度有4张图，宽度有3张图
    scale:图像放大倍数[y,x]
    """
    h_list=[]
    for i in range(target_grid[0]):
        # print()
        h_i=np.concatenate(img_list[i*target_grid[1]:(i+1)*target_grid[1]],axis=1)
        h_list.append(h_i)
    
    all_img=np.concatenate(h_list,axis=0)
    all_img=cv2.resize(all_img,None,fx=scale[1],fy=scale[0],interpolation=cv2.INTER_AREA)
    return all_img

def img_cross_area(full_x,full_y,minx,miny,maxx,maxy):
    """
    full_x,full_y是大图的尺寸
    minx,miny,maxx,maxy是小图的
    """
    full_img_minx=max(0,minx)
    full_img_miny=max(0,miny)
    full_img_maxx=min(maxx,full_x)
    full_img_maxy=min(maxy,full_y)

    sub_img_minx=full_img_minx-minx
    sub_img_miny=full_img_miny-miny
    sub_img_maxx=full_img_maxx-full_img_minx+sub_img_minx
    sub_img_maxy=full_img_maxy-full_img_miny+sub_img_miny

    return [full_img_miny,full_img_maxy,full_img_minx,full_img_maxx],[sub_img_miny,sub_img_maxy,sub_img_minx,sub_img_maxx]
        



"处理函数"
def to_paddle(x):
    if isinstance(x,list):
        return paddle.to_tensor(np.array(x,dtype=np.float32))
    if isinstance(x,paddle.Tensor):
        return x
    if isinstance(x,np.ndarray):
        return paddle.to_tensor(x.astype(np.float32)) 
def one_hot(x,num_classes):
    x_shape=np.shape(x)
    x_flat=np.reshape(x,-1)
    vec=np.eye(num_classes)[x_flat]
    vec=np.reshape(vec,[*x_shape,num_classes]).astype(np.float32)
    return vec

class moving_avg():
    def __init__(self,gamma=0.99):
        self.gamma=gamma
        self.avg=None
    def update(self,x):
        if self.avg is None:
            self.avg=x
        else:
            self.avg=self.avg*self.gamma+(1-self.gamma)*x
    def __repr__(self):
        return str(self.avg)
    # def __str__(self):
    #     return 

"处理类"
class continue_to_discrete():
    def __init__(self,discrete_num,hard_mode,sigmoid_scale=8.0):
        self.discrete_num=discrete_num
        self.dim=int(np.log2(discrete_num-1))+1
        self.prod_list=[2**i for i in range(self.dim)]
        self.hard_mode=hard_mode
        self.tanh_scale=sigmoid_scale
    def to_prob(self,x):
        # return 1 / (1 + np.exp(-self.tanh_scale*x))
        return (np.sign(x)*np.abs(x)**0.5)*0.5+0.5
    def to_discrete(self,action):
        if len(action)!=self.dim:
            exit(f"to_discrete action dim {np.shape(action)} != self.dim {self.dim}")
        probs=self.to_prob(action)
        # print(self.dim,probs)
        if self.hard_mode:
            code_2=(probs>=0.5).astype(np.int32)
            # print(code_2)
        else:
            code_2=(np.random.uniform(0, 1, self.dim)<probs).astype(np.int32)
        code_10=np.sum(code_2*self.prod_list)
        if code_10<self.discrete_num:
            return code_10
        else:
            return np.random.randint(0,self.discrete_num)
"神经网络模块"
class Reslayer(nn.Layer):
    def __init__(self,layer):
        super(Reslayer,self).__init__()
        self.layer=layer
    def forward(self,x):
        return x+self.layer(x)
class dense_block(nn.Layer):
    def __init__(self,fc_num_list,act=nn.LeakyReLU(negative_slope=0.1),act_output=None):
        super(dense_block,self).__init__()
        self.fc_num_list=fc_num_list
        #get dense_layer_list
        dense_layer_list=[]
        for i in range(1, len(fc_num_list)):
            linear_part=nn.Linear(fc_num_list[i-1],fc_num_list[i])
            act_part=act_output if i==len(fc_num_list)-1 else act
            layer=linear_part if act_part==None else nn.Sequential(linear_part,act_part)
            dense_layer_list.append(layer)
        #get block
        self.block=nn.Sequential(*dense_layer_list)

    def forward(self,x):
        return self.block(x)

"标准化强化学习模块"
class Basic_Module():
    def __init__(self) -> None:
        pass
    def reset(self):
        pass
    def input(self):
        "接受外界输入，返回处理结果"
        pass
    def rpm_collect(self):
        "收集训练数据"
        pass
    def rpm_clear(self):
        "清空rpm的训练数据"
        pass
    def learn(self):
        "更新网络"
        pass
    def save_model(self):
        pass
    def load_model(self):
        pass
class DataFormat():
    def __init__(self,name,shape,dtype) -> None:
        self.name=name
        self.shape=shape
        self.dtype=dtype
        pass
class ReplayMemory():
    """
    data_format_dict={
        'obs':{"shape":[512],"dtype":np.float32},
        }
    """
    def __init__(self,data_format_list,max_size) -> None:
        self._data_dict={}
        self._data_format_list=data_format_list
        self._max_size=max_size
        
        for dataformat in data_format_list:
            data_name=dataformat.name
            data_shape=dataformat.shape
            data_type=dataformat.dtype
            data=np.zeros([max_size,*data_shape],data_type)
            self._data_dict[data_name]=data
        
        self._curr_pos=0
        self._size=0
    def sample_batch(self,batch_size):
        rand_indexs=np.random.randint(0,self._size,batch_size)
        samples={}
        for key in self._data_dict.keys():
            samples[key]=self._data_dict[key][rand_indexs]
        return samples
    def collect(self,collect_data_dict):
        for key in self._data_dict.keys():
            self._data_dict[key][self._curr_pos]=collect_data_dict[key]
        self._curr_pos=(self._curr_pos+1)%self._max_size
        self._size=min(self._size+1,self._max_size)
        pass
    def clear(self):
        for key in self._data_dict.keys():
            self._data_dict[key]=np.zeros_like(self._data_dict[key])
        self._curr_pos=0
        self._size=0
    def __len__(self):
        return self._size

        