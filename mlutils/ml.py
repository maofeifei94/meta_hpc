from mlutils.common import *
from mlutils.multiprocess import state_dict_to_np,set_state_dict_from_np
"处理函数"
def array_in(a,b):
    """
    a in b?
    """
    a=np.array(a)
    b=np.array(b)
    "b中是否含有a"
    return np.any(np.all(a==b,axis=-1))
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
        x=np.array(x)
        if self.avg is None:
            self.avg=x
        else:
            self.avg=self.avg*self.gamma+(1-self.gamma)*x
    def __repr__(self):
        return str(self.avg)
    # def __str__(self):
    #     return 
class gammafilter():
    def __init__(self,gamma) -> None:
        self.gamma=gamma
        self.steps=0
        self.x=0.0
    def update(self,y):
        self.steps+=1
        self.x=self.x*self.gamma+(1-self.gamma)*y
        return self.x/(1-self.gamma**self.steps)
class EstimateMovingAVerage():
    # 二阶平均估计，通过计算两种gamma值的均值估计，推测出当前的均值
    #通过重心积分公式可得，gamma曲线重心在1/ln(gamma)处
    def __init__(self) -> None:
        self.gamma1=0.99995
        self.gamma2=0.9999
        self.gamma3=0.9998
        self.c1=1/np.log(self.gamma1)
        self.c2=1/np.log(self.gamma2)
        self.c3=1/np.log(self.gamma3)
        self.g1=gammafilter(gamma=self.gamma1)
        self.g2=gammafilter(gamma=self.gamma2)
        self.g3=gammafilter(gamma=self.gamma3)
        mat=[
            [(self.c1)**2,self.c1,1.],
            [(self.c2)**2,self.c2,1.],
            [(self.c3)**2,self.c3,1.],
        ]
        self.inv_mat=np.linalg.inv(mat)
        self.mean=0.0
    def update(self,x):
        m1=self.g1.update(x)
        m2=self.g2.update(x)
        m3=self.g3.update(x)
        self.mean=np.matmul(self.inv_mat,[[m1],[m2],[m3]])[-1][0]
        return self.mean
"处理类"
class continue_to_discrete():
    def __init__(self,discrete_num,hard_mode,tanh_scale=4):
        self.discrete_num=discrete_num
        self.dim=int(np.log2(discrete_num-1))+1
        self.prod_list=[2**i for i in range(self.dim)]
        self.hard_mode=hard_mode
        self.tanh_scale=tanh_scale
    def to_prob(self,x):
        # return 1 / (1 + np.exp(-self.tanh_scale*x))
        "x范围[-1,1],通过根号的方式形成凸函数"
        # return (np.sign(x)*np.abs(x)**0.1+1)*0.5
        return np.tanh(self.tanh_scale*x)

    def to_discrete(self,action):
        "action 归一化到 [-1，1]"
        if len(action)!=self.dim:
            exit(f"to_discrete action dim {np.shape(action)} != self.dim {self.dim}")
        probs=self.to_prob(action)

        if self.hard_mode:
            code_2=(probs>=0.0).astype(np.int32)
        else:
            code_2=(np.random.uniform(-1, 1, self.dim)<probs).astype(np.int32)
        code_10=np.sum(code_2*self.prod_list)
        if code_10<self.discrete_num:
            return code_10
        else:
            return np.random.randint(0,self.discrete_num)
"神经网络模块"
class GroupGru(nn.Layer):
    def __init__(self,input_size,hidden_size,group_size):
        super().__init__()
        self.single_gru_hidden_size=int(hidden_size/group_size)
        self.gru_list=[]
        for i in range(group_size):
            gru_i=nn.GRU(input_size,self.single_gru_hidden_size)
            gru_i.flatten_parameters()
            setattr(self,f"gru_{i}",gru_i)
            self.gru_list.append(gru_i)
    def forward(self,x,hidden_state):
        y_list=[]
        h_list=[]
        for i,gru in enumerate(self.gru_list):
            y,h=gru(x,hidden_state[:,:,i*self.single_gru_hidden_size:(i+1)*self.single_gru_hidden_size])
            y_list.append(y)
            h_list.append(h)
        return paddle.concat(y_list,axis=-1),paddle.concat(h_list,axis=-1)

class HierarchicalGru(nn.Layer):
    def __init__(self,input_size,hidden_size,sub_steps1,sub_steps2):
        super().__init__()
        
        self.sub_gru_l1=nn.GRU(input_size,hidden_size)
        self.sub_gru_l2=nn.GRU(hidden_size,hidden_size)
        self.main_gru=nn.GRU(hidden_size,hidden_size)
        self.hidden_size=hidden_size
        self.sub_steps1=sub_steps1
        self.sub_steps2=sub_steps2
    def forward(self,x,hidden_state):
        _b,_t,_=x.shape

        "substep 1"
        _b1=int(_b*_t//self.sub_steps1)
        _t1=self.sub_steps1
        sub_result1,_=self.sub_gru_l1(x.reshape([_b1,_t1,-1]),paddle.zeros([1,_b1,self.hidden_size]))
        "substep2"
        _b2=int(_b*_t//(self.sub_steps1*self.sub_steps2))
        _t2=self.sub_steps2
        sub_result2,_=self.sub_gru_l2(sub_result1[:,-1:].reshape([_b2,_t2,-1]),paddle.zeros([1,_b2,self.hidden_size]))
        "main_step"
        _bmain=_b
        _tmain=int(_t//(self.sub_steps1*self.sub_steps2))
        main_result,_=self.main_gru(sub_result2[:,-1:].reshape([_bmain,_tmain,-1]),hidden_state)
        
        return main_result,_

class HierarchicalStepGru(nn.Layer):
    def __init__(self,input_size,hidden_size,sub_gru_max_steps):
        super().__init__()
        self.sub_gru=nn.GRU(input_size,hidden_size)
        self.main_gru=nn.GRU(hidden_size,hidden_size)
        self.hidden_size=hidden_size

        self.sub_gru_max_steps=sub_gru_max_steps
        self.forward_step=0

    def forward(self,x,main_hidden_state,sub_hidden_state):
        _b,_t,_=x.shape
        _t_s=_t//self.sub_gru_max_steps

        self.forward_step=self.forward_step%self.sub_gru_max_steps
        if self.forward_step==0:
            if _t%self.sub_gru_max_steps==0:
                sub_hidden_state=paddle.zeros([1,_b*_t_s,self.hidden_size])

                sub_result,sub_next_hidden_state=self.sub_gru(x.reshape([_b*_t_s,self.sub_gru_max_steps,-1]),sub_hidden_state)
                sub_result_trace=paddle.reshape(sub_result,[_b,_t,-1])
                
                main_result,main_next_hidden_state=self.main_gru(paddle.reshape(sub_result[:,-1:],[_b,_t_s,-1]),main_hidden_state)
                main_result_trace=paddle.concat([paddle.reshape(main_hidden_state,[_b,1,-1]),main_result[:,:-1]],axis=1).reshape([_b,_t_s,1,-1])
                main_result_trace=paddle.tile(main_result_trace,[1,1,self.sub_gru_max_steps,1])
                main_result_trace=paddle.reshape(main_result_trace,[_b,_t,-1])

                result_trace=paddle.concat([main_result_trace,sub_result_trace],axis=-1)

                self.forward_step=(self.forward_step+_t)%self.sub_gru_max_steps
                return result_trace,[main_next_hidden_state,paddle.zeros([1,_b*_t_s,self.hidden_size])]
            else:
                pass
        else:
            pass





class Reslayer(nn.Layer):
    def __init__(self,layer,nl_func):
        super(Reslayer,self).__init__()
        self.layer=layer
        self.nl_func=nl_func
    def forward(self,x):
        if self.nl_func is None:
            return x+self.layer(x)
        else:
            return x+self.nl_func(self.layer(x))

class dense_block(nn.Layer):
    def __init__(self,fc_num_list,act=nn.LeakyReLU(negative_slope=0.1),act_output=None):
        super().__init__()
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
class res_block(nn.Layer):
    def __init__(self,fc_num_list,act,act_output=None):
        super().__init__()
        print("act=",act)
        self.fc_num_list=fc_num_list
        #get dense_layer_list
        dense_layer_list=[]
        for i in range(1, len(fc_num_list)):
            input_dim,output_dim=fc_num_list[i-1],fc_num_list[i]
            linear_part=nn.Linear(input_dim,output_dim)
            act_part=act_output if i==len(fc_num_list)-1 else act

            if input_dim==output_dim:
                layer=Reslayer(linear_part,act_part)
            else:
                layer=linear_part if act_part is None else nn.Sequential(linear_part,act_part)
            dense_layer_list.append(layer)
        #get block
        self.block=nn.Sequential(*dense_layer_list)

    def forward(self,x):
        return self.block(x)
class ResNormBlock(nn.Layer):
    def __init__(self,fc_num_list,act,last_layer_norm,act_output=None):
        super().__init__()
        print("act=",act)
        self.fc_num_list=fc_num_list
        #get dense_layer_list
        dense_layer_list=[]
        for i in range(1, len(fc_num_list)):
            input_dim,output_dim=fc_num_list[i-1],fc_num_list[i]
            linear_part=nn.Linear(input_dim,output_dim)
            act_part=act_output if i==len(fc_num_list)-1 else act

            if input_dim==output_dim:
                layer=Reslayer(linear_part,act_part)
            else:
                layer=linear_part if act_part is None else nn.Sequential(linear_part,act_part)
            
            if i==len(fc_num_list)-1 and not last_layer_norm:
                layer=layer
            else:
                layer=nn.Sequential(layer,nn.LayerNorm([output_dim]))
            dense_layer_list.append(layer)
        #get block
        self.block=nn.Sequential(*dense_layer_list)

    def forward(self,x):
        return self.block(x)
#重参数技巧的Normal分布
#标准差 std,sigma,scale
#方差   var,sigma**2，
class ReparamNormal():
    def __init__(self,disshape) -> None:
        self.normal_dis=paddle.distribution.Normal(paddle.zeros(disshape),paddle.ones(disshape))
        pass
    def sample(self,mean,log_var,sample_num=1):
        std=(log_var/2).exp()
        if sample_num==1:
            return self.normal_dis.sample([sample_num])[0]*std+mean
        else:
            return self.normal_dis.sample([sample_num])*std+mean
def gauss_KL(mean0,sigma0,mean1,sigma1):
    ratio=sigma0/sigma1
    diff=mean1-mean0
    return 0.5*(ratio**2+(diff/sigma1)**2-1-2*paddle.log(ratio))
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
    def __init__(self,name,shape,dtype,key_head=None) -> None:
        # key_head:LMDB数据的key有8个字节64比特，其中第一个字节用于表示数据的特征编号
        self.name=name
        self.shape=shape
        self.dtype=dtype
        self.key_head=key_head
        
    def __repr__(self):
        return f"{self.name},{self.shape},{self.dtype}"

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
        self._shape_dict={}
        self.save_iter=0
        
        for dataformat in data_format_list:
            print(dataformat)
            data_name=dataformat.name
            data_shape=dataformat.shape
            data_type=dataformat.dtype
            data=np.zeros([max_size,*data_shape],data_type)
            self._data_dict[data_name]=data
            self._shape_dict[data_name]=list(data_shape)
            # print(data_name,data.shape)
        
        self._curr_pos=0
        self._size=0
    def sample_batch(self,batch_size):
        rand_indexs=np.random.randint(0,self._size,batch_size)
        samples={}
        for key in self._data_dict.keys():
            samples[key]=self._data_dict[key][rand_indexs]
        return samples
    def sample_batch_seq(self,batch_size,seq_len):
        "从数据中取出[batch，seq_len,dim]形状的序列"
        batch_slices_list=[]
        for i in range(batch_size):
            if self._size<self._max_size:
                rand_start_index=np.random.randint(0,self._curr_pos-seq_len)
            else:
                rand_start_index=(np.random.randint(0,self._max_size-seq_len)+self._curr_pos)%self._max_size

            rand_end_index=rand_start_index+seq_len

            if rand_end_index>self._max_size:
                batch_slices_list.append(slice(rand_start_index,self._max_size,1))
                batch_slices_list.append(slice(0,rand_end_index%self._max_size,1))
            else:
                batch_slices_list.append(slice(rand_start_index,rand_end_index,1))

        samples={}
        for key in self._data_dict.keys():
            batch_seq_data_list=[]
            for _slice in batch_slices_list:
                batch_seq_data_list.append(self._data_dict[key][_slice])
            batch_seq_data=np.concatenate(batch_seq_data_list,axis=0)
            samples[key]=np.reshape(batch_seq_data,[batch_size,seq_len,*self._shape_dict[key]])
            # print(key,samples.shape)
        return samples
    def get_data_dict(self):
        return self._data_dict
    def collect(self,collect_data_dict):
        for key in self._data_dict.keys():
            ck=collect_data_dict[key]
            if self._shape_dict[key]!=list(np.shape(ck)):
                raise ValueError(f"rmp collect data {key} get shape {np.shape(ck)} != {self._shape_dict[key]}")
            self._data_dict[key][self._curr_pos]=ck

        self._curr_pos=(self._curr_pos+1)%self._max_size
        self._size=min(self._size+1,self._max_size)
        pass
    def collect_dict_of_batch(self,collect_data_batch_dict):
        """
        eg:{img:np.ndarray([2048,3,64,64])}
        """
        left_to_collect_size=len(collect_data_batch_dict[list(collect_data_batch_dict.keys())[0]])
        curr_data_pos=0

        while 1:
            curr_collect_size=min(self._max_size-self._curr_pos,left_to_collect_size)

            for key in self._data_dict.keys():
                self._data_dict[key][self._curr_pos:self._curr_pos+curr_collect_size]=collect_data_batch_dict[key][curr_data_pos:curr_data_pos+curr_collect_size]


            self._curr_pos+=curr_collect_size
            self._curr_pos%=self._max_size
            self._size=min(self._size+curr_collect_size,self._max_size)

            curr_data_pos+=curr_collect_size
            left_to_collect_size-=curr_collect_size

            if left_to_collect_size<=0:
                break

        
    def collect_dict_list(self,collect_data_dict_list):
        for data_dict in collect_data_dict_list:
            self.collect(data_dict)
    def save(self,save_dir,f_name):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(f"{save_dir}/{f_name}.npy",self._data_dict)
        print(f"save {save_dir}/{f_name}.npy of size {self._size}/{self._max_size} success")
        self.save_iter+=1
    def clear(self):
        for key in self._data_dict.keys():
            self._data_dict[key]=np.zeros_like(self._data_dict[key])
        self._curr_pos=0
        self._size=0
    def __len__(self):
        return self._size


class ModelSaver():
    def save_model(self,model,save_dir,iter_num):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(f"start_save_model {save_dir}/{iter_num}_model.pdparams")
        paddle.save(model.state_dict(),f"{save_dir}/{iter_num}_model.pdparams")
        print(f"save model {save_dir}/{iter_num}_model.pdparams success")
        
    def load_model(self,model,save_dir,iter_num):
        print(f"start load_model {save_dir}/{iter_num}_model.pdparams")
        try:
            params_state = paddle.load(path=f"{save_dir}/{iter_num}_model.pdparams")
            model.set_state_dict(params_state)
            print(f"load model {save_dir}/{iter_num}_model.pdparams success")
        except:
            print(f"load Error:load model {save_dir}/{iter_num}_model.pdparams failed")
        
    def update_model(self,model,state_dict):
        model.set_state_dict(state_dict)
        print("update param success")
    def update_model_from_np(self,model,state_dict_np):
        set_state_dict_from_np(model.state_dict(),state_dict_np)
    def send_model_to_np(self,model):
        return state_dict_to_np(model.state_dict())