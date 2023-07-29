import numpy as np

class DataFormat():
    def __init__(self,name,shape,dtype) -> None:
        self.name=name
        self.shape=shape
        self.dtype=dtype
        pass
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
        self._name_shape_dict={}
        
        for dataformat in data_format_list:
            print(dataformat)
            data_name=dataformat.name
            data_shape=dataformat.shape
            data_type=dataformat.dtype
            data=np.zeros([max_size,*data_shape],data_type)
            self._data_dict[data_name]=data
            self._name_shape_dict[data_name]=list(data_shape)
        
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
            samples[key]=np.reshape(batch_seq_data,[batch_size,seq_len,*self._name_shape_dict[key]])
        return samples

    def collect(self,collect_data_dict):
        for key in self._data_dict.keys():
            ck=collect_data_dict[key]
            if self._name_shape_dict[key]!=list(np.shape(ck)):
                raise ValueError(f"rmp collect data {key} get shape {np.shape(ck)} != {self._name_shape_dict[key]}")
            self._data_dict[key][self._curr_pos]=ck

        self._curr_pos=(self._curr_pos+1)%self._max_size
        self._size=min(self._size+1,self._max_size)
        pass
    def collect_batch_dict(self,collect_data_batch_dict):
        batch_size=len(collect_data_batch_dict[collect_data_batch_dict.keys()[0]])
        for i in range(batch_size):
            collect_data_dict={collect_data_batch_dict[key][i] for key in collect_data_batch_dict.keys()}
            self.collect(collect_data_dict)

    def clear(self):
        for key in self._data_dict.keys():
            self._data_dict[key]=np.zeros_like(self._data_dict[key])
        self._curr_pos=0
        self._size=0
    def __len__(self):
        return self._size