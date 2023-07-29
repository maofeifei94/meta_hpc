from mlutils.ml import DataFormat,ReplayMemory
import numpy as np
class SEQReplayMemory(ReplayMemory):
    def sample_batch_seq(self,batch_size,seq_len):
        "get rand_start_index"
        batch_indexes_list=[]
        for i in range(batch_size):
            if self._size<self._max_size:
                rand_start_index=np.random.randint(0,self._curr_pos-seq_len)
            else:
                rand_start_index=(np.random.randint(0,self._max_size-seq_len)+self._curr_pos)%self._max_size

            batch_indexes_list.append(list(range(rand_start_index,rand_start_index+seq_len)))

        batch_indexes=np.mod(np.concatenate(batch_indexes_list),self._max_size)

        samples={}
        for key in self._data_dict.keys():
            samples[key]=np.reshape(self._data_dict[key][batch_indexes],[batch_size,seq_len,-1])
        return samples
        
