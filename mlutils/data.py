import imp
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
# import hyperparam as hp
import numpy as np
import time
import cv2
import os
import copy
import shutil
import zipfile
import paddle.vision.datasets as datasets
import lmdb
import threading

from mlutils.ml import ReplayMemory,DataFormat

class mnist_data():
    def __init__(self) -> None:
        self.train_dataset = datasets.MNIST(mode='train')
        self.images=np.array(self.train_dataset.images)/255
    def get_rand_batch(self,batch_size):
        rand_index=np.random.randint(0,len(self.train_dataset),[batch_size])
        rand_images=self.images[rand_index]
        rand_images=np.reshape(rand_images,[batch_size,28,28,1])
        return rand_images
    

class LMDBRecoder():
    # 规定：key-value为数据库，index-data为数据
    def __init__(self,data_format_list,rpm_buffer_size,lmdb_dir=None,map_size=int(2**40)) -> None:
        self.lmdb_dir=lmdb_dir
        self.map_size=map_size
        "超参数"
        self.key_head_fold=int(2**56)
        "检查"
        self._check_data_format_key_head(data_format_list)
        self.data_format_list=data_format_list
        self.data_format_dict={dataformat.name:dataformat for dataformat in data_format_list}

        self.rpm_buffer_size=int(rpm_buffer_size)
        self.rpm=ReplayMemory([*data_format_list,DataFormat('index',[1],np.int64)],self.rpm_buffer_size)
        self.recent_rpm=ReplayMemory([*data_format_list,DataFormat('index',[1],np.int64)],self.rpm_buffer_size)

        self.rpm_lock=threading.Lock()
        self.txn_lock=threading.Lock()
        self.rpm_updatae_sleep=10#等待10s
        self.rpm_start_update=False


        self.init()
        th1=threading.Thread(target=self.update_rpm_thread)
        th1.start()
        # while len(self.rpm)<self.rpm_size:
        #     print("rpmsize<max")
        #     time.sleep(1)
        # print("rpmsize==max")


    def init(self):
        self.env=lmdb.open(self.lmdb_dir,map_size=self.map_size)
        # print("init lock acquire")
        self.txn_lock.acquire()
        self.txn=self.env.begin(write=True)
        # print("init lock release")
        self.txn_lock.release()
        
        self.min_index=self.find_min_index()
        self.max_index=self.find_max_index(self.min_index)
        if self.min_index is None and self.max_index is None:
            self.min_index=0
            self.max_index=-1
        print("************************")
        print(f"LMDB recoder with minindex={self.min_index} maxindex={self.max_index}")

    def update_rpm_thread(self):
        while not self.rpm_start_update:
            time.sleep(1)
            # print("update_rpm waiting")



        while 1:


            #recent
            recent_seq_index=np.array(list(range(self.max_index+1-self.rpm_buffer_size,self.max_index+1)),dtype=np.uint64)
            recent_sample={'index':np.reshape(recent_seq_index,[self.rpm_buffer_size,1]).astype(np.int64)}
            for dataformat in self.data_format_list:
                # print("getting",dataformat.name)
                recent_data_seq_index=recent_seq_index+dataformat.key_head*self.key_head_fold
                recent_value_list=self.get_index_value(recent_data_seq_index)
                recent_data=np.array(np.frombuffer(b''.join(recent_value_list),dataformat.dtype)).reshape([self.rpm_buffer_size,*dataformat.shape])
                recent_sample[dataformat.name]=recent_data
            self.rpm_lock.acquire()
            self.recent_rpm.collect_dict_of_batch(recent_sample)
            self.rpm_lock.release()


            #history
            start_index=np.random.randint(self.min_index,self.max_index-self.rpm_buffer_size)
            end_index=start_index+self.rpm_buffer_size
            seq_index=np.array(list(range(start_index,end_index)),dtype=np.uint64)
            sample={'index':np.reshape(seq_index,[self.rpm_buffer_size,1]).astype(np.int64)}
            for dataformat in self.data_format_list:
                # print("getting",dataformat.name)
                data_seq_index=seq_index+dataformat.key_head*self.key_head_fold
                value_list=self.get_index_value(data_seq_index)
                data=np.array(np.frombuffer(b''.join(value_list),dataformat.dtype)).reshape([self.rpm_buffer_size,*dataformat.shape])
                sample[dataformat.name]=data
            self.rpm_lock.acquire()
            self.rpm.collect_dict_of_batch(sample)
            self.rpm_lock.release()


            time.sleep(self.rpm_updatae_sleep)

    def collect_dict_of_batch(self,dict_of_batch):
        batch_size=np.shape(dict_of_batch[self.data_format_list[0].name])[0]
        base_key_array=np.array(list(range(self.max_index+1,self.max_index+1+batch_size)),np.uint64)
        for dataformat in self.data_format_list:
            data_batch=dict_of_batch[dataformat.name]
            # 检查数据格式
            assert data_batch.dtype==dataformat.dtype
            assert list(np.shape(data_batch))==[batch_size,*dataformat.shape]
            key_array=dataformat.key_head*self.key_head_fold+base_key_array
            self.put_batch_data(key_array,data_batch)
        self.max_index+=batch_size
        print(f"collect finish max_index={self.max_index} {type(self.max_index)} min_index={self.min_index} {type(self.min_index)} {type(self.max_index-self.min_index)}")
        self._update()
    def sample_batch_seq(self,batch_size,seq_len,use_recent_rpm):

        self.rpm_lock.acquire()
        if use_recent_rpm:
            sample_old=self.rpm.sample_batch_seq(batch_size//2,seq_len)
            sample_recent=self.recent_rpm.sample_batch_seq(batch_size//2,seq_len)
            sample={key:np.concatenate([sample_old[key],sample_recent[key]],axis=0) for key in sample_recent}
        else:
            sample=self.rpm.sample_batch_seq(batch_size,seq_len)
        self.rpm_lock.release()
        return sample
    def sample_h_update_seq(self,start_index,seq_len):
        batch_size=1
        seq_index=np.array(list(range(start_index,start_index+seq_len)),dtype=np.uint64)
        sample={'index':np.reshape(seq_index,[batch_size,seq_len,1])}
        for key in ['obs','action_env']:
            dataformat=self.data_format_dict[key]
            seq_index=seq_index+dataformat.key_head*self.key_head_fold
            value_list=self.get_index_value(seq_index)
            data=np.array(np.frombuffer(b''.join(value_list),dataformat.dtype)).reshape([batch_size,seq_len,*dataformat.shape])
            sample[key]=data
        return sample
    def h_update(self,glo_h_update,loc_h_update,klpred_h_update,index_data):
        batch,seq_len,_=index_data.shape
        index_flat=np.reshape(index_data,[-1])
        index_glo_h=index_flat+self.key_head_fold*self.data_format_dict['ec_glo_gru'].key_head
        index_loc_h=index_flat+self.key_head_fold*self.data_format_dict['ec_loc_gru'].key_head
        index_klpred_h=index_flat+self.key_head_fold*self.data_format_dict['ec_klpred_gru'].key_head
        self.put_batch_data(index_glo_h,np.reshape(glo_h_update,[batch*seq_len,-1]))
        self.put_batch_data(index_loc_h,np.reshape(loc_h_update,[batch*seq_len,-1]))
        self.put_batch_data(index_klpred_h,np.reshape(loc_h_update,[batch*seq_len,-1]))
        self._update()

        
    # def sample_batch_seq(self,batch_size,seq_len):
    #     start_index_batch=np.random.randint(self.min_index,self.max_index-seq_len,[batch_size])
    #     batch_seq_index=np.reshape([np.array(list([range(start_index,start_index+seq_len)]),dtype=np.uint64) for start_index in start_index_batch],-1)
    #     sample={'index':batch_seq_index.reshape(batch_size,seq_len)}
    #     # lmdb_time=0
    #     # np_time=0
    #     for dataformat in self.data_format_list:
    #         data_seq_index=batch_seq_index+dataformat.key_head*self.key_head_fold
    #         # t1=time.time()
    #         value_list=self.get_index_value(data_seq_index)
    #         # t2=time.time()
    #         data=np.array(np.frombuffer(b''.join(value_list),dataformat.dtype)).reshape([batch_size,seq_len,*dataformat.shape])
    #         # t3=time.time()
    #         # lmdb_time+=t2-t1
    #         # np_time+=t3-t2
    #         sample[dataformat.name]=data
    #     # print(f"lmdb_time cost {lmdb_time},np cost {np_time}")
    #     return sample



    def put_batch_data(self,index_array,data_array):
        key_bytes_list=self._transfer_index_to_key(index_array)
        data_bytes_list=self._transfer_data_to_value(data_array)
        for key_bytes,data_bytes in zip(key_bytes_list,data_bytes_list):
            self._put_key_value(key_bytes,data_bytes)

    def get_index_value(self,index):
        if isinstance(index,int):
            key=self._transfer_index_to_key([int(index)])[0]
            return self._get_value(key)
        elif isinstance(index,np.ndarray) or isinstance(index,list):
            key_list=self._transfer_index_to_key(np.array(index,np.uint64))
            return [self._get_value(key) for key in key_list]
    def find_min_index(self):
        # print("find min index lock acquire")
        self.txn_lock.acquire()
        for key,value in self.txn.cursor():
            min_index=int(self._transfer_key_to_index(key))
            self.txn_lock.release()
            return min_index
        self.txn_lock.release()
        return None
        # print("find min index lock release")
        

        return None
    def find_max_index(self,min_index):
        if min_index is None:
            return None
        lower=min_index
        upper=None
        while 1:
            if upper is None:
                check_index=1 if lower==0 else int(lower*2)
            else:
                check_index=int((lower+upper)//2)
            res=self.get_index_value(check_index)
            print(lower,upper,check_index)
            if res is None:
                upper=check_index
            else:
                lower=check_index

            if upper is None:
                pass
            elif upper-lower==1:
                return int(lower)
    def clear_buffer(self):
        self.rpm_lock.acquire()
        self.rpm_start_update=True
        self.rpm.clear()
        self.rpm_lock.release()
    def _check_data_format_key_head(self,data_format_list):
        key_head_list=[]
        for dataformat in data_format_list:
            key_head=dataformat.key_head
            assert key_head is not None
            assert key_head not in key_head_list
            key_head_list.append(key_head)
    def _cursor_key(self):
        key_list=[]
        for key,value in self.txn.cursor():
            return key

    def _get_value(self,key_bytes):
        # print("_get_value lock acquire")
        self.txn_lock.acquire()
        value=self.txn.get(key=key_bytes)
        # print("_get_value lock release")
        self.txn_lock.release()
        return value

    def _put_key_value(self,key_bytes,value_bytes):
        # print("_put_key_value lock acquire")
        self.txn_lock.acquire()
        res=self.txn.put(key=key_bytes,value=value_bytes)
        # print("_put_key_value lock release")
        self.txn_lock.release()
        return res
    def _transfer_key_to_index(self,key):
        if isinstance(key,list):
            return np.array(np.frombuffer(b''.join(key),np.uint64),np.uint64).byteswap(1)
        elif isinstance(key,bytes):
            return np.array(np.frombuffer(key,np.uint64),np.uint64).byteswap(1)[0]
        else:
            raise ValueError(f"key is {type(key)} neither list nor bytes")
    def _transfer_index_to_key(self,index):
        # byteswap调转bytes方向，
        # 比如数字1，我们期望它的二进制为00000001，但是numpy里会变成10000000，需要调转
        # byteswap(False)代表不改变原数组，若为True则改变原数组
        if isinstance(index,list) or isinstance(index,np.ndarray):
            key_array=np.array(index,np.uint64).byteswap(0).tobytes()
            key_array_list=[key_array[8*i:8*(i+1)] for i in range(len(index))]
            return key_array_list
        elif isinstance(index,int) or isinstance(index,np.uint64):
            return np.uint64(index).byteswap(0).tobytes()
        else:
            raise ValueError(f"index is {type(index)} not in[list,np.ndarray,int,np.uint64]")

    def _transfer_data_to_value(self,array):
        assert len(np.shape(array))>1
        type_bytes_dict={
            np.uint8().dtype:1,
            np.uint64().dtype:8,
            np.float32().dtype:4,
        }
        
        all_bytes=array.tobytes()
        single_array_bytes=int(np.prod(np.shape(array)[1:])*type_bytes_dict[array.dtype])
        array_bytes_list=[all_bytes[single_array_bytes*i:single_array_bytes*(i+1)] for i in range(array.shape[0])]
        return array_bytes_list
    def _transfer_value_to_data(self,all_bytes,dtype,shape):
        array=np.array(np.frombuffer(all_bytes,dtype),dtype).reshape(shape)
        return array
    def _update(self):
        # print("_update lock acquire")
        self.txn_lock.acquire()
        self.txn.commit()
        self.txn=self.env.begin(write=True)
        # print("_update lock release")
        self.txn_lock.release()
    def __len__(self):
        return max(int(self.max_index-self.min_index),0)

        
    

    
class npy_compress():
    def __init__(self,index=None,pnum=None):
        self.index=index
        self.pnum=pnum
        pass
    def check_npy_and_compress(self):
        data_dir="./all_data/minigrid_history"
        for i in range(1000):
            
            if i%self.pnum!=self.index:
                continue
            npy_name=f"{i}.npy"
            current_f_path=f"{data_dir}/{npy_name}"
            if os.path.exists(current_f_path):
                check_result=self.check_npy(current_f_path)
                if check_result:
                    self.compress(data_dir,npy_name)
                    self.delete_f(current_f_path)
                    print(f"{current_f_path} compress success")
            else:
                print(f"{current_f_path} not exist")
    def check_npy(self,current_f):
        try:
            current_data=np.load(current_f,allow_pickle=True).item()
            return True
        except:
            print(f"{current_f} broken")
            return False

    def delete_f(self,current_f_path):
        os.remove(current_f_path)
        
    def compress(self,ori_dir,ori_f_name):
        z_file_path=ori_dir+"/"+ori_f_name+".zip"
        z_file=zipfile.ZipFile(z_file_path,'w',zipfile.ZIP_LZMA)
        z_file.write(ori_dir+"/"+ori_f_name,arcname=ori_f_name)


    def decompress(self,z_file_path,out_file_dir,out_file_name):
        zipfile.ZipFile(z_file_path,"r").extractall(f"{out_file_dir}/")

class DataRecorder():
    def __init__(self,save_steps,buffer_max_steps,max_save_rpm_num,data_format_list,data_dir,tmp_dir="/tmp/data",current_rpm_sample_ratio=0.25,refresh_buffer_per_sample=200) -> None:
        """
        强化学习环境数据记录器
        例如，PPO每训练10次，就保存2048x10=20480帧图像数据作为numpy文件，并将此numpy文件压缩成zip文件
        当环境重启后，再读取到内存中。
        训练时有一定概率读取以往的数据进行训练
        save_steps:存储为numpy->zip文件的步数。此值应该固定
        buffer_max_steps:用于训练的数据最大容量，建议为save_steps的整数倍
         
        """
        assert(buffer_max_steps%save_steps==0)
        
        self.save_steps=save_steps
        self.buffer_max_steps=buffer_max_steps
        self.max_save_rpm_num=max_save_rpm_num
        self.data_dir=data_dir
        self.tmp_dir=tmp_dir
        
        self.current_rpm_sample_ratio=current_rpm_sample_ratio
        self.refresh_buffer_per_sample=refresh_buffer_per_sample

        self.rpm_save=ReplayMemory(
            data_format_list,save_steps
        )
        self.rpm_buffer=ReplayMemory(
            data_format_list,buffer_max_steps
        )
        self.rpm_current=ReplayMemory(
            data_format_list,save_steps
        )
        self.sample_iters=0
        self._init_recorder_file_system()
        self._refresh_rpm_buffer()
    def collect_dict_of_batch(self,dict_of_batch):
        self.rpm_save.collect_dict_of_batch(dict_of_batch)
        self.rpm_current.collect_dict_of_batch(dict_of_batch)
        self.save_rpm()
    def sample_batch_seq(self,batch_size,seq_len):
        # print(self.rpm_buffer._size,self.rpm_current._size)
        if self.rpm_buffer._size<seq_len:
            if self.rpm_current._size<seq_len:
                data_out={}
            else:
                data_current=self.rpm_current.sample_batch_seq(batch_size,seq_len)
                data_out=data_current
        else:
            if self.rpm_current._size<seq_len:
                data_buffer=self.rpm_buffer.sample_batch_seq(batch_size,seq_len)
                data_out=data_buffer
            else:
                current_avg_batch=batch_size*self.current_rpm_sample_ratio
                current_batch_size=int(current_avg_batch)+int(np.random.uniform(0,1)<=current_avg_batch-int(current_avg_batch))
                buffer_batch_size=batch_size-current_batch_size

                if current_batch_size==0:
                    data_out=self.rpm_buffer.sample_batch_seq(buffer_batch_size,seq_len)
                elif buffer_batch_size==0:
                    data_out=self.rpm_current.sample_batch_seq(current_batch_size,seq_len)
                else:
                    data_buffer=self.rpm_buffer.sample_batch_seq(buffer_batch_size,seq_len)
                    data_current=self.rpm_current.sample_batch_seq(current_batch_size,seq_len)
                    # print(data_buffer.keys(),data_current.keys())
                    data_out={key:np.concatenate([data_buffer[key],data_current[key]],axis=0) for key in data_buffer.keys()}
        # 若sample 200次则更新一次buffer
        self.sample_iters+=1
        if self.sample_iters%self.refresh_buffer_per_sample==0:
            self._refresh_rpm_buffer()
            print(f"iter {self.sample_iters} refresh")
        return data_out
    def save_rpm(self):
        if self.rpm_save._size==self.rpm_save._max_size:
            if self.rpm_save._curr_pos==0:
                # 存储rpm
                save_num=self._max_file_num()+1
                self.rpm_save.save(self.data_dir,save_num)
                zip_path=self._transfer_npy_to_zip(self.data_dir,save_num)
                #重置rpm
                self.rpm_save.clear()
                #更新file dict
                self.file_path_dict[save_num]=zip_path
                self._sort_out_file_path_dict()
            else:
                raise IndexError(f"rpm_save size={self.rpm_save._size},but curr_pos={self.rpm_save._curr_pos}!=0,maybe have missed the save pos")
    
    def _init_recorder_file_system(self):
        # 删除并创建临时文件夹
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        # 读取目录，加载zip文件列表
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.file_path_dict={}
        for file in os.listdir(self.data_dir):
            if file[-4:]=='.zip':
                file_path=f"{self.data_dir}/{file}"
                file_num=int(file[:-4])
                self.file_path_dict[file_num]=file_path

                self._read_rpm_dict_from_zip(file_path) #解压所有的zip
        self._sort_out_file_path_dict()
        


    def _refresh_rpm_buffer(self):
        "重置rpm buffer"
        self.rpm_buffer.clear()

        file_num_list=list(self.file_path_dict.keys())
        file_num_list.sort()
        needed_zip_num=self.buffer_max_steps//self.save_steps
        "要加载的zip num列表"
        if len(file_num_list)<=needed_zip_num:
            load_file_num_list=file_num_list
        else:
            init_index=np.random.randint(0,len(file_num_list)-needed_zip_num+1)
            load_file_num_list=file_num_list[init_index:init_index+needed_zip_num]
        print(f"加载文件{load_file_num_list}")
        t1=time.time()
        "加载zip文件"
        for num in load_file_num_list:
            file_path=self.file_path_dict[num]
            rpm_dict=self._read_rpm_dict_from_zip(file_path)
            self.rpm_buffer.collect_dict_of_batch(rpm_dict)
        print(f"用时{time.time()-t1}")
    
    def _sort_out_file_path_dict(self):
        """清理file_path_dict的数据，如果大于最大限制，则删除"""
        file_num_list=list(self.file_path_dict.keys())
        file_num_list.sort()
        if len(file_num_list)>self.max_save_rpm_num:
            for file_num in file_num_list[:-self.max_save_rpm_num]:
                # 删除多余的文件
                file_path=self.file_path_dict[file_num]
                os.remove(file_path)
                # 删除dict
                dict().pop(file_num)
        else:
            pass

    def _max_file_num(self):
        if len(list(self.file_path_dict.keys()))==0:
            return 0
        else:
            return np.max(list(self.file_path_dict.keys()))
    def _transfer_npy_to_zip(self,npy_dir,npy_num):
        npy_name=f"{npy_num}.npy"
        npy_path=f"{npy_dir}/{npy_num}.npy"
        tmp_npy_path=f"{self.tmp_dir}/{npy_num}.npy"
        zip_path=f"{npy_dir}/{npy_num}.zip"
        t1=time.time()
        print("start zip")
        z_file=zipfile.ZipFile(zip_path,'w',zipfile.ZIP_LZMA)
        z_file.write(npy_path,arcname=npy_name)
        print(f"finish zip,cost {time.time()-t1}")
        # npy移动到tmp文件夹
        # os.rename(npy_path,tmp_npy_path)
        shutil.move(npy_path,tmp_npy_path)
        return zip_path
    def _read_rpm_dict_from_zip(self,zip_path):
        npy_path=zip_path.replace(self.data_dir,self.tmp_dir).replace(".zip",".npy")
        if os.path.exists(npy_path):
            pass
        else:
            zipfile.ZipFile(zip_path,"r").extractall(f"{self.tmp_dir}/")
        rpm_dict=np.load(npy_path,allow_pickle=True).item()
        return rpm_dict
    def __len__(self):
        return len(self.rpm_current)+len(self.rpm_buffer)