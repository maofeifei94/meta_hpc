# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"




# try:
#     import paddle
#     import paddle.nn as nn
#     import paddle.optimizer as optim
#     from myf_ML_util import dense_block,ReplayMemory,concat_imglist
# except:
#     pass
import numpy as np
# import cv2
import random
# from matplotlib import pyplot as plt

import time
# from hpc import Conv_Encoder_net,Conv_Decoder_net
# a=paddle.ones([1,3,64,64])
# ct=nn.Conv2DTranspose(3,3,kernel_size=[3,3],stride=2,padding="SAME")
# b=ct(a)
# print(b.shape)
# # print(b)
def test_RM():
    data_format_dict={"img":[[64,64,3],np.float32]}
    rm=ReplayMemory(data_format_dict,10000)
    print(rm.img.shape,rm.data_dict)
def train_AE():
    train_dataset = paddle.vision.datasets.MNIST(mode='train')
    print(len(train_dataset),np.shape(np.array(train_dataset[0][0])),train_dataset[0][1])
    img_dataset=[cv2.resize(np.array(x[0],dtype=np.float32)/255,dsize=(64,64)) for x in train_dataset]
    # img_dataset/=255
    print(np.shape(img_dataset),np.max(img_dataset),np.min(img_dataset))
    
    encoder=Conv_Encoder_net([3,64,64])
    dnet=dense_block([64,64,64],act_output=nn.LeakyReLU(negative_slope=0.1))
    decoder=Conv_Decoder_net()

    optimizer=optim.Adam(learning_rate=0.0003,parameters=[*encoder.parameters(),*dnet.parameters(),*decoder.parameters()])

    for i in range(10000000000):
        train_imgs=random.sample(img_dataset,64)
        train_imgs=np.expand_dims(train_imgs,axis=1)
        train_imgs=np.repeat(train_imgs,3,axis=1)
        # print(np.shape(train_imgs))
        train_imgs_paddle=paddle.to_tensor(train_imgs)
        x=encoder(train_imgs_paddle)
        # x=paddle.reshape(x,[64,-1])
        # x=dnet(x)
        # x=paddle.reshape(x,[64,1,8,8])
        # print(x)
        restore_img=decoder(x)
        loss=paddle.mean((train_imgs_paddle-restore_img)**2)
        print(loss)

        optimizer.clear_grad()
        loss.backward()
        optimizer.step()

        if i%100==0:
            show_img=np.transpose(train_imgs_paddle[0].numpy(),[1,2,0])
            show_restore=np.transpose(restore_img[0].numpy(),[1,2,0])
            cv2.imshow("show_img",cv2.resize(np.concatenate([show_img,show_restore],axis=1),dsize=None,fx=3,fy=3,interpolation=cv2.INTER_AREA))
            cv2.waitKey(10)
def test_vae():
    from hpc import V_Auto_Encoder

    vae=V_Auto_Encoder()
    train_dataset = paddle.vision.datasets.MNIST(mode='train')
    print(len(train_dataset),np.shape(np.array(train_dataset[0][0])),train_dataset[0][1])
    img_dataset=[cv2.resize(np.array(x[0],dtype=np.float32)/255,dsize=(64,64)) for x in train_dataset]
    print(np.shape(img_dataset))

    for img in img_dataset:
        vae.rpm_collect(np.repeat(np.expand_dims(img,axis=-1),3,axis=-1))
    print(len(vae.rpm))

    for i in range(1000000000):
        img,recon_img=vae.learn()
        # print(img.shape)
        if i %100==0:
            show_img=np.concatenate([np.transpose(img[0],[1,2,0]),np.transpose(recon_img[0],[1,2,0])],axis=1)
            show_img=cv2.resize(show_img,dsize=None,fx=5,fy=5,interpolation=cv2.INTER_AREA)
            cv2.imshow("img",show_img)
            cv2.waitKey(10)

class rect():
    def __init__(self,pt1,pt2,color) -> None:
        self.pt1,self.pt2,self.color=pt1,pt2,color
        # print(pt1,pt2,color)
        pass
class circle():
    def __init__(self,center,radiu,color) -> None:
        self.center,self.radiu,self.color=center,radiu,color
        pass
def test_draw_cv():
    np.int32

    rect_list=[]
    circle_list=[]
    for i in range(100):
        pt1=tuple(np.random.randint(0,600,[2]))
        pt2=tuple(pt1+np.random.randint(20,50,[2]))
        color=(np.random.randint(120,250),np.random.randint(120,250),np.random.randint(120,250))
        rect_list.append(rect(pt1,pt2,color))

        center=tuple(np.random.randint(0,600,[2]))
        radiu=np.random.randint(15,40)
        color=(np.random.randint(120,250),np.random.randint(120,250),np.random.randint(120,250))
        circle_list.append(circle(center,radiu,color))


    for i in range(10):
        img=np.zeros([600,600,3],dtype=np.uint8)
        a=time.time()
        for _rect in rect_list:
            cv2.rectangle(img,_rect.pt1,_rect.pt2,_rect.color,-1)
        for _circle in circle_list:
            cv2.circle(img,_circle.center,_circle.radiu,_circle.color,-1)
        print("cv cost",time.time()-a)
    cv2.imshow("img",img)
    cv2.waitKey()



def test_draw_gl():

        

    def init():
        glClearColor(0.0, 0.0, 0.0, 1.0)
        gluOrtho2D(-1.0, 1.0, -1.0, 1.0)

    def drawFunc():
        glClear(GL_COLOR_BUFFER_BIT)

        glColor3f(0.0, 1.0, 1.0)
        glPolygonMode(GL_FRONT, GL_LINE)
        glPolygonMode(GL_BACK, GL_FILL)
        glBegin(GL_QUADS)
        glVertex2f(-0.8, -0.8)
        glVertex2f(-0.8, 0.8)
        glVertex2f(0.8, 0.8)
        glVertex2f(0.8, -0.8)
        glEnd()

        glFlush()

    glutInit()
    glutInitDisplayMode(GLUT_RGBA|GLUT_SINGLE)
    glutInitWindowSize(600, 600)
    glutCreateWindow(b"My OpenGL window")

    glutDisplayFunc(drawFunc)
    init()
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    a=time.time()
    data = glReadPixels(0, 0, 600, 600, GL_RGBA, GL_UNSIGNED_BYTE)
    print("read buffer",time.time()-a)
    print(len(data))
    glutMainLoop()

    


def test_cv():
    img=np.zeros([30,30,3],dtype=np.uint8)
    bg_img=np.random.randint(30,80,[30,30,3],np.uint8)
    cv2.circle(img,(15,15),10,(50,200,50),-1)
    show_img=np.where(img>0,img,bg_img)
    show_img=cv2.resize(show_img,(600,600),interpolation=cv2.INTER_AREA)
    
    cv2.imshow("img",show_img)
    cv2.waitKey()
    pass
def test_time():
    t1=time.time()
    a=np.random.randint(0,100,[2])
    for i in range(50):
        
        a*=0
        a+=1
        a*=(3*3*3*3*3*3*3*3)

    print(time.time()-t1)
def compare_cv_and_texture():
    img1=np.zeros([600,600,3],np.uint8)
    img2=np.zeros([600,600,3],np.uint8)
    img3=paddle.zeros([600,600,3],'int32')

    texture_img=np.zeros([31,31,3],np.uint8)
    cv2.circle(texture_img,(15,15),15,(200,200,200),-1)

    texture_img_paddle=paddle.to_tensor(texture_img,'int32')

    center_site_list=np.random.randint(100,500,[50000,2])

    t1=time.time()
    for c in center_site_list:
        cv2.circle(img1,(c[0],c[1]),15,(200,200,200),-1)
    print(f"cv cost {time.time()-t1}")
    

    t1=time.time()
    for c in center_site_list:
        img2[c[1]-15:c[1]+16,c[0]-15:c[0]+16]=texture_img
    print(f"texture cost {time.time()-t1}")

    t1=time.time()
    for c in center_site_list:
        img3[int(c[1]-15):int(c[1]+16),int(c[0]-15):int(c[0]+16)]=texture_img_paddle
    print(f"texture paddle cost {time.time()-t1}")


    cv2.imshow("img1",img1)
    cv2.imshow("img2",img2)
    cv2.waitKey()

    
def test_cmd():
    for i in range(99):
        os.system("start cmd")
def test_paddle_sum():
    a=paddle.ones([5,4,3])
    b=a.sum(axis=-1)
    print(b.shape)
def test_show_plot():
    import os
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']=''
    reward_str=open('reward_str.txt','r').readlines()

    # print(reward_str)
    reward_list=[]
    for rs in reward_str:
        if 'ppo reward=' in rs:
            print(rs)
            reward=rs.split("\n")[0].split('ppo reward=')[-1]
            print(reward)
            reward_list.append(int(reward))
    print(reward_list)
    avg_reward_list=[]

    avg_gamma=0.5
    for i in range(len(reward_list)):
        if i==0:
            avg_reward=reward_list[i]
        else:
            avg_reward=avg_reward*avg_gamma+(1-avg_gamma)*reward_list[i]
        avg_reward_list.append(avg_reward)
    plt.plot(avg_reward_list)
    plt.show()

def test_place_cell():
    # import os
    # os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']=''
    maze_shape=[16,16]
    maze_size=np.prod(maze_shape)
    hid_dim=32
    data=paddle.to_tensor(np.eye(maze_size,maze_size,dtype=np.float32),'float32')
    label=data


    encoder=nn.Sequential(
        nn.Linear(maze_size,64),nn.Tanh(),
        nn.Linear(64,64),nn.Tanh(),
        nn.Linear(64,64),nn.Tanh(),
        nn.Linear(64,64),nn.Tanh(),
        nn.Linear(64,64),nn.Tanh(),
        nn.Linear(64,hid_dim),nn.Dropout()
        )
    decoder=nn.Sequential(nn.Linear(hid_dim,maze_size),nn.Softmax())

    optimizer=optim.Adam(0.0001,parameters=[*encoder.parameters(),*decoder.parameters()])

    for i in range(999999999999999):
        hid=encoder(data)
        output=decoder(hid)
        # print(output.shape)
        loss=paddle.mean((output-label)**2)
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()
        # print(i)
        if i %100==0:
            print(f"{i} loss= {loss.numpy()}")
            all_img=hid.numpy()
            img_list=[np.reshape(all_img[:,j],maze_shape) for j in range(hid_dim)]
            img_list=[(img-np.min(img))/(np.max(img)-np.min(img)) for img in img_list]

            test_img=concat_imglist(img_list,target_grid=[4,8],scale=[10,10])
            cv2.imshow("img",test_img)
            cv2.waitKey(10)
            # plt.imshow(test_img)
            # plt.show()
        
def test_grid_cell():
    import os
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']=''
    maze_shape=[32,32]
    maze_size=np.prod(maze_shape)

    site_mat=np.zeros([maze_size,maze_size])
    next_site_mat=np.zeros([maze_size,maze_size])


    for i in range(maze_shape[0]):
        for j in range(maze_shape[1]):
            site_mat[i*maze_shape[1]+j][i*maze_shape[1]+j]=1
            next_site_mat[i*maze_shape[1]+j][i*maze_shape[1]+j]=1
            p1=[i-1,j]
            p2=[i+1,j]
            p3=[i,j-1]
            p4=[i,j+1]

            for p in [p1,p2,p3,p4]:
                if p[0]>=0 and p[0]<=maze_shape[0]-1 and p[1]>=0 and p[1]<=maze_shape[1]-1:
                    next_site_mat[i*maze_shape[1]+j][p[0]*maze_shape[1]+p[1]]=0.2

    for rand_index in range(maze_size):
        site_img=np.reshape(site_mat[rand_index],maze_shape)
        next_site_img=np.reshape(next_site_mat[rand_index],maze_shape)

        # cv2.imshow('site',cv2.resize(site_img,None,fx=10,fy=10,interpolation=cv2.INTER_AREA))
        # cv2.imshow('next_site',cv2.resize(next_site_img,None,fx=10,fy=10,interpolation=cv2.INTER_AREA))
        # cv2.waitKey()
    # def eigdecomp(M):
    #     E, V = np.linalg.eig(M)
    #     idx = np.argsort(E)[::-1]
    #     return V[:, idx[:]].astype(float), E
    # def plot_eigvecs(eig_vecs, maze, grid_size=10, filename=None, figsize=(15,15)):
    #     plt.figure(figsize=figsize)
    #     for i in range(eig_vecs.shape[0]-1):
    #         if i >= grid_size**2:
    #             break
    #         ax = plt.subplot(grid_size, grid_size, i+1)
    #         ax.matshow(vec2maze(eig_vecs[:, i+1], maze), cmap='RdBu_r')
    #         ax.axes.get_xaxis().set_ticks([])
    #         ax.axes.get_yaxis().set_ticks([])
    #         plt.grid(False)
    #     plt.tight_layout()
    #     if filename is not None:
    #         plt.savefig(filename, bbox_inches='tight')
    #     plt.show()
    
    # V,E=eigdecomp(next_site_mat)
    # print(next_site_mat.shape,V.shape,E.shape)

    # for i in range(maze_size):
    #     V_i=V[:,i]
    #     E_i=E[i]
    #     print(E_i)
    #     plt.imshow(np.abs(V_i).reshape(maze_shape),cmap='RdBu_r')
    #     plt.show()



    data=paddle.to_tensor(site_mat,'float32')
    label=paddle.to_tensor(next_site_mat,'float32')

    hid_dim=16
    encoder=nn.Sequential(
        nn.Linear(maze_size,hid_dim),
        )
    decoder=nn.Sequential(nn.Linear(hid_dim,maze_size))

    optimizer=optim.Adam(0.0001,parameters=[*encoder.parameters(),*decoder.parameters()])

    for i in range(999999999999999):
        hid=encoder(data)
        output=decoder(hid)
        # print(output.shape)
        loss=paddle.mean((output-label)**2)
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()
        # print(i)
        if i %1000==0:
            print(f"{i} loss= {loss.numpy()}")
            all_img=hid.numpy()
            img_list=[np.reshape(all_img[:,j],maze_shape) for j in range(hid_dim)]
            img_list=[(img-np.min(img))/(np.max(img)-np.min(img)) for img in img_list]

            test_img=concat_imglist(img_list,target_grid=[4,4],scale=[10,10])
            # cv2.imshow("img",test_img)
            # cv2.waitKey(10)
            plt.imshow(test_img,cmap='RdBu_r')
            plt.show()
def logger():
    import logging

    # 通过下面的方式进行简单配置输出方式与日志级别
    logging.basicConfig(filename='log/logger.txt', level=logging.INFO)


    logging.info('reward=0')



def algorhtmReq():
    import requests
    import base64
    import json
    # cv2.imwrite('/home/kylin/桌面/test.jpg',np.zeros([5,5,3],np.uint8))
    # image = open('/home/kylin/桌面/test.jpg', 'rb')
    image = open('/home/kylin/桌面/mmexport1660746408523_3050x4280.jpg', 'rb')
    image_read = image.read()
    # image_read=np.zeros([5,5,3],dtype=np.uint8)
    image_64_encode = base64.encodestring(image_read).decode('utf-8')
    ss = json.dumps(
    {
        "parameter": {
            "rsp_media_type": "jpg",
        },
        "extra": {},
        "media_info_list": [{
            "media_data": image_64_encode,
            "media_profiles": {
                "media_data_type":"jpg"
            },
            "media_extra": {
            }
        }]

    }
    )

    AIBeauty_url = "https://openapi.mtlab.meitu.com/v1/image_restoration?api_key=ca99a2ef354e44408bd34d3aa57b8e71&api_secret=c3a3bcd622c94b118c536759f5da4314"
    # print(ss)
    response = requests.post(AIBeauty_url, data=ss)
    ss2 = json.dumps(response.json())
    base64_result=response.json()['media_info_list'][0]['media_data']
    print(type(image_64_encode),"\n",type(base64_result))
    print(type(base64_result))
    # print(response.json()['media_info_list'][0]['meadia_data'])
    with open('/home/kylin/桌面/result.jpg','wb') as f:
        f.write(base64.b64decode(base64_result))
    # print(response.status_code)
def test_mail():
    import smtplib
    import email
    # 负责构造文本
    from email.mime.text import MIMEText
    # 负责构造图片
    from email.mime.image import MIMEImage
    # 负责将多个对象集合起来
    from email.mime.multipart import MIMEMultipart
    from email.header import Header
    mail_host = "smtp.qq.com"
    stp = smtplib.SMTP()
    # 设置发件人邮箱的域名和端口，端口地址为25
    print(1)
    # stp.con
    stp.connect(mail_host, 25)  
    print(2)
def test_gauss1():
    "假设数据是高斯分布，网络输出固定值，均方差损失"
    from paddle.distribution import Normal
    input_rand_dim=1
    output_dim=1
    label_dim=1
    batch_size=128
    show_batch_size=1000

    class gauss_net(nn.Layer):
        def __init__(self, name_scope=None, dtype="float32"):
            super().__init__(name_scope, dtype)

            self.layer=nn.Sequential(
                nn.Linear(input_rand_dim,16),nn.LeakyReLU(0.1),
                nn.Linear(16,16),nn.LeakyReLU(0.1),
                nn.Linear(16,16),nn.LeakyReLU(0.1),
                nn.Linear(16,16),nn.LeakyReLU(0.1),
                nn.Linear(16,output_dim)
            )
        def forward(self,x):
            out=self.layer(x)
            return out


    
    net=gauss_net()
    input_source=Normal([0.0],[1.0])
    optimizer=optim.Adam(0.0001,parameters=net.parameters())


    for i in range(10000000000):
        output=net(input_source.sample([batch_size]))
        # print(output.shape)
        label=paddle.cast(paddle.randint(0,2,[batch_size]),np.float32)
        loss=paddle.mean((output-label)**2)

        optimizer.clear_grad()
        loss.backward()
        optimizer.step()

        if i %1000==0:
            print(i,"loss=",loss.numpy())
            show_data=net(input_source.sample([show_batch_size]))
            plt.hist(show_data.numpy(),bins=50,range=[-0.5,1.5])
            plt.show()

def test_gauss2():
    "假设数据固定值，网络输出高斯分布，最大化概率"
    from paddle.distribution import Normal
    input_rand_dim=4
    output_dim=2
    label_dim=1
    batch_size=128
    show_batch_size=1000

    repeat_times=30

    class gauss_net(nn.Layer):
        def __init__(self, name_scope=None, dtype="float32"):
            super().__init__(name_scope, dtype)

            nl_func=nn.LeakyReLU()

            self.layer=nn.Sequential(
                nn.Linear(input_rand_dim,16),nl_func,
                nn.Linear(16,16),nl_func,
                nn.Linear(16,16),nl_func,
                nn.Linear(16,16),nl_func,
                nn.Linear(16,output_dim)
            )
        def forward(self,x):
            out=self.layer(x)
            mean=out[:,:label_dim]
            log_var=out[:,label_dim:]
            return mean,log_var


    
    net=gauss_net()
    input_source=Normal([0.0]*input_rand_dim,[1.0]*input_rand_dim)
    optimizer=optim.Adam(0.0001,parameters=net.parameters())


    for i in range(10000000000):
        mean,log_var=net(input_source.sample([batch_size*repeat_times]))
        # print(output.shape)
        mean_repeat=paddle.reshape(mean,[batch_size,repeat_times,1])
        log_var_repeat=paddle.reshape(log_var,[batch_size,repeat_times,1])

        label=paddle.cast(paddle.randint(0,3,[batch_size]),np.float32)
        label_repeat=paddle.tile(paddle.reshape(label,[batch_size,1,1]),[1,repeat_times,1])

        



        # loss=paddle.mean((output-label)**2)
        var_repeat=paddle.exp(log_var_repeat)

        prob_repeat=1/((2*np.pi*var_repeat)**0.5)*paddle.exp(-0.5*(mean_repeat-label_repeat)**2/var_repeat)+1e-8

        loss=-paddle.mean(paddle.log(paddle.mean(prob_repeat,axis=1)))
        # log_std=log_var/2
        # loss=paddle.mean(-(-0.5*log_var-0.5*(mean-label)**2/var))
        # loss=paddle.mean(-1/((2*np.pi*var)**0.5)*paddle.exp(-0.5*(mean-label)**2/var))

        optimizer.clear_grad()
        loss.backward()
        optimizer.step()

        if i %1000==0:
            print(i,"loss=",loss.numpy())
            show_data_mean,show_data_log_var=net(input_source.sample([show_batch_size]))
            show_data_std=paddle.exp(show_data_log_var/2)
            show_data_sample=Normal(show_data_mean,show_data_std).sample([5])
            print(show_data_sample.shape)
            plt.hist(show_data_sample.numpy().reshape([-1,label_dim]),bins=100,range=[-0.5,2.5])
            plt.show()

def test_gauss3():
    import os
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']=''
    "假设数据固定值，网络输出高斯分布，最大化概率"
    from paddle.distribution import Normal
    input_rand_dim=4
    # output_dim=2*5
    output_gauss_num=2
    label_dim=1
    batch_size=128
    show_batch_size=1000

    # repeat_times=30

    class gauss_net(nn.Layer):
        def __init__(self, name_scope=None, dtype="float32"):
            super().__init__(name_scope, dtype)

            nl_func=nn.LeakyReLU()

            self.layer=nn.Sequential(
                nn.Linear(input_rand_dim,16),nl_func,
                nn.Linear(16,16),nl_func,
                nn.Linear(16,16),nl_func,
                nn.Linear(16,16),nl_func,
                nn.Linear(16,output_gauss_num*2)
            )
        def forward(self,x):
            out=self.layer(x)
            mean=out[:,:output_gauss_num]
            log_var=out[:,output_gauss_num:]
            return mean,log_var


    
    net=gauss_net()
    input_source=Normal([0.0]*input_rand_dim,[1.0]*input_rand_dim)
    optimizer=optim.Adam(0.0001,parameters=net.parameters())


    for i in range(10000000000):
        mean,log_var=net(input_source.sample([batch_size]))
        # print(output.shape)
        # mean_repeat=paddle.reshape(mean,[batch_size,repeat_times,1])
        # log_var_repeat=paddle.reshape(log_var,[batch_size,repeat_times,1])

        label=paddle.cast(paddle.randint(0,3,[batch_size,1]),np.float32)
        # label_repeat=paddle.tile(paddle.reshape(label,[batch_size,1,1]),[1,repeat_times,1])

        



        # loss=paddle.mean((output-label)**2)
        var=paddle.exp(log_var)

        # print(var.shape,mean.shape,label.shape)

        prob_repeat=1/((2*np.pi*var)**0.5)*paddle.exp(-0.5*(mean-label)**2/var)+1e-8

        loss=-paddle.mean(paddle.log(paddle.mean(prob_repeat,axis=1)))
        # log_std=log_var/2
        # loss=paddle.mean(-(-0.5*log_var-0.5*(mean-label)**2/var))
        # loss=paddle.mean(-1/((2*np.pi*var)**0.5)*paddle.exp(-0.5*(mean-label)**2/var))

        optimizer.clear_grad()
        loss.backward()
        optimizer.step()

        # if i %1000==0:
        #     print(i,"loss=",loss.numpy())
        #     show_data_mean,show_data_log_var=net(input_source.sample([show_batch_size]))
        #     show_data_std=paddle.exp(show_data_log_var/2)
        #     show_data_sample=Normal(show_data_mean,show_data_std).sample([5])
        #     print(show_data_sample.shape)
        #     plt.hist(show_data_sample.numpy().reshape([-1,label_dim]),bins=100,range=[-0.5,2.5])
        #     plt.show()
        if i %1000==0:
            print(i,"loss=",loss.numpy())
            show_data_mean,show_data_log_var=net(input_source.sample([1]))
            show_data_var=paddle.exp(show_data_log_var)

            def gauss(x,mean,var):
                return 1/((2*np.pi*var)**0.5)*np.exp(-0.5*(x-mean)**2/var)


            show_x=np.linspace(-1,3,1000)
            # show_y
            show_y=0
            for g in range(show_data_mean.shape[1]):
                show_y_g=gauss(show_x,show_data_mean.numpy()[0][g],show_data_var.numpy()[0][g])
                plt.plot(show_x,show_y_g)
                show_y+=show_y_g
            plt.plot(show_x,show_y)
            plt.show()

            # show_data_std=paddle.exp(show_data_log_var/2)
            # # show_data_sample=Normal(show_data_mean,show_data_std).sample([5])
            # print(show_data_sample.shape)
            # plt.hist(show_data_sample.numpy().reshape([-1,label_dim]),bins=100,range=[-0.5,2.5])
            # plt.show()
def test_randn():
    # for i in range(1,20):
    #     a=paddle.randn([10000,i])
    #     a=paddle.mean(a**2,axis=1,keepdim=False)**0.5
    #     a=a.numpy()
    #     plt.hist(a**2,bins=100,range=[0,2])
    # plt.show()
    a=paddle.randn([10000,1])
    # print(paddle.mean(a**2))

    plt.hist(a**2,bins=100,range=[0,2])
    plt.show()
def test_ce():
    import os
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']=''
    def ce(x,y=0.05):
        return -y*np.log(x)-(1-y)*np.log(1-x)

    data=np.linspace(0.0001,0.99,1000)
    loss=ce(data)
    print(np.min(loss))

    plt.plot(data,loss,label="ce")
    plt.legend()
    plt.show()
    plt.savefig()
def test_sae_param():
    import os
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']=''
    npy_dir="all_models/sae_npy"

    fig,ax = plt.subplots()
    for f in os.listdir(npy_dir):
        file=f"{npy_dir}/{f}"
        loss=np.load(file,allow_pickle=True)
        avg_fire_loss_list=[l[1][-2] for l in loss]
        print(avg_fire_loss_list)
        ax.plot(avg_fire_loss_list,label=f)
    
    ax.legend()
    plt.show()
def test_sparse_loss():
    import os
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']=''
    hid=paddle.create_parameter([1000],'float32')
    optimizer=optim.Adam(0.001,parameters=[hid])


    def kl_loss(x,ratio):
        return -(ratio*paddle.log(x)+(1-ratio)*paddle.log(1-x))

    for i in range(10000000):
        hid_code=nn.functional.sigmoid(hid)
        code1=paddle.mean(hid_code**2)
        code2=paddle.mean((1-hid_code)**2)

        # loss=-paddle.mean(0.05*paddle.log(code1)+(1-0.05)*paddle.log(code2))
        # loss=20*paddle.abs(code1 - 0.05)**2 +paddle.abs(code2 - (1-0.05))**2
        loss=kl_loss(code1,0.05)+kl_loss(code2,1-0.05)

        optimizer.clear_grad()
        loss.backward()
        optimizer.step()

        if i %1000==0:
            print(i,loss.numpy(),code1.numpy(),code2.numpy())
            plt.hist(hid_code.numpy(),range=[0,1],bins=100)
            plt.show()
def test_info_entropy():
    def entropy(x):
        x=np.array(x)
        return np.sum(-x*np.log(x))
    def cross_entropy(x1,x2):
        x1=np.array(x1)
        x2=np.array(x2)
        return np.sum(-x1*np.log(x2+1e-8))
    x1=[0.1 for i in range(10)]
    x2=[0.00,*[(1-0.00)/9 for i in range(9)]]
    print(x1)
    print(x2)
    print(entropy(x1),entropy(x2))
    print(cross_entropy(x1,x2))
def test_expand():
    a=paddle.randn([3,3])
    b=a.reshape([3,1,3])
    c=paddle.tile(b,[1,5,1])
    print(a==c[:,0])
def test_tril():
    a=paddle.ones([5,5])
    b=paddle.tril(a)
    print(a,b)
def test_gauss_kl():
    from mlutils.ml import gauss_KL
    from paddle import distribution
    def sample_from_param(mean,sigma,num=10000):
        dis=distribution.Normal(mean,sigma)
        sample=dis.sample([num])
        return sample
    def plot_gauss(mean,sigma,x_data):
        dis=distribution.Normal(mean,sigma)
        probs=dis.probs(paddle.to_tensor(x_data))
        # print(probs)
        plt.plot(x_data,probs.numpy())


    mean=paddle.create_parameter([1],'float32')
    log_var=paddle.create_parameter([1],'float32')

    mean1=paddle.create_parameter([1],'float32')
    log_var1=paddle.to_tensor([-3.0])
    mean2=paddle.create_parameter([1],'float32')
    log_var2=paddle.to_tensor([-1.0])

    optimizer=optim.Adam(0.001,parameters=[mean,log_var,mean1,mean2])

    # x_grid=np.linspace(-2,2,50)
    # y_grid=np.linspace(-2,2,50)
    # grid=np.meshgrid(x_grid,y_grid)
    # grid=
    # print(np.shape(grid))

    for i in range(1000000):
        sigma=paddle.exp(log_var/2)
        sigma1=paddle.exp(log_var1/2)
        sigma2=paddle.exp(log_var2/2)

        loss=gauss_KL(mean1,sigma1,mean,sigma)+gauss_KL(mean2,sigma2,mean,sigma)
        # loss=gauss_KL(mean,sigma,mean1,sigma1)+gauss_KL(mean,sigma,mean2,sigma2)
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()

        if i %1000==0:
           
            


            print(i,loss.numpy())
            x_data=np.linspace(-4,4,1000).astype(np.float32)
            plot_gauss(mean,sigma,x_data)
            plot_gauss(mean1,sigma1,x_data)
            plot_gauss(mean2,sigma2,x_data)
            plt.show()
            # sample1=sample_from_param(mean1,sigma1,100000)
            # sample2=sample_from_param(mean2,sigma2,100000)
            # print(sample1.shape,sample2.shape)
            # sample_label=paddle.concat([sample1,sample2],axis=0).numpy()
            # sample=sample_from_param(mean,sigma,100000).numpy()

            # plt.subplot(211)
            # plt.hist2d(sample_label[:,1],sample_label[:,0],bins=50,range=[[-4,4],[-4,4]])
            # # plt.show()
            # plt1.subplot(212)
            
            # plt.hist2d(sample[:,1],sample[:,0],bins=50,range=[[-4,4],[-4,4]])
            # plt.show()

            
            # all_sample=np.concatenate([sample_label,sample],axis=0)
            # plt.hist2d(all_sample[:,1],all_sample[:,0],bins=50,range=[[-4,4],[-4,4]])
            # plt.show()


def train_UAE():
    from mlutils.ml import ReparamNormal
    train_dataset = paddle.vision.datasets.MNIST(mode='train')
    # print(len(train_dataset),np.shape(np.array(train_dataset[0][0])),train_dataset[0][1])
    img_dataset=[np.array(x[0],dtype=np.float32)/255 for x in train_dataset]
    print(np.shape(img_dataset))
    # img_dataset=[cv2.resize(np.array(x[0],dtype=np.float32)/255,dsize=(64,64)) for x in train_dataset]
    # img_dataset/=255
    # print(np.shape(img_dataset),np.max(img_dataset),np.min(img_dataset))
    # input()
    hid_dim=256
    batch_size=256
    encoder=dense_block([784,256,256,256,256,hid_dim*2])

    decoder=dense_block([hid_dim,256,256,256,256,256,784],act_output=nn.Sigmoid())

    optimizer=optim.Adam(learning_rate=0.0003,parameters=[*encoder.parameters(),*decoder.parameters()])

    repapram=ReparamNormal([batch_size,hid_dim])

    for i in range(10000000000):
        train_imgs=random.sample(img_dataset,batch_size)
        # train_imgs=np.expand_dims(train_imgs,axis=1)
        # train_imgs=np.repeat(train_imgs,3,axis=1)
        # print(np.shape(train_imgs))
        train_imgs_paddle=paddle.to_tensor(train_imgs).reshape([batch_size,784])

        x=encoder(train_imgs_paddle)

        mean,sigma=x[:,:hid_dim],nn.functional.softplus(x[:,hid_dim:])
        rand_delta=paddle.rand([x.shape[0],hid_dim])*2-1
        rand_sample=mean+rand_delta*sigma

        # mean,log_var=x[:,:hid_dim],x[:,hid_dim:]
        # rand_sample=repapram.sample(mean,log_var)


        restore_img=decoder(rand_sample)
        loss=paddle.mean((train_imgs_paddle-restore_img)**2)+paddle.mean((sigma-1.0)**2)#paddle.mean(sigma)*0.0001
        # print(loss)

        optimizer.clear_grad()
        loss.backward()
        optimizer.step()

        if i%100==0:
            print(i,loss.numpy()[0],paddle.mean(sigma).numpy()[0])
            show_img=np.reshape(train_imgs_paddle[0].numpy(),[28,28,1])
            show_restore=np.reshape(restore_img[0].numpy(),[28,28,1])
            cv2.imshow("show_img",cv2.resize(np.concatenate([show_img,show_restore],axis=1),dsize=None,fx=3,fy=3,interpolation=cv2.INTER_AREA))
            cv2.waitKey(10)

def test_uniform_area():
    def get_box(mean,sigma):
        min_axis=mean-sigma
        max_axis=mean+sigma
        return min_axis,max_axis
    def cross_area(mean1,sigma1,mean2,sigma2):
        min1,max1=get_box(mean1,sigma1)
        min2,max2=get_box(mean2,sigma2)
        paddle.any(max1<min2,axis=-1,keepdim=True)
        paddle.any(max2<min1,axis=-1,keepdim=True)

        have_cross=1-paddle.cast(paddle.any((paddle.minimum(max1,max2)-paddle.maximum(min1,min2))<=0,axis=-1,keepdim=True),'float32')
        cross_area=paddle.prod(paddle.minimum(max1,max2)-paddle.maximum(min1,min2),axis=-1,keepdim=True)*have_cross

        return cross_area
    def union_area(mean1,sigma1,mean2,sigma2):
        return area(mean1,sigma1)+area(mean2,sigma2)-cross_area(mean1,sigma1,mean2,sigma2)
    def border_area(mean1,sigma1,mean2,sigma2):
        min1,max1=get_box(mean1,sigma1)
        min2,max2=get_box(mean2,sigma2)

        min_border=paddle.minimum(min1,min2)
        max_border=paddle.maximum(max1,max2)

        border_area=paddle.prod(max_border-min_border,axis=-1,keepdim=True)
        return border_area
    def area(mean,sigma):
        min1,max1=get_box(mean,sigma)
        return paddle.prod(max1-min1,axis=-1,keepdim=True)

    def render(min_max_xy_list):
        bg_img_size=np.array([600,600])
        bg_img=np.zeros([*bg_img_size,3],np.uint8)
        all_minx,all_maxx,all_miny,all_maxy=0,0,0,0
        
        for min_max_xy in min_max_xy_list:
            min_xy,max_xy=min_max_xy
            all_minx=min(all_minx,min_xy[1])
            all_miny=min(all_miny,min_xy[0])
            all_maxx=max(all_maxx,max_xy[1])
            all_maxy=max(all_maxy,max_xy[0])

        center=[(all_miny+all_maxy)/2,(all_minx+all_maxx)/2]
        length=max(all_maxy-all_miny,all_maxx-all_minx)*1.2
        # print(min_max_xy_list,center,all_minx,all_maxx,all_miny,all_maxy)

        color_list=[(255,0,0),(0,255,0),(0,0,255)]
        for i,min_max_xy in enumerate(min_max_xy_list):
            min_xy,max_xy=min_max_xy

            min_yx_pixel=(((min_xy-center)/length+0.5)*bg_img_size).astype(np.int)
            max_yx_pixel=(((max_xy-center)/length+0.5)*bg_img_size).astype(np.int)
            cv2.rectangle(bg_img,(min_yx_pixel[1],min_yx_pixel[0]),(max_yx_pixel[1],max_yx_pixel[0]),color_list[i],3 if i==2 else -1)
        for i,min_max_xy in enumerate(min_max_xy_list):
            min_xy,max_xy=min_max_xy

            min_yx_pixel=(((min_xy-center)/length+0.5)*bg_img_size).astype(np.int)
            max_yx_pixel=(((max_xy-center)/length+0.5)*bg_img_size).astype(np.int)
            cv2.rectangle(bg_img,(min_yx_pixel[1],min_yx_pixel[0]),(max_yx_pixel[1],max_yx_pixel[0]),color_list[i],3)
        cv2.imshow("Img",bg_img)
        cv2.waitKey(100)

    def KL_loss(PA,PB,area_A,area_B,cross_area_AB,out_prob=1e-2):
        return cross_area_AB/area_B*paddle.log(PB/PA)+(1-cross_area_AB/area_B)*paddle.log(PB/out_prob)+(area_A-cross_area_AB)*out_prob*paddle.log(out_prob/PA)
        



    meanA=paddle.create_parameter([2],'float32')
    sigmaA_origin=paddle.create_parameter([2],'float32')

    meanB=paddle.create_parameter([2],'float32')
    sigmaB_origin=paddle.create_parameter([2],'float32')
    meanC=paddle.create_parameter([2],'float32')
    sigmaC_origin=paddle.create_parameter([2],'float32')

    generator=dense_block([2,8,8,8,8,8,8,8,2])

    optimizer=optim.Adam(0.01,parameters=[meanA,sigmaA_origin,meanB,sigmaB_origin,meanC,sigmaC_origin,*generator.parameters()])

    # x_grid=np.linspace(-2,2,50)
    # y_grid=np.linspace(-2,2,50)
    # grid=np.meshgrid(x_grid,y_grid)
    # grid=
    # print(np.shape(grid))

    for i in range(1000000):
        sigmaA=nn.functional.softplus(sigmaA_origin)
        sigmaB=nn.functional.softplus(sigmaB_origin)
        sigmaC=nn.functional.softplus(sigmaC_origin)

        area_A=area(meanA,sigmaA)
        area_B=area(meanB,sigmaB)
        area_C=area(meanC,sigmaC)
        PA=1/area_A
        PB=1/area_B
        PC=1/area_C

        cross_area_AB=cross_area(meanA,sigmaA,meanB,sigmaB)
        cross_area_AC=cross_area(meanA,sigmaA,meanC,sigmaC)
        cross_area_BC=cross_area(meanB,sigmaB,meanC,sigmaC)

        union_area_AB=union_area(meanA,sigmaA,meanB,sigmaB)
        union_area_AC=union_area(meanA,sigmaA,meanC,sigmaC)

        border_area_AB=border_area(meanA,sigmaA,meanB,sigmaB)
        border_area_AC=border_area(meanA,sigmaA,meanC,sigmaC)

        # loss_AB=(area_B-cross_area_AB)/area_B-cross_area_AB/area_A
        # loss_AC=(area_C-cross_area_AC)/area_C-cross_area_AC/area_A

        # loss_AB=(border_area_AB-union_area_AB)/border_area_AB+(area_B-cross_area_AB)/area_B*2-cross_area_AB/area_A
        # loss_AC=(border_area_AC-union_area_AC)/border_area_AC+(area_C-cross_area_AC)/area_C*2-cross_area_AC/area_A
        out_prob=1e-8
        loss_AB=KL_loss(PA,PB,area_A,area_B,cross_area_AB)
        loss_AC=KL_loss(PA,PC,area_A,area_C,cross_area_AC)

        # loss_BC=cross_area_BC/area_B*paddle.log(PB/PC)+cross_area_BC/area_C*paddle.log(PC/PB)
        # loss_BC=cross_area_BC/area_B+cross_area_BC/area_C
        # meanB,sigma=x[:,:hid_dim],nn.functional.softplus(x[:,hid_dim:])
        batch_size=128

        rand_deltaB=paddle.rand([batch_size,2])*2-1
        rand_sample_B=meanB+rand_deltaB*sigmaB
        result_B=generator(rand_sample_B)
        # loss_result_B=paddle.mean(paddle.nn.functional.softmax_with_cross_entropy(result_B,paddle.zeros([batch_size,1],'int64')))/PB
        loss_result_B=paddle.mean((result_B-paddle.to_tensor([1.0,0.0]))**2)

        rand_deltaC=paddle.rand([batch_size,2])*2-1
        rand_sample_C=meanC+rand_deltaC*sigmaC
        result_C=generator(rand_sample_C)
        # print(result_C.shape)
        # loss_result_C=paddle.mean(paddle.nn.functional.softmax_with_cross_entropy(result_C,paddle.ones([batch_size,1],'int64')))/PC
        loss_result_C=paddle.mean((result_C-paddle.to_tensor([0.0,1.0]))**2)
        

        


        loss=loss_AB*1+loss_AC*2+(loss_result_B+loss_result_C)*0.0
        # loss=gauss_KL(mean,sigma,mean1,sigma1)+gauss_KL(mean,sigma,mean2,sigma2)
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()

        if i %10==0:
           
            minA,maxA=get_box(meanA,sigmaA)
            minB,maxB=get_box(meanB,sigmaB)
            minC,maxC=get_box(meanC,sigmaC)

            min_max_xy_list=[

                [minB.numpy(),maxB.numpy()],
                [minC.numpy(),maxC.numpy()],
                [minA.numpy(),maxA.numpy()],
            ]

            print(i,loss.numpy()[0],loss_result_B.numpy()[0],loss_result_C.numpy()[0])
            print(sigmaA.numpy(),area_A.numpy()[0],area_B.numpy()[0],area_C.numpy()[0],(area_C/area_B).numpy()[0])
            render(min_max_xy_list)
def test_gnd():
    def gnd(a,n,b):
        if n==1:
            return a**b
        elif n>1:
            if b==1:
                return a
            elif b>1:
                return gnd(a,n-1,gnd(a,n,b-1))
    result=4
    for i in range(64):
        result=gnd(3,result,3)
    print(result)







    # print(mean)
def test_adam():
    def render(x,loss):
        img_size=300
        img=np.zeros([img_size,img_size],np.uint8)
        site_w=int(x*img_size/2+img_size/2)
        site_h=int(loss*img_size)
        cv2.circle(img,(site_w,site_h),5,(255),-1)
        cv2.imshow("img",img)
        cv2.waitKey(10)
    x=paddle.create_parameter([1],'float32')
    optimizer=optim.Momentum(0.001,momentum=0.99,parameters=[x])
    while 1:
        loss=x**2
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()
        print(x.numpy()[0])
        render(x.numpy()[0],loss.numpy()[0])
def test_minigrid_fourroom():
    from env.minigrid import MiniGrid_fourroom
    env=MiniGrid_fourroom()
    t1=time.time()
    for i in range(100):
        env.reset()
    print("reset cost",time.time()-t1)
    done=False
    t2=time.time()
    while not done:
        obs,reward,done=env.step(np.clip(np.random.randint(0,4),0,2))
    print("step cost",time.time()-t2)
        # env.render()
        # print(np.shape(obs),reward,done)
def test_global():
    global a
    def f1():
        global a
        a+=1
        print("f1",a)
    def f2():
        global a
        a+=1
        print("f2",a)
    f1()
    f2()
def test_gru_time():
    v_encoder_hid_c=4
    v_z_c=4
    v_decoder_hic_c=4

    def nl_func():
        return nn.LeakyReLU(0.1)

    class Conv_Encoder_net(nn.Layer):
        def __init__(self,img_shape):
            super(Conv_Encoder_net,self).__init__()
            hid_c=v_encoder_hid_c
            self.conv_net=nn.Sequential(
                nn.Conv2D(
                    in_channels=img_shape[0],out_channels=hid_c,
                    kernel_size=[3,3],stride=[2,2],padding='SAME'),nl_func(),
                nn.Conv2D(
                    in_channels=hid_c,out_channels=hid_c,
                    kernel_size=[3,3],stride=[2,2],padding='SAME'),nl_func(),
                nn.Conv2D(
                    in_channels=hid_c,out_channels=hid_c,
                    kernel_size=[3,3],stride=[2,2],padding='SAME'),nl_func(),

                nn.Conv2D(
                    in_channels=hid_c,out_channels=v_z_c,
                    kernel_size=[3,3],stride=[1,1],padding='SAME'),
                nl_func()
            )
        def forward(self,img):
            x=img
            x=self.conv_net(x)
            return x
    class Conv_Decoder_net(nn.Layer):
        def __init__(self):
            super(Conv_Decoder_net,self).__init__()
            hid_c=v_decoder_hic_c
            self.deconv_net=nn.Sequential(
                nn.Conv2D(
                    in_channels=v_z_c,out_channels=hid_c,
                    kernel_size=[3,3],stride=[1,1],padding='SAME'),nl_func(),
                # Reslayer(
                #     nn.Conv2D(
                #         in_channels=hid_c,out_channels=hid_c,
                #         kernel_size=[3,3],stride=[1,1],padding='SAME'),nl_func()
                # ),
                nn.Conv2DTranspose(
                    in_channels=hid_c,out_channels=hid_c,
                    kernel_size=[3,3],stride=2,padding="SAME"),nl_func(),
                nn.Conv2DTranspose(
                    in_channels=hid_c,out_channels=hid_c,
                    kernel_size=[3,3],stride=2,padding="SAME"),nl_func(),
                nn.Conv2DTranspose(
                    in_channels=hid_c,out_channels=hid_c,
                    kernel_size=[3,3],stride=2,padding="SAME"),nl_func(),
                # nn.Conv2DTranspose(
                #     in_channels=hid_c,out_channels=hid_c,
                #     kernel_size=[3,3],stride=1,padding="SAME"),nl_func(),
                nn.Conv2DTranspose(
                    in_channels=hid_c,out_channels=3,
                    kernel_size=[3,3],stride=1,padding="SAME"),nn.Sigmoid(),
                )
        def forward(self,img):
            return self.deconv_net(img)
    def btd_layer(net,x):
        batch,time,dim=x.shape
        return net(x.reshape([batch*time,dim])).reshape([batch,time,-1])
    def btchw_layer(convnet,x):
        batch,time,c,h,w=x.shape
        conv_result=convnet(x.reshape([batch*time,c,h,w]))
        _,newc,newh,neww=conv_result.shape
        return conv_result.reshape([batch,time,newc,newh,neww])

    train_batchsize=128
    gru_dim=256
    feature_dim=256
    history_len=100

    "no conv net 0.02s/batch"
    # encoder_gru=nn.GRU(feature_dim,gru_dim)
    # decoder_gru=nn.GRU(feature_dim,gru_dim)
    # optimizer=optim.Adam(0.0003,parameters=[
    #     *encoder_gru.parameters(),
    #     *decoder_gru.parameters()])

    # while 1:
    #     t1=time.time()
    #     data=paddle.uniform([train_batchsize,history_len,feature_dim],'float32')
    #     encoder_hid,_=encoder_gru(data,paddle.zeros([1,train_batchsize,gru_dim]))
    #     output,_=decoder_gru(data,paddle.transpose(encoder_hid[:,-1:,:],[1,0,2]))
    #     loss=paddle.mean((data-output)**2)
    #     optimizer.clear_grad()
    #     loss.backward()
    #     optimizer.step()

    #     print("train cost",time.time()-t1)

    "conv gru 0.5s"
    # encoder_input_conv=Conv_Encoder_net([3,64,64])
    # encoder_gru=nn.GRU(feature_dim,gru_dim)
    # decoder_gru=nn.GRU(feature_dim,gru_dim)
    # decoder_output_conv=Conv_Decoder_net()

    # optimizer=optim.Adam(0.0003,parameters=[
    #     *encoder_input_conv.parameters(),
    #     *encoder_gru.parameters(),
    #     *decoder_gru.parameters(),
    #     *decoder_output_conv.parameters(),
    #     ])

    # while 1:
    #     t1=time.time()
    #     data=paddle.uniform([train_batchsize,history_len,3,64,64],'float32')
    #     conv_data=btchw_layer(encoder_input_conv,data).reshape([train_batchsize,history_len,-1])

    #     encoder_hid,_=encoder_gru(conv_data,paddle.zeros([1,train_batchsize,gru_dim]))

    #     decoder_hid,_=decoder_gru(conv_data,paddle.transpose(encoder_hid[:,-1:,:],[1,0,2]))
        
    #     pred_img=btchw_layer(decoder_output_conv,paddle.reshape(decoder_hid,[train_batchsize,history_len,4,8,8]))
    #     loss=paddle.mean((data-pred_img)**2)

    #     optimizer.clear_grad()
    #     loss.backward()
    #     optimizer.step()

    #     print("train cost",time.time()-t1)

    "conv + gru"
    encoder_input_conv=Conv_Encoder_net([3,64,64])
    encoder_gru=nn.GRU(feature_dim,gru_dim)
    decoder_gru=nn.GRU(feature_dim,gru_dim)
    decoder_output_conv=Conv_Decoder_net()

    optimizer=optim.Adam(0.0003,parameters=[
        *encoder_input_conv.parameters(),
        *encoder_gru.parameters(),
        *decoder_gru.parameters(),
        *decoder_output_conv.parameters(),
        ])

    while 1:
        t1=time.time()
        data=paddle.uniform([train_batchsize,history_len,3,64,64],'float32')
        conv_data=btchw_layer(encoder_input_conv,data)#.reshape([train_batchsize,history_len,-1])
        recon_data=btchw_layer(decoder_output_conv,paddle.reshape(conv_data,[train_batchsize,history_len,4,8,8]))
        loss1=paddle.mean((data-recon_data)**2)

        # conv_data_for_gru=conv_data.reshape([train_batchsize,history_len,-1]).detach()
        # encoder_hid,_=encoder_gru(conv_data_for_gru,paddle.zeros([1,train_batchsize,gru_dim]))
        # decoder_hid,_=decoder_gru(conv_data_for_gru,paddle.transpose(encoder_hid[:,-1:,:],[1,0,2]))
        # loss2=paddle.mean((conv_data_for_gru-decoder_hid)**2)
        # # pred_img=btchw_layer(decoder_output_conv,paddle.reshape(decoder_hid,[train_batchsize,history_len,4,8,8]))
        # loss=loss1+loss2

        optimizer.clear_grad()
        loss1.backward()
        optimizer.step()

        print("train cost",time.time()-t1)
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/peterjc123/
 
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/

# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
# conda config --show-sources
# conda install paddlepaddle==2.1.2 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
# ————————————————
# 版权声明：本文为CSDN博主「Mr.Q」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/jizhidexiaoming/article/details/122874488
def test_cd():
    from paddle.distribution import Normal
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
                code_2=(np.random.uniform(0, 1, self.dim)<probs).astype(np.int32)
            code_10=np.sum(code_2*self.prod_list)
            if code_10<self.discrete_num:
                return code_10
            else:
                return np.random.randint(0,self.discrete_num)
    # from mlutils.ml import continue_to_discrete
    from matplotlib import pyplot as plt
    import os
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']=''

    cd=continue_to_discrete(2,False,tanh_scale=100)
    x=np.linspace(-1,1,1000)
    probs=cd.to_prob(x)*800

    center_mean_prob_list=[]
    for center in np.linspace(0,1,100):
        normal=Normal(center,np.exp(-1))
        sample=normal.sample([10000]).numpy()
        sample_to_prob=cd.to_prob(sample)
        # print(sample)
        mean_prob=np.mean(sample_to_prob)
        center_mean_prob_list.append(mean_prob)
    print(center_mean_prob_list)
    # plt.plot(np.linspace(0,1,100),center_mean_prob_list)
    # plt.show()
    
    # print("mean prob",np.mean(sample_to_prob))
    # print(sample)
    # print(probs)
    # plt.plot()

    plt.hist(sample,bins=50)
    print(np.mean(sample<0))
    plt.plot(x,probs)
    
    plt.show()
def test_nan():
    a=[1,float('-inf'),2]
    print(np.min(a),np.max(a),np.mean(a))
    print(a)

def test_log_prob():

    from paddle.distribution import Normal
    while 1:
        action_mean=paddle.uniform([3],'float32',0,0.0000001)
        action_logstd=paddle.uniform([3],'float32',-20,-10)
        dist = Normal(action_mean,action_logstd.exp())
        action = dist.sample([1])[0]
        print(action.numpy())
        # print("action_sample",action.shape)
        action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)
        log_prob=action_log_probs.numpy()[0]
        print(log_prob)
        print(paddle.exp(paddle.to_tensor([300.],'float32')).numpy())

def test_cal_minigrid_inner_reward():

    from env.minigrid_neverdone import MiniGrid_fourroom_obs8x8
    from matplotlib import pyplot as plt
    env=MiniGrid_fourroom_obs8x8()
    env.reset()

    contain_goal_inner_reward_list=[]
    no_goal_inner_reward_list=[]
    contain_goal_real_kl_reward_list=[]
    no_goal_real_kl_reward_list=[]
    contain_goal_pred_kl_reward_list=[]
    no_goal_pred_kl_reward_list=[]
    max_ep=1000
    ep_num=0
    while 1:
        _,_,_,info=env.step(np.random.randint(0,3))
        # print(info)
        if info['inner_done']:
            if info['contain_goal']:
                contain_goal_inner_reward_list.append(info['ep_inner_reward'])
                contain_goal_real_kl_reward_list.append(info['ep_inner_real_reward'])
                contain_goal_pred_kl_reward_list.append(info['ep_inner_pred_reward'])
            else:
                no_goal_inner_reward_list.append(info['ep_inner_reward'])
                no_goal_real_kl_reward_list.append(info['ep_inner_real_reward'])
                no_goal_pred_kl_reward_list.append(info['ep_inner_pred_reward'])

            ep_num+=1
            print(ep_num)
            print(np.mean(contain_goal_inner_reward_list),np.mean(contain_goal_real_kl_reward_list),np.mean(contain_goal_pred_kl_reward_list))
            print(np.mean(no_goal_inner_reward_list),np.mean(no_goal_real_kl_reward_list),np.mean(no_goal_pred_kl_reward_list))
        if ep_num>=max_ep:
            break
    # print(contain_goal_reward_list,no_goal_reward_list)
    np.save("cal.npy",[np.array(x) for x in [contain_goal_inner_reward_list,no_goal_inner_reward_list,contain_goal_real_kl_reward_list,no_goal_real_kl_reward_list,contain_goal_pred_kl_reward_list,no_goal_pred_kl_reward_list]])
    plt.hist(no_goal_inner_reward_list,bins=50,alpha = 0.3)
    plt.hist(contain_goal_inner_reward_list,bins=50,alpha = 0.3)
    plt.show()

    plt.hist(no_goal_real_kl_reward_list,bins=50,alpha = 0.3)
    plt.hist(contain_goal_real_kl_reward_list,bins=50,alpha = 0.3)
    plt.show()

    plt.hist(no_goal_pred_kl_reward_list,bins=50,alpha = 0.3)
    plt.hist(contain_goal_pred_kl_reward_list,bins=50,alpha = 0.3)
    plt.show()

def test_plt_hist_to_plot():
    from matplotlib import pyplot as plt
    x=[1,1,1,2,3,4,5,5,5,6,7,8,8,9]
    y=np.array(x)+2
    def hist_to_plot(x,alpha):
        # plt.
        a,b,_=plt.hist(x,alpha = alpha)
        # print(a,b)
        plt.scatter(b[:-1]+0.5*(b[-1]-b[-2]),a)
    hist_to_plot(x,1)
    hist_to_plot(y,0.3)
    plt.show()
def show_cal_npy():
    from matplotlib import pyplot as plt
    data=np.load("cal.npy",allow_pickle=True)
    contain_goal_inner_reward_list,no_goal_inner_reward_list,contain_goal_real_kl_reward_list,no_goal_real_kl_reward_list,contain_goal_pred_kl_reward_list,no_goal_pred_kl_reward_list=data

    _range=[2,15]
    plt.hist(no_goal_inner_reward_list,bins=50,alpha = 0.3,range=_range)
    plt.hist(contain_goal_inner_reward_list,bins=50,alpha = 0.3,range=_range)
    plt.show()

    plt.hist(no_goal_real_kl_reward_list,bins=50,alpha = 0.3,range=_range)
    plt.hist(contain_goal_real_kl_reward_list,bins=50,alpha = 0.3,range=_range)
    plt.show()

    plt.hist(no_goal_pred_kl_reward_list,bins=50,alpha = 0.3,range=_range)
    plt.hist(contain_goal_pred_kl_reward_list,bins=50,alpha = 0.3,range=_range)
    plt.show()
def test_show_reward():

    from env.minigrid_neverdone import MiniGrid_fourroom_obs8x8
    from matplotlib import pyplot as plt
    env=MiniGrid_fourroom_obs8x8()
    env.reset()

    contain_goal_inner_reward_list=[]
    no_goal_inner_reward_list=[]
    contain_goal_real_kl_reward_list=[]
    no_goal_real_kl_reward_list=[]
    contain_goal_pred_kl_reward_list=[]
    no_goal_pred_kl_reward_list=[]
    max_ep=1000
    ep_num=0
    while 1:
        _,_,_,info=env.step(np.random.randint(0,3))
        print(info)
        env.render()
def test_perfect_env():
    from env.minigrid_neverdone_perfect_reward import MiniGrid_fourroom_obs8x8
    env=MiniGrid_fourroom_obs8x8()
    env.reset()
    while 1:
        obs,reward,done,info=env.step(np.clip(np.random.randint(0,4),0,2))
        print(info)
        env.render()
def test_Categorical():
    from paddle.distribution import Normal, Categorical,multinomial
    class m_Categorical(Categorical):
        def entropy(self):
            dist_sum = paddle.sum(self.logits,axis=-1, keepdim=True)
            prob = self.logits / dist_sum


            neg_entropy = paddle.sum(
                prob * (paddle.log(prob)),axis=-1, keepdim=True)
            entropy = -neg_entropy
            return entropy

    probs=[0.0,1.0,-1e-8]
    print(Categorical(paddle.to_tensor(probs,'float32')).entropy())
    print(m_Categorical(paddle.to_tensor(probs,'float32')).entropy())
    print(Categorical(paddle.to_tensor(probs,'float32')).sample([1000]))


    std_probs=probs/np.sum(probs)

def test_one_hot():
    x=paddle.randint(0,5,[2,3])
    print(paddle.nn.functional.one_hot(x,10).shape)
def test_gauss_sample_ratio():
    from paddle.distribution import Normal
    import paddle
    normal=Normal([1.],[np.exp(-0.5)])
    sample=normal.sample([1000000])
    print(paddle.mean(paddle.cast(sample<0,'float32')))
def test_beta():
    # from paddle.distribution import Beta
    # print(Beta)
    def beta(x,a,b):
        return x**(a-1)*(1-x)**(b-1)
    
    x=np.linspace(0.0000000,1.0,1000)
    y=beta(x,6.4,2)
    print(np.sum(y[:500]),np.sum(y[500:]))
    print(y)
    p1=np.sum(y[:500])
    p2=np.sum(y[500:])

    print(p2/(p1+p2))
    from matplotlib import pyplot as plt
    plt.plot(x,y)
    plt.show()
def paddle_cmake():
    "cmake .. -DPY_VERSION=3.7 -DPYTHON_EXECUTABLE=/home/kylin/Program/python37_v2/bin/python3.7 -DPYTHON_INCLUDE_DIR=/home/kylin/Program/python37_v2/include -DPYTHON_LIBRARY=/home/kylin/Program/python37_v2/lib -DWITH_ARM=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DON_INFER=ON -DWITH_XBYAK=OFF"
def test_minigridlife():
    from env.minigrid_life import MiniGridLife
    env=MiniGridLife()
    obs= env.reset()
    while 1:
        obs,reward,done,info=env.step(np.random.randint(0,4))
        print(np.shape(obs),obs[-2:],reward,done,info)
        env.render()
def test_breakout():
    import gym
    # from gym import envs
    # print(envs.registry.all())
    # BreakoutDeterministic-v4
    # BreakoutNoFrameskip-v4
    env = gym.make('BreakoutDeterministic-v4')
    obs = env.reset()
    step = 0
    while True:
        obs,reward,done,info= env.step(env.action_space.sample())
        # env.render()
        print(np.shape(obs))
        # obs_mini=cv2.resize(obs,(64,84),interpolation=cv2.INTER_LINEAR)
        cv2.imshow("obs",cv2.resize(obs[2:,:,::-1],None,fx=4,fy=4,interpolation=cv2.INTER_AREA))
        cv2.waitKey()
        step += 1
        if done:
            break
def test_class_contrib():
    class vae_info():
        test_info=1
    print(vae_info.test_info)
def test_png():
    png1=cv2.imread("/data/E/myf/2022/公务文件/12月执法证考试报名/431122199405170514.png",cv2.IMREAD_UNCHANGED)
    print(np.shape(png1))
    for i in range(4):
        cv2.imshow("png",png1[:,:,i:i+1])
        cv2.waitKey()
    cv2.imwrite("/data/E/myf/2022/公务文件/12月执法证考试报名/431122199405170514_2.png",png1,[cv2.IMWRITE_PNG_COMPRESSION,9,cv2.IMWRITE_PNG])
def test_elbo_fit():
    from paddle.distribution import Normal
    from matplotlib import pyplot as plt
    from mlutils.ml import ReparamNormal
    def hist_data(x):
        x.numpy().reshape([-1])
        plt.hist(x,bins=100)
        plt.show()
    def ori_data_sample(batch_size):
        dis1=Normal(paddle.to_tensor([-1.],'float32'),paddle.to_tensor([0.3],'float32'))
        dis2=Normal(paddle.to_tensor([1.],'float32'),paddle.to_tensor([0.3],'float32'))

        sample1=dis1.sample([batch_size//2])
        sample2=dis2.sample([batch_size//2])
        return paddle.concat([sample1,sample2])
    # sample=ori_data_sample(64000)
    # hist_data(sample)
    class vae(nn.Layer):
        def __init__(self, name_scope=None, dtype="float32"):
            super().__init__(name_scope, dtype)
            self.gauss_dim=128
            self.batch_size=512
            self.bottle_dim=16
            self.encoder=dense_block([1,*[self.bottle_dim]*4,self.gauss_dim*2])
            self.decoder=dense_block([self.gauss_dim,*[self.bottle_dim]*4,1])

            self.target_gauss_mean=paddle.create_parameter([self.gauss_dim],'float32')
            self.target_gauss_log_var=paddle.create_parameter([self.gauss_dim],'float32')

            self.optimizer=optim.Adam(0.0001,parameters=[*self.encoder.parameters(),*self.decoder.parameters(),self.target_gauss_mean,self.target_gauss_log_var])

            self.train()

        def train(self):
            for i in range(1000000000000):
                train_data=ori_data_sample(self.batch_size)
                gauss_param=self.encoder(train_data)
                gauss_mean,gauss_log_var=gauss_param[:,:self.gauss_dim],gauss_param[:,self.gauss_dim:]
                sample=ReparamNormal([self.batch_size,self.gauss_dim]).sample(gauss_mean,gauss_log_var)
                recon=self.decoder(sample)
                loss_recon=paddle.mean((recon-train_data)**2)


                dis=Normal(gauss_mean,paddle.exp(gauss_log_var/2))
                target_dis=Normal(self.target_gauss_mean,paddle.exp(self.target_gauss_log_var/2))
                loss_kl=paddle.mean(dis.kl_divergence(target_dis))

                loss=loss_recon+loss_kl
                self.optimizer.clear_grad()
                loss.backward()
                self.optimizer.step()

                if i%1000==0:
                    print(i,loss.numpy()[0],loss_recon.numpy()[0],loss_kl.numpy()[0])
                    target_sample=target_dis.sample([10000])
                    target_recon=self.decoder(target_sample)
                    hist_data(target_recon)
     
    vae()
def test_fast_mem_seq():
    from paddle.nn.functional import sigmoid
    from paddle.nn import GRU
    from matplotlib import pyplot as plt
    sparse_ratio=0.05
    neuron_num=1000
    sparse_num=int(sparse_ratio*neuron_num)

    prob_weight=paddle.create_parameter([neuron_num,neuron_num],'float32')
    optimizer1=optim.SGD(0.005,parameters=[prob_weight])

    # gru=GRU(sparse_num,)

    def sparse_code(value):
        vec=np.zeros([1,neuron_num])
        index_center=int(value*1000)
        vec[:,index_center-sparse_num//2:index_center+sparse_num//2]=1.0
        return paddle.to_tensor(vec,'float32')
    seq1=[0.1,0.5,0.9,0.4,0.55,0.6,0.53,0.82]
    seq2=[0.1,0.4,0.5,0.6,0.9,0.55,0.39,0.53]

    def train_seq_weight(seq):
        # prev_code=None
        pred_v_code_list=[]
        v_code_list=[]
        sig_weight=prob_weight
        for value in seq:
            v_code=sparse_code(value)
            v_code_list.append(v_code)
            # print(v_code.shape)
            pred_v_code=paddle.matmul(v_code,sig_weight)
            pred_v_code_list.append(pred_v_code)
        target_all=paddle.concat(v_code_list[1:])
        pred_all=paddle.concat(pred_v_code_list[:-1],axis=0)
        pred_loss=paddle.sum((target_all-pred_all)**2)
        loss=pred_loss
        optimizer1.clear_grad()
        loss.backward()
        optimizer1.step()
        return loss.numpy()[0],target_all.numpy(),pred_all.numpy()
    # def train_seq_gru(seq):


    while 1:
        loss1_list=[]
        for i in range(20):
            loss,target_all1,pred_all1=train_seq_weight(seq1)
            loss1_list.append(loss)
            print(f"{i} seq1 loss={loss}")
            cv2.imshow("target1",cv2.resize(target_all1,None,fx=1,fy=20,interpolation=cv2.INTER_NEAREST))
            cv2.imshow("pred1",cv2.resize(pred_all1,None,fx=1,fy=20,interpolation=cv2.INTER_NEAREST))
            cv2.waitKey()
            
        loss2_list=[]
        for i in range(20):
            loss,target_all2,pred_all2=train_seq_weight(seq2)
            loss2_list.append(loss)
            print(f"{i} seq2 loss={loss}")
            cv2.imshow("target2",cv2.resize(target_all2,None,fx=1,fy=20,interpolation=cv2.INTER_NEAREST))
            cv2.imshow("pred2",cv2.resize(pred_all2,None,fx=1,fy=20,interpolation=cv2.INTER_NEAREST))
            cv2.waitKey()
            

        plt.plot(loss1_list)
        # plt.show()
        plt.plot(loss2_list)
        plt.show()
        
        

def test_state_dict():
    
    from multiprocessing import Process,Queue,Pipe

    def psend(pipe_send):
        from ec.conv_VAE import ConvVAE
        import paddle
        from paddle import fluid
        from paddle.fluid.framework import EagerParamBase
        
        cvae=ConvVAE()

        sd=cvae.state_dict()
        sd_numpy={}
        for key in sd.keys():
            sd_numpy[key]=sd[key].numpy()
        pipe_send.send(sd_numpy)


    def precv(pipe_recv):
        from ec.conv_VAE import ConvVAE
        import paddle
        cvae=ConvVAE()
        sd=cvae.state_dict()
        sd_numpy=pipe_recv.recv()
        
        # print(sd)
        for key in sd_numpy:
            print("before",paddle.mean(sd[key]).numpy()[0],np.mean(sd_numpy[key]),sd_numpy[key].dtype)
            sd[key].set_value(sd_numpy[key])
            print("after",paddle.mean(cvae.state_dict()[key]).numpy()[0])
        print("update_value success")

    p_send,p_recv=Pipe()
    p1=Process(target=psend,args=(p_send,))
    p2=Process(target=precv,args=(p_recv,))
    p_list=[p1,p2]
    [p.start() for p in p_list]
    [p.join() for p in p_list]

def test_class():
    class A():
        def __init__(self) -> None:
            pass
        def func1(self):
            print("A func1")
    def modify_A(A):
        def func2():
            A.func1()
        A.func1=func2
    # class B():
    #     def __init__(self,A) -> None:
    #         def func2():
    #             A.func
    #         A.func1=func2
    a=A()
    modify_A(a)
    a.func1()
def test_change_color():
    import cv2

    img=cv2.imread("/home/kylin/桌面/图片1.png")
    float_img=np.array(img,np.float32)
    blue,green,red=float_img[:,:,0:1],float_img[:,:,1:2],float_img[:,:,2:]
    cv2.imwrite("/home/kylin/桌面/图片1_gray.png",red*0.8+green*0.2)
    cv2.imshow("img",(red*0.6+green*0.4)/255)
    cv2.waitKey()
def test_gather_nd():
    import paddle
    x=paddle.randint(0,100,(1,64,64,3))
    index=paddle.to_tensor(np.random.randint(0,64,(10,10,3)),'int64')
    index[:,:,0]=0
    gather_data=paddle.gather_nd(x,index)
    print(gather_data.shape)

    a=paddle.ones([3,3],'float32')
    a[:2]+=1
    print(a)
def test_fisheye():
    from matplotlib import pyplot as plt
    import numpy as np
    def func1(x):
        return 4*x**2
    x=np.linspace(0,1,1000)

    y1=func1(x)
    plt.plot(x,x)
    plt.plot(x,y1)
    plt.show()
def test_fisheye_img():
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    "save img"
    # import gym
    # env = gym.make('BreakoutDeterministic-v4')
    # obs = env.reset()
    # step = 0
    # for i in range(10):
    #     obs,reward,done,info= env.step(env.action_space.sample())
    #     cv2.imwrite(f"all_data/breakout/{i}.png",obs)

    def rfunc1(r):
        return 4*r**2
    def trans_func(ori_yx):
        ori_r=(ori_yx[:,:,:1]**2+ori_yx[:,:,1:]**2)**0.5
        trans_r=rfunc1(ori_r)
        trans_yx=ori_yx*trans_r/ori_r
        return trans_yx
    def plot_yxr(yx,r):
        fig=plt.figure()
        ax=fig.gca(projection='3d')
        # r=(yx[:,:,:1]**2+yx[:,:,1:]**2)**0.5
        ax.plot_surface(yx[:,:,0],yx[:,:,1],r[:,:,0])
        plt.show()



    x=np.linspace(0,63,64)
    y=np.linspace(0,63,64)
    ori_yx=np.meshgrid(y,x)
    ori_yx=np.transpose(ori_yx,[2,1,0])

    ori_yx-=31.5
    ori_r=(ori_yx[:,:,:1]**2+ori_yx[:,:,1:]**2)**0.5

    trans_yx=trans_func(ori_yx)
    trans_r=(trans_yx[:,:,:1]**2+trans_yx[:,:,1:]**2)**0.5

    # plot_yxr(ori_yx,trans_r)
    img=cv2.imread(f"all_data/breakout/1.png")
    cv2.imshow("img",img)
    cv2.waitKey()
def test_np_slice():
    data_size=100000
    x=np.random.uniform(0,1,[data_size,256])

    import time
    seq_len=400

    t1=time.time()
    for i in range(10000):
        start_index=np.random.randint(0,data_size-seq_len)

        # y=x[start_index:start_index+seq_len]

        # slices=list(range(start_index,start_index+seq_len))
        # y=x[slices]
        # print(y.shape)

        slices=[slice(start_index,start_index+seq_len,1),slice(5,200,1)]
        y=x[slices]
        print(y.shape)
    print(f"cost {time.time()-t1}")
def test_entropy():
    p_list=[0.0001,0.0001,0.9998]
    print(np.sum(p_list*np.log(p_list)))
    
def test_derk():
    from gym_derk.envs import DerkEnv

    env = DerkEnv(
    n_arenas=1 # Change to 100 for instance, to run many instances in parallel
    )

    observation_n = env.reset()

    while True:
        # Take random actions each step
        action_n = [env.action_space.sample() for i in range(env.n_agents)]
        observation_n, reward_n, done_n, info_n = env.step(action_n)

        if all(done_n):
            break
        env.close()
def test_class_attribute():
    from hyperparam import Hybrid_PPO_HyperParam as hp
    # inspect
    # print(type(dir(hp)[0]))
    for attr in dir(hp):
        if attr.startswith("__") and attr.endswith("__"):
            print(None)
        else:
            print(attr,getattr(hp,attr))
    
def get_question_from_html():
    import lxml
    from lxml import etree
    from bs4 import BeautifulSoup
    import bs4
    
    fpath="/home/kylin/桌面/196__index.html"
    fcontent=open(fpath,"r").readlines()
    html_content=""
    for c in fcontent:
        html_content+=c
    # print(html_content)
    soup=BeautifulSoup(html_content,"html.parser")

    question_title_list=soup.find_all(attrs={'class',"ui-question-title"})

    question_list=[qt.parent for qt in question_title_list]

    for question in question_list:
        title=question.find(class_="ui-question-title").find(class_="ui-question-content-wrapper").text.replace("\n","")
        print(title)
        answers=[ans.text.replace("\n","") for ans in question.find(class_="ui-question-options").find_all(class_="ui-question-content-wrapper")]
        print(answers)
def get_question_and_answer():
    import lxml
    from lxml import etree
    from bs4 import BeautifulSoup
    import bs4
    import os
    fpath="/home/kylin/桌面/执法题库答案html/134.html"

    def get_questions_and_answer_from_html(html_path):
        fcontent=open(html_path,"r").readlines()

        html_content=""
        for c in fcontent:
            html_content+=c
        soup=BeautifulSoup(html_content,"html.parser")
        question_title_list=soup.find_all(attrs={'class',"ui-question-title"})
        question_list=[qt.parent for qt in question_title_list]
        question_and_answer_list=[]
        choice_tag=["A","B","C","D","E","F"]

        title_txt_list=[]
        answer_txt_list=[]
        correct_choices_txt_list=[]
        for _q,question in enumerate(question_list):
            title=question.find(class_="ui-question-title").find(class_="ui-question-content-wrapper").text.replace("\n","")
            answers=[ans.text.replace("\n","") for ans in question.find(class_="ui-question-options").find_all(class_="ui-question-content-wrapper")]
            correct_answers=[ans.find(class_="ui-question-content-wrapper").text.replace("\n","") for ans in question.find(class_="ui-question-options").find_all(class_="ui-correct-answer")]
            correct_choices=[choice_tag[answers.index(ca)] for ca in correct_answers]

            question_and_answer_list.append([title,answers,correct_choices])

            title_txt_list.append(title)
            answer_txt_list.append(answers)
            correct_choices_txt_list.append(correct_choices)


        return title_txt_list,answer_txt_list,correct_choices_txt_list,question_and_answer_list


    def write_text(txt_path):
        text_content=open(txt_path,"w")
        correct_choice_str=""
        correct_choice_str+=" 正确答案："
        for cc in correct_choices:
            correct_choice_str+=cc
        correct_choice_str+="\n"
        text_content.write(f"{_q+1}.{title}{correct_choice_str}\n")
        
        answer_str=""
        for _c,ans in enumerate(answers):
            answer_str+=f"{choice_tag[_c]}.{ans} "
        answer_str+="\n"
        text_content.write(answer_str)


    # question_html_dir="/home/kylin/桌面/执法题库html"

    all_title_txt_list,all_answer_txt_list,all_correct_choices_txt_list,all_question_and_answer_list=[],[],[],[]
    answer_html_dir="/home/kylin/桌面/执法题库答案html"
    for f in os.listdir(answer_html_dir):
        answer_html=f"{answer_html_dir}/{f}"
        title_txt_list,answer_txt_list,correct_choices_txt_list,question_and_answer_list=get_questions_and_answer_from_html(answer_html)
        # print(title_txt_list,answer_txt_list,correct_choices_txt_list)
        all_question_and_answer_list+=question_and_answer_list
        all_title_txt_list+=title_txt_list
    print(len(all_title_txt_list))
    print(len(set(all_title_txt_list)))
    
def get_web_question():
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    browser = webdriver.Chrome()
    # class="ui-question-options-order"
    
    browser.get(url="https://hnzfks.edu-edu.com/")
    browser.maximize_window()

    while 1:
        
        input()
        print(len(browser.window_handles))
        x=input()
        browser.switch_to.window(browser.window_handles[int(x)])
        content = browser.page_source
        browser.find
        print(browser.find_elements(By.CLASS_NAME,"ui-question-content-wrapper"))
        # browser.find_element()
        # print(content)
        # open("timu.txt","w").write(content)
    
def cal_var():
    def get_sample(steps):
        r=0
        for i in range(steps):
            r+=[-1.0,1.0][np.random.randint(0,2)]
        return r
    import os
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']=''
    from matplotlib import pyplot as plt
    sample_num=64
    # step_num=4



    for step_num in [1,5]:
        mean_reward_list=[]
        for i in range(10000):
            mean_reward=0
            for j in range(sample_num):
                mean_reward+=get_sample(step_num)
            mean_reward/=sample_num
            mean_reward_list.append(mean_reward)
        print(step_num)
        plt.hist(mean_reward_list,bins=50,alpha=0.5)
    plt.show()

def test_double_softplus():
    import paddle.nn.functional as F
    import paddle
    import os
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']=''
    from matplotlib import pyplot as plt
    x=paddle.linspace(-10,10,1000)
    y=F.softplus(2-F.softplus(x))

    plt.plot(x.numpy(),y.numpy())
    plt.show()
    plt.legend()

def test_normal_entropy():
    import paddle
    import paddle.optimizer as optim
    import paddle.nn as nn
    from paddle.distribution import Categorical,Normal
    from matplotlib import pyplot as plt
    std=paddle.to_tensor([0.2])
    mean=paddle.to_tensor([0.0])
    dist=Normal(mean,std)
    print(dist.entropy()-0.5*np.log(2*np.pi*np.exp(1)))
    print(paddle.log(std))
    data=dist.sample(shape=[10000]).numpy().reshape([-1])
    plt.hist(data,bins=50)
    plt.show()
def test_at():
    def func_a(fn):
        aaa=111
        fn()
    @func_a
    def func_b():
        print(aaa)
    func_b()
def test_reparam_normal():
    import paddle
    disshape=[32,100,64]
    normal_dis=paddle.distribution.Normal(paddle.zeros(disshape),paddle.ones(disshape))
    print(normal_dis.sample([10]).shape)
def test_bwalker_npy():
    import numpy as np
    data=np.load("C:/Users/Administrator.PC-20160914PWOW/Desktop/code/meta_hpc/all_data/bwalker/process_0/0.npy",allow_pickle=True).item()
    # print(data['reward'])
    for key in data.keys():
        print(key,data[key].shape)
    print(data["state"][:5])
def test_group_gru():
    import paddle.nn as nn
    import paddle
    import time
    input_dim=128
    _b=128
    _t=256
    hidden_size=128
    group_num=4

    data=paddle.ones([_b,_t,input_dim])

    "group gru"

    group_gru=[nn.GRU(input_dim,hidden_size) for n in range(group_num)]
    init_h=paddle.zeros([1,_b,hidden_size])


    for gru in group_gru:
        gru(data,init_h)
    t1=time.time()
    for i in range(500):
        y_list=[]
        for gru in group_gru:
            y_list.append(gru(data,init_h)[0])
        y=paddle.concat(y_list,axis=-1)
    print(y.shape)
    print("group cost",time.time()-t1)


    "gru"
    gru=nn.GRU(input_dim,hidden_size*group_num)
    init_h=paddle.zeros([1,_b,hidden_size*group_num])

    gru(data,init_h)
    t1=time.time()
    for i in range(500):
        y=gru(data,init_h)[0]
    print(y.shape)
    print("gru cost",time.time()-t1)
def test_categorial():
    from pfc.alg.ppo_gru_hybrid_dcu_alg import m_Categorical
    import paddle
    probs=paddle.rand([2,3,10])
    probs/=paddle.sum(probs,axis=-1,keepdim=True)
    # print(probs,probs.sum(axis=-1))
    dist=m_Categorical(probs)
    print(dist.sample([5]).shape)
def combine_npy():
    data_dir="E:/下载/breakout"
    for p_num in range(4):
        p_data_list=[]
        for npy_num in range(14):
            npy_path=f"{data_dir}/process_{p_num}/{npy_num}.npy"
            npy_data=np.load(npy_path,allow_pickle=True).item()
            # print(p_num,npy_num,[[key,npy_data[key].shape] for key in npy_data.keys()])
            p_data_list.append(npy_data)
        p_all_data={key:np.concatenate([data[key] for data in p_data_list],axis=0) for key in npy_data.keys()}
        print(p_num,npy_num,[[key,p_all_data[key].shape] for key in p_all_data.keys()])
        np.save(f"{data_dir}/process_{p_num}.npy",p_all_data)
def show_train_plot():
    from mlutils.logger import read_log,round_sf_np,moving_avg_list,norm_list
    from matplotlib import pyplot as plt
    import numpy as np
    import os    
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']=''
    for file in ["log/no_norm.txt","log/h_128.txt","log/h_256.txt","log/h_step.txt"]:
        log_data=read_log(file)
        print(log_data.keys())
        for key in ["pred_loss"]:
            data=log_data[key]
            data_moving=moving_avg_list(np.log10(data)[:5000],0.99)
            # reward_norm=norm_list(reward_moving)
            plt.plot(data_moving,label=file+"/"+key)
    plt.legend()
    plt.show()
def test_nptake():
    import numpy as np
    import time
    data=np.ones([50000,128])
    slice_len=128

    t1=time.time()
    for i in range(100000):
        rand_index=np.random.randint(0,len(data)-slice_len)
        x=np.take(data,list(range(rand_index,rand_index+slice_len)),0)
    print("cost",time.time()-t1)

    t1=time.time()
    for i in range(100000):
        rand_index=np.random.randint(0,len(data)-slice_len)
        x=data[rand_index:rand_index+slice_len]
    print("cost",time.time()-t1)


    # print(np.take(np.ones([1000,2]),[1,2,3],0).shape)
def test_gwy():
    import os
    import numpy as np
    img_dir="/data/E/myf/2023/2月公务员省考/7_排考场/照片/考生照片"
    save_dir="/data/E/myf/2023/2月公务员省考/7_排考场/照片/缺少的照片标准版"
    need_img_list=open("/data/E/myf/2023/2月公务员省考/7_排考场/照片/缺少照片的考生.txt","r").readlines()
    need_img_list=[l.replace("\n",".jpg") for l in need_img_list]
    print(need_img_list)
    for f in os.listdir(img_dir):
        # print(f)
        if f in need_img_list:
        # if 1:
            img_path=f"{img_dir}/{f}"
            img=cv2.imread(img_path)
            print(np.shape(img))
            # cv2.imshow("img",img)
            # cv2.waitKey()
            cv2.imwrite(f"{save_dir}/{f}",cv2.resize(img,(114,156)))
            
def test_paddle_gather():
    import paddle
    import numpy as np
    import time
    _b=256
    _t=512
    _d=128

    len_warmup=128
    len_max_history=512

    len_pred=16
    len_kl=256

    "gather"
    indexes=paddle.to_tensor(np.concatenate([list(range(i,i+len_pred)) for i in range(len_warmup,len_max_history-len_pred)]),'int32')
    print(indexes.shape)
    t1=time.time()
    for i in range(1000):
        a=paddle.randn([_b,_t,_d],'float32')
        result=paddle.gather(a,indexes,axis=1)
        # print(i)
        if i%100==0:
            print(i)
    print(result.shape)
    print("gather cost",time.time()-t1)

    "slice"
    starts=list(range(len_warmup,len_max_history-len_pred))[:3]
    ends=list(range(len_warmup+len_pred,len_max_history))[:3]
    axes=([1]*(len_max_history-len_pred-len_warmup))[:3]
    # print(slices)
    print(starts[:5])
    print(ends[:5])
    print(axes[:5])
    t1=time.time()
    for i in range(100):
        a=paddle.randn([_b,_t,_d],'float32')
        # print(a.shape)
        result=paddle.concat([a[:,_j:_j+len_pred] for _j in range(len_warmup,len_max_history-len_pred)],axis=1)
        # print(i)
        if i%100==0:
            print(i)
    print(result.shape)
    print("gather cost",time.time()-t1)
def test_var():
    import numpy as np
    class a:
        def __init__(self) -> None:
            pass
        def log(self,x):
            print(x.data)
    class b(a):
        def __init__(self) -> None:
            super().__init__()
            self.data=1
        def log(self):
            super().log(self)
    b().log()
def test_range():
    import paddle
    len_warmup=128
    len_max_history=512
    len_pred=128
    len_kl=256
    interval_pred=8
    interval_kl=8
    index_global_info=paddle.to_tensor(np.concatenate([[glo_i] for glo_i in range(len_warmup+len_pred,len_max_history,interval_pred)]),'int32')
    index_loc_info=paddle.to_tensor(np.concatenate([list(range(glo_i-len_pred,glo_i)) for glo_i in range(len_warmup+len_pred,len_max_history,interval_pred)]),'int32')
    index_pred_target=index_loc_info+1

    index_kl_start=paddle.to_tensor(list(range(0,len_max_history-len_kl)),'int32').reshape([-1,1])
    index_kl_start=paddle.tile(index_kl_start,[1,len_kl//interval_kl]).reshape([-1])
    # index_kl_final=paddle.to_tensor(list(range(len_kl,len_max_history)),'int32')
    index_kl_final=paddle.to_tensor([list(range(interval_kl,len_kl+interval_kl,interval_kl)) for _ in range(len_max_history-len_kl)],'int32').reshape([-1])
    
    print(index_kl_start.numpy().shape,index_kl_final.numpy().shape)
def testMiniGridGoToDoor():
    from env.minigrid_go_to_door import MiniGridGoToDoor
    env=MiniGridGoToDoor()
    env.reset()
    while 1:
        obs,reward,done,info=env.step(np.random.randint(0,4))
        cv2.imshow("outobs",cv2.resize(obs,(160,160),interpolation=cv2.INTER_AREA))
        env.render()
def test_while():
    t1=time.time()
    for i in range(int(1e16)):
        if i%1000000==0:
            print(i,time.time()-t1)
def test_queue_speed():
    import time
    from multiprocessing import Queue,Process
    def p1(queue):
        while 1:
            if queue.qsize()>0:
                t1=time.time()
                queue.get()
                print("get cost",time.time()-t1)
            else:
                time.sleep(0.01)
    def p2(queue):
        while 1:
            if queue.qsize()<50:
                t1=time.time()
                queue.put(np.zeros([2048,64,64,3]))
                print("put cost",time.time()-t1)
            else:
                time.sleep(0.01)

    queue=Queue(100)

    p_list=[Process(target=p1,args=(queue,)),Process(target=p2,args=(queue,))]
    [p.start() for p in p_list]
    [p.join() for p in p_list]
def test_pipe_speed():
    import time
    from multiprocessing import Pipe,Process
    
    def p1(pipe):
        while 1:
            time.sleep(2)
            # continue
            t1=time.time()
            pipe.recv()
            print("pipe get cost",time.time()-t1)
    def p2(pipe):
        while 1:
            t1=time.time()
            pipe.send(np.random.zeros([2048,64,64,3]))
            print("pipe send cost",time.time()-t1)

    pipe_send,pipe_recv=Pipe()

    p_list=[Process(target=p1,args=(pipe_send,)),Process(target=p2,args=(pipe_recv,))]
    [p.start() for p in p_list]
    [p.join() for p in p_list]
def test_thread_queue_speed():
    import time
    import threading
    from queue import Queue
    def p1(queue):
        while 1:
            if queue.qsize()>0:
                t1=time.time()
                queue.get()
                print("get cost",time.time()-t1)
            else:
                time.sleep(0.01)
    def p2(queue):
        while 1:
            t1=time.time()
            x=np.zeros([2048,64,64,3])
            print("generate cost",time.time()-t1)

            t1=time.time()
            queue.put(x)
            print("put cost",time.time()-t1)
            # x=

    queue=Queue(100)

    p_list=[threading.Thread(target=p1,args=(queue,)),threading.Thread(target=p2,args=(queue,))]
    [p.start() for p in p_list]
    # [p.join() for p in p_list]
    print("finish")
def test_sendr_eceiver_speed():
    import time
    from multiprocessing import Queue,Process
    from mlutils.multiprocess import Sender,Receiver
    def p1(queue_list):
        receiver=Receiver(queue_list)
        t_start=time.time()
        count=0
        while 1:
            
            print(receiver.qsize())
            t1=time.time()
            print("outer t1=",t1)
            receiver.get()
            t2=time.time()
            print("outer t2=",t2)
            count+=1
            print("p1 get cost",(time.time()-t_start)/count,t2-t1)
            # if receiver.qsize()>0:
            #     t1=time.time()
            #     receiver.get()
            #     print("get cost",time.time()-t1)
            # else:
            #     time.sleep(0.01)

    def p2(queue_list):
        sender=Sender(queue_list)
        while 1:
            if sender.qsize()<50:
                t1=time.time()
                sender.put(np.zeros([512,64,64,3]))
                # print("put cost",time.time()-t1)
            else:
                time.sleep(0.01)

    queue_list=[Queue(2) for i in range(10)]

    p_list=[Process(target=p1,args=(queue_list,)),Process(target=p2,args=(queue_list,))]
    [p.start() for p in p_list]
    [p.join() for p in p_list]
def test_array():
    from multiprocessing import Manager,Process
    from multiprocessing import shared_memory
    from ctypes import c_float
    from multiprocessing.sharedctypes import Array
    import numpy as np
    import time
    data=np.random.randint(0,255,[2,2,2])
    print(data)

    shm=shared_memory.SharedMemory(create=True,size=data.nbytes)

    print(shm.buf)
    print(shm.name)

    b=np.ndarray(data.shape,dtype=data.dtype,buffer=shm.buf)
    print(b)
    
    b[:]=data


    c=np.ndarray(data.shape,dtype=data.dtype,buffer=shm.buf)
    print("c=",c)
    
def test_manager():
    from multiprocessing import Manager,Process

    a=Manager().dict()
    print(a)
def test_getitem():
    class test_c():
        def __init__(self) -> None:
            
            pass
        def __getitem__(self,key):
            return 123
    c=test_c()
    print(c[1])
def test_recursion():
    def func(x):
        if isinstance(x,dict):
            return {key:func(x[key]) for key in x.keys()}
        else:
            return "finish"
    x={
        1:{
            3:"a"
        },
        2:"b"
    }
    print(func(x))
def test_shm():
    from mlutils.multiprocess import ProcessFilter,TaskFlow,Task,Cache
    from multiprocessing import Manager,shared_memory
    import numpy as np
    import copy
    import sys
    "共享内存调度信息"
    share_info=Manager().dict()
    "进程锁，用来确保share_info安全"
    lock=Manager().Lock()
    data={"env_data":{'img':np.ones([2048,3,64,64],dtype=np.float32),'action':np.zeros([100,10])}}
    cache=Cache(share_info,lock)
    cache.update(data)

    a=cache['env_data']['img']
    print(a[0,0,0,0])
    b=cache['env_data']['img']
    print(a[0,0,0,0])

def test_animal_env():
    from env.env_animalai import AnimalPlay
    import cv2
    env=AnimalPlay()
    env.reset()
    while 1:
        key=cv2.waitKey(20)
        if key!=-1:
            print(key)
        next_env_obs,reward,done,info=env.step(max(key-48,0))
        cv2.imshow('obs',next_env_obs)
def test_gauss_weight_mask():
    from cae.cae import Conv_Auto_Encoder
    from brain.hyperparam import brain_pfc_ec_hpc_hp
    from matplotlib import pyplot as plt
    wm=Conv_Auto_Encoder(brain_pfc_ec_hpc_hp.brain_hyperparam.cae_hyperparam).get_gauss_weight_mask()
    # print(wm)
    plt.imshow(wm.numpy())
    plt.show()

def test_vision():
    "草帽曲线边缘检测"
    import cv2
    import paddle
    from paddle import nn
    img=cv2.imread("/home/kylin/桌面/王慧林.jpg")
    img=paddle.to_tensor(img,'float32')
    img=paddle.transpose(img,[2,0,1])
    img=paddle.reshape(img,[1,*img.shape])
    
    
    conv_layer=nn.Conv2D(1,1,[5,5],1,'SAME')
    # weight=conv_layer.state_dict()['weight']
    weight=dict(conv_layer.named_parameters())['weight']
    new_weight=paddle.ones([1,1,5,5])/(-25.0)
    new_weight[0,0,2,2]=24/25
    weight.set_value(new_weight)

    out_img_list=[]
    for i in range(3):
        out_img=conv_layer(img[:,i:i+1])+110
        print(paddle.max(out_img),paddle.min(out_img))
        out_img=np.transpose(out_img.numpy().astype(np.uint8)[0],[1,2,0])
        out_img_list.append(out_img)
        cv2.imshow("out_img",out_img)
        cv2.waitKey()

    final_img=np.concatenate(out_img_list,axis=-1)
    print(np.shape(final_img))
    cv2.imshow("out_img",final_img)
    cv2.waitKey()
def test_activ_func():
    from paddle import nn
    import paddle
    from matplotlib import pyplot as plt
    func_dict={
        "Gelu":nn.GELU(),
        
        # "Elu":nn.ELU(),
        "Selu":nn.SELU(),
        # "Silu":nn.Silu(),
        "Swish":nn.Swish(),
    }
    x=paddle.to_tensor(np.arange(-5,5,0.01))
    for key in func_dict.keys():
        func=func_dict[key]
        y=func(x)
        plt.plot(x.numpy(),y.numpy(),label=key)
    plt.legend()
    plt.show()
def test_save_npy():
    import numpy as np
    a=np.zeros([204800,3,64,64],dtype=np.uint8)
    np.save("tes.npy",a)
def test_datarecorder():
    from mlutils.data import DataRecorder
    from mlutils.ml import DataFormat
    import numpy as np
    dr=DataRecorder(
        save_steps=2048*10,
        buffer_max_steps=2048*50,
        max_save_rpm_num=100,
        data_format_list=[DataFormat('img',[3,64,64],np.uint8),DataFormat('action',[5],np.float32)],
        data_dir='all_data/test',)
    for i in range(1000):
        print(i)
        
        data_dict={
            'img':np.random.randint(0,256,[2048,3,64,64]),
            'action':np.random.uniform(0,1,[2048,5])
        }
        dr.collect_dict_of_batch(data_dict)
        for j in range(20):
            dr.sample_batch_seq(4,512)
def test_animal():
    from env.env_animalai import Animal
    env=Animal()
    env.reset()
    while 1:
        obs=env.step(1)
        print(obs)
def subp():
    import subprocess
    p_args=['/home/aistudio/work/meta_hpc/env/animalai_env/env/AAI_v3.0.1_build_linux_090422.x86_64', '--mlagents-port', '5000', '--playerMode', '0', '--useCamera', '--resolution', '64', '--raysPerSide', '2', '--rayMaxDegrees', '60', '--decisionPeriod', '3', '-batchmode']
    p=subprocess.Popen(
                p_args,

                start_new_session=True,
                stdout=-3,
                stderr=-3,
            )
    print(p)
    print(p.poll())
def test_str():
    import re
    import time
    a="1"*10000000*8

    t1=time.time()
    re.findall(r'.{8}', a)
    print("re cost",time.time()-t1)

    t1=time.time()
    [a[8*i:8*(i+1)] for i in range(len(a)//8)]
    print("re cost",time.time()-t1)

def test_lmdbdata():
    from mlutils.data import LMDBRecoder
    from mlutils.ml import DataFormat
    import numpy as np
    import random
    import time
    lr=LMDBRecoder(data_format_list=[DataFormat('test',[1],np.float32,0)],lmdb_dir="./lmdbtest",map_size=int(2**32))
    
    for i in range(int(10**7)):
        lr._put_key_value(lr._transfer_num_to_key([i])[0],b'1')
        if i%1000000==0:
            print(i)
    lr._update()
    t1=time.time()
    min_index=lr.find_min_index()
    print(time.time()-t1)
    # print(lr.env.stat())
    # print(lr.env.info())
    # print(lr.env.max_key_size())
def test_logger():
    # import logging
    # logging.basicConfig(filename="C:/Users/admin/Desktop/code/meta_hpc/log/test/_2023_0707_221837.txt")
    from mlutils.logger import mllogger
    a=mllogger("C:/Users/admin/Desktop/code/meta_hpc/log/test")
    a.log_str("test")
    # import logging
    # import time
    # log_dir="C:/Users/admin/Desktop/code/meta_hpc/log/test"
    # log_fname=""
            


    # logging.basicConfig(filename=f"{log_dir}/{log_fname}_"+time.strftime("%Y_%m%d_%H%M%S",time.localtime(time.time()))+".txt",level=logging.INFO)

def test_compare_leveldb_lmdb():
    import time
    import lmdb
    import leveldb
    import numpy as np

    def transfer_data_to_value(array):
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
    def transfer_index_to_key(index):
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
    
    lmdb_db=lmdb.open("./lmdb",map_size=int(2**40)).begin(write=True)
    leveldb_db=leveldb.LevelDB("./leveldb")


    # data=np.zeros([1,1024,1024],np.float32)
    # data_value=transfer_data_to_value(data)
    # lmdb_db.put(b'0',data_value[0])
    # leveldb_db.Put(b'0',data_value[0])

    # t1=time.time()
    # lmdb_db.get(b'0')
    # t2=time.time()
    # leveldb_db.Get(b'0')
    # t3=time.time()

    # print(f"get big data cost lmdb:{t2-t1} leveldb{t3-t2}")

    # seq_index=transfer_index_to_key(list(range(0,int(1e7))))
    # seq_value=transfer_data_to_value(np.random.randint(0,255,[int(1e7),64,64,3],np.uint8))
    # print(len(seq_index,len(seq_value)))

    data_size=int(1e5)
    print(dir(leveldb_db))
    
    for i in range(int(data_size)):
        key=transfer_index_to_key(i)
        value=transfer_data_to_value(np.random.randint(0,255,[1,64,64,3],np.uint8))[0]
        # print(len(key),len(value))
        lmdb_db.put(key,value)
        leveldb_db.Put(key,value)
        if i%1000==0:
            print(i)
    leveldb_db.CompactRange()
    t1=time.time()
    for _,_ in lmdb_db.cursor():
        pass
    t2=time.time()
    for _,_ in leveldb_db.RangeIter():
        pass
    t3=time.time()
    
    print(f"lmdbcost {t2-t1} leveldbcost {t3-t2}")



def test_animal2():
    from env.env_animalai import Animal
    env=Animal(play=False,config_file="env/animalai_env/configs/competition/01-28-01.yaml")
    env.reset()

    steps=0
    start=time.time()
    while 1:
        obs=env.step(1)
        # print(obs)
        steps+=1
        if steps%2048==0:
            print(f"2048 frame cost {(time.time()-start)/steps*2048}")

def test_procgen():
    from procgen import ProcgenGym3Env
    # env = ProcgenGym3Env(num=1, env_name="coinrun")
    import gym
    import cv2
    import numpy as np
    import time
    # env = gym.make('procgen:procgen-coinrun-v0',start_level=0,num_levels=0,render=True)
    env = gym.make('procgen:procgen-jumper-v0',start_level=0,num_levels=0,render=True)
    # env = gym.make('procgen:procgen-ninja-v0',start_level=0,num_levels=0,render=True)
    # env = gym.make('procgen:procgen-caveflyer-v0',start_level=0,num_levels=0,render=True)
    obs = env.reset()
    print(env.action_space)

    steps=0
    start=time.time()

    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        steps+=1
        if steps%2048==0:
            print(f"2048 frame cost {(time.time()-start)/steps*2048}")
        # print(np.shape(obs))
        cv2.imshow("obs",cv2.resize(obs,None,fx=5,fy=5,interpolation=cv2.INTER_AREA)[:,:,::-1])
        cv2.waitKey(1)
        env.render()
        
        if done:
            pass
def test_rand_slice():
    import random
    a=list(range(0,10000+1,100))
    b=[]
    for i in range(len(a)-1):
        b.append({"start":a[i],"end":a[i+1]})
    
    random.shuffle(b)
    print(b)
    while 1:
        b.pop(0)
        print(len(b))

def test_pid():
    from matplotlib import pyplot as plt
    from scipy.signal import savgol_filter
    from mlutils.ml import EstimateMovingAVerage
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
    class pidcontroller():
        def __init__(self,pid_integral_gamma,pid_delta,pid_integral,pid_error):
            self.x=1.0

            self.pre_e=0

            self.integral_e1=0
            self.pid_integral_gamma1=0.999999
            self.pid_integral1=2.0

            self.pid_delta=0.0
            
            self.pid_error=0.00001

            
        def _cal_update_pid(self,e):
            self.integral_e1=(self.pid_integral_gamma1*self.integral_e1+(1-self.pid_integral_gamma1)*e)*0.995

            delta_e=e-self.pre_e
            integral_e=self.integral_e1*self.pid_integral1

            self.x=self.x+self.pid_error*e+self.pid_delta*delta_e+integral_e

            self.pre_e=e
        def update(self,y):
            e=y-self.x
            self._cal_update_pid(e)
            return self.x
    class meanfilter():
        def __init__(self,max_size):
            pass
            self.x=1.0
            self.max_size=max_size
            self.y_list=[1.0]*self.max_size
        def update(self,y):
            self.y_list.append(y)
            if len(self.y_list)>self.max_size:
                out_y=self.y_list.pop(0)

            self.x+=(y-out_y)/self.max_size

            return self.x 
    class LMA():
        pass
        # 线性移动平均估计，通过计算两种gamma值的均值估计，推测出当前的均值
    class parafilter():
        def __init__(self,gamma):
            self.f1=meanfilter(10000)
            self.f2=meanfilter(20000)
            self.f3=meanfilter(30000)
        def update(self,y):
            m1=self.f1.update(10000)
    class gammafilter():
        def __init__(self,gamma) -> None:
            self.gamma=gamma
            self.steps=0
            self.x=0.0
        def update(self,y):
            self.steps+=1
            self.x=self.x*self.gamma+(1-self.gamma)*y
            return self.x/(1-self.gamma**self.steps)
    pid1=pidcontroller(pid_integral_gamma=0.995,pid_delta=0.0,pid_integral=0.01,pid_error=0.0001)
    # pid2=pidcontroller(pid_integral_gamma=0.995,pid_delta=0.0,pid_integral=0.01,pid_error=0.0001)
    # pid3=pidcontroller(pid_integral_gamma=0.995,pid_delta=0.0,pid_integral=0.01,pid_error=0.0001)
    # pid4=pidcontroller(pid_integral_gamma=0.995,pid_delta=0.0,pid_integral=0.01,pid_error=0.0001)
    w1=40000
    w2=20000
    w3=10000
    f1=meanfilter(max_size=40000)
    f2=meanfilter(max_size=20000)    
    f3=meanfilter(max_size=10000)
    fmat=[
        [1./3*w1**2,-0.5*w1,1.],
        [1./3*w2**2,-0.5*w2,1.],
        [1./3*w3**2,-0.5*w3,1.],
    ]
    inv_fmat=np.linalg.inv(fmat)

    gamma1=0.99995
    gamma2=0.9999
    gamma3=0.9998
    c1=1/np.log(gamma1)
    c2=1/np.log(gamma2)
    c3=1/np.log(gamma3)
    g1=gammafilter(gamma=gamma1)
    g2=gammafilter(gamma=gamma2)
    g3=gammafilter(gamma=gamma3)
    mat=[
        [(c1)**2,c1,1.],
        [(c2)**2,c2,1.],
        [(c3)**2,c3,1.],
    ]
    inv_mat=np.linalg.inv(mat)
    print(inv_mat)
    info_dict=read_log("log/_2023_0727_110400.txt")
    print(info_dict.keys())
    y_list=info_dict['klpred_target_original']
    x_list=[]

    ema=EstimateMovingAVerage()
    for y in y_list:
        # m1=f1.update(y)
        # m2=f2.update(y)
        # m3=f3.update(y)

        m1=g1.update(y)
        m2=g2.update(y)
        m3=g3.update(y)
        # m1=f1.update(y)
        # m2=f2.update(y)
        # m3=f3.update(y)
        # print(np.matmul(inv_mat,[[m1],[m2],[m3]]))
        # input()
        x_list.append(ema.update(y))
        # x_list.append(pid1.update(y))
        # x_list.append(2*m3-m2)
        # x_list.append(m2+(m2-m1)*(np.log(gamma1)/(np.log(gamma2)-np.log(gamma1))))
    z_list=savgol_filter(y_list,window_length=20000,polyorder=2)
    plt.plot(info_dict['klpred_target_original'],label='klpred_target_original')
    plt.plot(info_dict['klpred_moving_avg'],label='klpred_moving_avg')
    plt.plot(z_list,label='savgol_filter')
    plt.plot(x_list,label='filter')
    plt.ylim(0,np.max(x_list)*1.3)
    plt.legend()
    plt.show()
def cal_mean():
    gamma=0.999
    def mean_func(x,gamma):
        x=np.array(x)
        return np.sum(x*gamma**-x)/np.sum(gamma**-x)
    x=list(range(-10000000,1))
    print(len(x))
    print(mean_func(x,gamma))
    print(1/np.log(gamma))
    
if __name__=="__main__":
    # cal_mean()
    test_pid()
    # test_rand_slice()
    # test_procgen()
    # test_animal2()
    # test_compare_leveldb_lmdb()
    # test_logger()
    # test_lmdbdata()
    # test_str()
    # subp()
    # test_animal()
    # test_datarecorder()
    # test_activ_func()
    # test_vision()
    # test_gauss_weight_mask()
    # test_animal_env()
    # test_shm()
    # test_recursion()
    # test_getitem()
    # test_manager()
    # test_array()
    # test_sendr_eceiver_speed()
    # test_thread_queue_speed()
    # test_pipe_speed()
    # test_queue_speed()
    # test_while()
    # testMiniGridGoToDoor()
    # test_range()
    # test_var()
    # test_paddle_gather()
    # test_gwy()
    # test_nptake()
    # show_train_plot()
    # combine_npy()
    # test_categorial()
    # test_group_gru()
    # test_bwalker_npy()
    # test_reparam_normal()
    # test_vae()
    # test_draw_cv()
    # test_draw_gl()
    # test_cv()
    # test_time()
    # compare_cv_and_texture()
    # test_cmd()
    # test_paddle_sum()
    # test_show_plot()
    # test_place_cell()
    # test_grid_cell()
    # logger()
    # algorhtmReq()
    # test_mail()
    # test_gauss1()
    # test_gauss2()
    # test_gauss3()
    # test_randn()
    # test_ce()
    # test_sae_param()
    # test_sparse_loss()
    # test_info_entropy()
    # test_expand()
    # test_tril()
    # test_gauss_kl()
    # train_UAE()
    # test_uniform_area()
    # test_gnd()
    # test_adam()
    # test_minigrid_fourroom()
    # test_global()
    # test_gru_time()
    # test_cd()
    # test_nan()
    # test_log_prob()
    # test_cal_minigrid_inner_reward()
    # test_plt_hist_to_plot()
    # show_cal_npy()
    # test_show_reward()
    # test_perfect_env()
    # test_Categorical()
    # test_one_hot()
    # test_gauss_sample_ratio()
    # test_beta()
    # test_minigridlife()
    # test_breakout()
    # test_class_contrib()
    # test_png()
    # test_elbo_fit()
    # test_fast_mem_seq()
    # test_state_dict()
    # test_class()
    # test_change_color()
    # test_gather_nd()
    # test_fisheye()
    # test_fisheye_img()
    # test_np_slice()
    # test_entropy()
    # test_derk()
    # test_class_attribute()
    # get_question_from_html()
    # get_web_question()
    # cal_var()
    # test_double_softplus()
    # test_normal_entropy()
    # test_at()



