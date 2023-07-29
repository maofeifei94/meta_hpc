# import IPython.display as display
def test():
    import sys
    import numpy as np
    import random
    import os
    import time
    import cv2
    from matplotlib import pyplot as plt
    from mlutils.logger import get_date_str
    from env.env_animalai import Animal
    log_dir=f"log/{get_date_str()}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    env=Animal()
    env.reset()
    for i in range(1000000000):
        t1=time.time()
        obs,reward,done,info=env.step(1)
        # cv2.imwrite("img.png",obs)
        # cv2.imwrite()
        # print(time.time()-t1)
        # print(obs.shape,np.max(obs),np.min(obs))
        cv2.imwrite(f"{log_dir}/{i}.png",obs)
        # cv2.imshow("obs",obs)
        # cv2.waitKey(10)
        # display.clear_output(wait=True)
        # plt.imshow(obs)
        # plt.show()
        # time.sleep(100)
if __name__=="__main__":
    test()