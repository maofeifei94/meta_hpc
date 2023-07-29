from calendar import TUESDAY
from mlutils.common import *
from mlutils.env import Rect
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



def crop_img(img,crop_rect:Rect):
    """
    从图像中裁剪出一块区域（miny,minx,maxy,maxx）。
    如果次区域超过图像边界，则填充0。
    """
    
    full_img_box,sub_img_box=img_cross_area(*np.shape(img)[:2],crop_rect)
    out_img=np.zeros([crop_rect.height,crop_rect.width,img.shape[-1]],dtype=img.dtype)
    out_img[sub_img_box.miny:sub_img_box.maxy+1,sub_img_box.minx:sub_img_box.maxx+1]=img[full_img_box.miny:full_img_box.maxy+1,full_img_box.minx:full_img_box.maxx+1]
    return out_img

def img_cross_area(full_x,full_y,cross_rect:Rect):
    """
    full_x,full_y是大图的尺寸
    minx,miny,maxx,maxy是小图的
    """
    full_img_minx=max(0,cross_rect.minx)
    full_img_miny=max(0,cross_rect.miny)
    full_img_maxx=min(cross_rect.maxx+1,full_x)
    full_img_maxy=min(cross_rect.maxy+1,full_y)

    sub_img_minx=full_img_minx-cross_rect.minx
    sub_img_miny=full_img_miny-cross_rect.miny
    sub_img_maxx=full_img_maxx-full_img_minx+sub_img_minx
    sub_img_maxy=full_img_maxy-full_img_miny+sub_img_miny

    full_img_rect=Rect(full_img_miny,full_img_minx,full_img_maxy-1,full_img_maxx-1)
    sub_img_rect=Rect(sub_img_miny,sub_img_minx,sub_img_maxy-1,sub_img_maxx-1)
    return full_img_rect,sub_img_rect