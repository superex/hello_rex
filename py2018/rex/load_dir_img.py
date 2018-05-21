from os import listdir
from os.path import isdir, join
from scipy.misc import imresize
import numpy as np
import imageio
import cv2

import PIL
from PIL import ImageEnhance



def image_augmentation4(x_train, y_train, channel=3):
    if len(x_train)==0:
        return x_train,y_train
        
    tmpx = []
    tmpy = []
    for x,y in zip(x_train,y_train):
                                
        x2=np.flip(x,axis=0)
        x3=np.flip(x,axis=1)
        x4=np.flip(x2,axis=1)        
        
        tmpx.append(x)
        if channel==1:
            tmpx.append(enBrightness(x2,1.3))
            tmpx.append(enContrast(x3,1.3))   
            tmpx.append(enBrightness(x4,0.7))
        else:
            tmpx.append(enBrightness(x2,1.3))
            tmpx.append(enContrast(x3,1.3))   
            tmpx.append(enBrightness(x4,0.7))
                                                                       
        tmpy.append(y)
        tmpy.append(y)
        tmpy.append(y)
        tmpy.append(y)
                                            
    tmpx=np.asarray(tmpx)    
    tmpy=np.asarray(tmpy).flatten()
    
    return tmpx,tmpy

def load_as_xy_resize(pathx,shrink_ratio=1,channel=3,center_hw=None,target_hw=None):
    onlydirs = [f for f in listdir(pathx) if isdir(join(pathx, f))]
    lst_tmpx = []
    lst_tmpy = []
    
    for d in onlydirs:
        dirn = pathx +"/" + d
        print(dirn)
        lst=load_dir_resize(dirn,shrink_ratio=shrink_ratio,channel=channel,center_hw=center_hw,target_hw=target_hw)
        lst_tmpx.extend(lst)
        
        yclass = d
        if yclass.find("_")!=-1: yclass = d[0:d.find("_")]
        
        yclass = yclass.upper()
        
        for _ in range(len(lst)):
            lst_tmpy.append(yclass)
                            
    classes = sorted(list(set(lst_tmpy)))
    y = []
    for i in lst_tmpy:
        y.append(classes.index(i))
    
    x = np.asanyarray(lst_tmpx)
    y = np.asanyarray(y)
                                
    return x,y,classes                    
    

def load_dirs_resize(pathx,h=240,w=320,channel=3):
    onlydirs = [f for f in listdir(pathx) if isdir(join(pathx, f))]
    tmplist = []
    
    lst=load_dir_resize(pathx,target_hw=(h,w),channel=channel)
    tmplist.extend(lst)
    
    for d in onlydirs:
        dirn = pathx +"/" + d
        print(dirn)
        lst=load_dir_resize(dirn,target_hw=(h,w),channel=channel)
        tmplist.extend(lst)
                         
    return tmplist

def load_file_resize(ffn,shrink_ratio=1,channel=3,center_hw=None,target_hw=None):
    if channel==1:
        #img = misc.imread(ffn,flatten=True)
        img = imageio.imread(ffn)
        if img.ndim==3:
            img = rgb2gray(img)                        
    else:
        #img = misc.imread(ffn)
        img = imageio.imread(ffn)
    
    if center_hw is not None:
        img = get_center(img,csh=center_hw[0],csw=center_hw[1])
    
    if target_hw is not None:
        img = get_target_size(img,h=target_hw[0],w=target_hw[1])            
    
    if shrink_ratio!=1:
        csh,csw = img.shape[0:2]            
        sample_size_w = int(csw / shrink_ratio)
        sample_size_h = int(csh / shrink_ratio)                    
        img=imresize(img,size=(sample_size_h,sample_size_w),interp='bicubic')
    
    if channel==1:
        img=cv2.GaussianBlur(img,(0,0),3)
        img=cv2.addWeighted(img, 1.5, img, -0.5, 0)
    
    return img    

def load_dir_resize(pathx,shrink_ratio=1,channel=3,center_hw=None,target_hw=None):
    tmplist = []
    for fn in listdir(pathx):
        
        fnl = fn.lower()
        if not (fnl.endswith(".jpg") or fnl.endswith(".png") or fnl.endswith(".bmp")): continue
                                    
        ffn = pathx+"/"+fn
        if isdir(ffn): continue
                
        try:
            img = load_file_resize(ffn,shrink_ratio=shrink_ratio,channel=channel,center_hw=center_hw,target_hw=target_hw)                    
            tmplist.append(img)
        except Exception as e:            
            print(str(e))
                                        
    return tmplist    

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
def get_target_size(img, h=-1, w=-1):
    if (h==-1): return img
    if (w==-1): return img
    img=imresize(img,size=(h,w),interp='bicubic')
    return img
            
def get_center(img, csh = -1, csw = -1):
    if csh==-1: return img
    if csw==-1: return img
            
    if img.shape[0] < csh: return img
    if img.shape[1] < csw: return img
    
    if img.shape[0:2]==(csh,csw): return img
                        
    w = int(img.shape[1])
    h = int(img.shape[0])    
    x = int((w-csw)/2)
    y = int((h-csh)/2)    
    return img[y:y+csh,x:x+csw]
    
def load_from_dirs(pathx,ratio=1):
    onlydirs = [f for f in listdir(pathx) if isdir(join(pathx, f))]
    tmplist = []
    
    lst=load_from_dir(pathx)
    tmplist.extend(lst)
    
    for d in onlydirs:
        dirn = pathx +"/" + d
        print(dirn)
        lst=load_from_dir(dirn,ratio)
        tmplist.extend(lst)   
                 
    return np.asarray(tmplist)    

def load_from_dir(pathx,ratio=1):
    
    tmplist = []
    for fn in listdir(pathx):
                        
        ffn = pathx+"/"+fn
        if isdir(ffn): continue
        
        try:
            #img=misc.imread(ffn)
            img = imageio.imread(ffn)
        except Exception as e:            
            continue        
        
        if ratio!=1:
            img=enLarge(img,ratio)            
        
        tmplist.append(img)
        #print(fn+"="+str(img.shape))
                        
    #print("size="+str(len(tmplist)))
    return tmplist    
    
def enBrightness(img, ratio=1.3):
    im=PIL.Image.fromarray(img)
    im=ImageEnhance.Brightness(im).enhance(ratio)    
    return np.array(im)    

def enContrast(img, ratio=1.3):
    im=PIL.Image.fromarray(img)
    im=ImageEnhance.Contrast(im).enhance(ratio)    
    return np.array(im)
    
def enLarge(img, ratio=1.2):
    h,w=img.shape[0:2]    
    img2=imresize(img,size=(int(h*ratio),int(w*ratio)),interp='bicubic')
    return img2    