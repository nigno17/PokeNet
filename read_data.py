from __future__ import print_function, division

import numpy as np
import os
from skimage import img_as_float
import cv2
from os.path import expanduser
from scipy.ndimage import gaussian_filter


home = expanduser("~")


base = 'BW_normalize/'
img_h = 128
img_w = 128
prng = np.random.RandomState(0)

act_dict = {'hook_draw':0, 'hook_push':1, 'hook_tap_from_right':2, 'hook_tap_from_left':3,
            'rake_draw':4, 'rake_push':5, 'rake_tap_from_right':6, 'rake_tap_from_left':7, 
            'stick_draw':8, 'stick_push':9, 'stick_tap_from_right':10, 'stick_tap_from_left':11}


def get_imgs_acts_dict(write_train = True):
  bf ={'imgs_array_before':list(), 'imgs_array_after':list()}
  bfn =dict()
  acts_list = list()
  cameras = ['left_camera', 'right_camera']
  tools = [t for t in os.listdir(base) if os.path.isdir(os.path.join(base,t))]
  for tool in tools:
    tool_path = os.path.join(base,tool)
    objs = os.listdir(tool_path)
    for obj in objs:
      obj_path = os.path.join(tool_path,obj)
      acts = os.listdir(obj_path)
      for act in acts:
        act_path = os.path.join(obj_path,act)
        for camera in cameras:
          imgs_path = os.path.join(act_path, camera)
          imgs = os.listdir(imgs_path)
          for img_number in range(int(len(imgs)/2)):
            acts_list.append(act_dict[tool+'_'+act])
            for bof in bf.keys():
              if bof.find('before')!=-1:
                img_name = 'before_' + str(img_number) + '_FG.jpg'
              else:
                img_name = 'after_' + str(img_number) + '_FG.jpg'
              img_path = os.path.join(imgs_path,img_name)
              img_raw = cv2.imread(img_path,0)
              img_resize = cv2.resize(img_raw,(img_h,img_w))
              img = gaussian_filter(img_resize,sigma=0.5)
              img_1 = img_as_float(img).astype(np.float32)
              bf[bof].append(img_1)
              
              
  def get_train(imgs_list):
    prng = np.random.RandomState(0)
    imgs_array = np.stack(imgs_list)
    imgs_array = imgs_array.reshape([-1,img_h*img_w])
    train_size = np.int(imgs_array.shape[0]*0.8)
    idxs = range(imgs_array.shape[0])
    prng.shuffle(idxs)
    idxs_train = idxs[:train_size] if write_train else idxs[train_size:]
    return imgs_array[idxs_train,:], idxs_train
    
  for bof in bf.keys():
    bfn[bof], idxs_train = get_train(bf[bof])
    
  acts_array = np.stack(acts_list)
  acts_array_train = acts_array[idxs_train]
  return bfn, acts_array_train
  
  
if __name__ == '__main__':
	imgs_dict, acts_array_train = get_imgs_acts_dict
