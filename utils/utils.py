# encoding: utf-8
from brian2 import *
from PIL import Image
import numpy as np
from scipy import misc
import matplotlib.pyplot as pyplot
import time
import math
import matlab.engine
import os
import scipy.io as sio

class Timer:
    def __init__(self, msg='Time elapsed'):
        self.msg = msg
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.end = time.time()
        duration = self.end - self.start
        print("f{self.msg}: {duration:.2f}s")

def read_spike_brian2_format(height, width, PATH, input_time, save_data, file_name):
    '''Read spike indice, spike times, and spike interval from dat file to fit the Brian2 format. Slow. '''
    w = width
    h = height
    c = np.fromfile(PATH, dtype='uint8', count=-1)
    indices = np.zeros(int(input_time * w * h / 25/us))
    times = np.zeros(int(input_time * w * h / 25/us))
    flag = 0
    start = time.clock()
    for t in range(int(input_time/25/us)):
        length = w * h / 8
        for loc in range(w * h):
            comparator = 2**(loc % 8)
            r = c[int(math.floor(loc/8) + length * t)] & comparator
            if r == comparator:
                indices[flag] = loc
                times[flag] = int(t * 25)
                flag += 1
        if t%100 == 0:
            end = time.clock()
            time1 = end - start
            print("Moment: %s x25us, Runtime is : %f s" % (t, time1))
    indices = indices[0:flag]
    times = times[0:flag]
    timematrix = [[] for i in range(w*h)]
    interval = [[] for i in range(w*h)]
    for idx in range(len(times)):
        a = int(indices[idx])
        b = int(times[idx])
        timematrix[a].append(b)
    for neuroidx in range(w*h):
        spiketimelast = 0
        for spiketime in timematrix[neuroidx]:
            itv = int((spiketime - spiketimelast)/25)
            spiketimelast = spiketime
            interval[neuroidx].append(itv)
        itv = int((input_time/us - spiketimelast)/25)
        interval[neuroidx].append(itv)
    if save_data:
        np.save('./indices-%s-%sus.npy'%(file_name,int(input_time/us)),indices)
        np.save('./times-%s-%sus.npy'%(file_name,int(input_time/us)),times)
        np.save('./interval-%s-%sus.npy'%(file_name,int(input_time/us)),interval)
    return indices, times, interval

def read_spike_raw(path, width, height):
    '''
    Load bit-compact raw spike data into an ndarray of shape
        (`frame number`, `height`, `width`). Fast.
    '''
    with open(path, 'rb') as f:
        fbytes = f.read()
    fnum = (len(fbytes) * 8) // (width * height)  # number of frames
    frames = np.frombuffer(fbytes, dtype=np.uint8)
    frames = np.array([frames & (1 << i) for i in range(8)])
    frames = frames.astype(np.bool).astype(np.uint8)
    frames = frames.transpose(1, 0).reshape(fnum, height, width)
    frames = np.flip(frames, 1)
    return frames

def spatial_filter(inputimg,i,j):
    inputimg_ij = inputimg[i:-1:2, j:-1:2] 
    d1 = (inputimg[i - 1:-2:2, j:-1:2] + inputimg[i + 1::2, j:-1:2])/2.0  - inputimg[i:-1:2, j:-1:2]
    d2 = (inputimg[i:-1:2, j - 1:-2:2] + inputimg[i:-1:2, j + 1::2])/2.0  - inputimg[i:-1:2, j:-1:2]
    d3 = (inputimg[i - 1:-2:2, j - 1:-2:2] + inputimg[i + 1::2, j + 1::2])/2.0 - inputimg[i:-1:2, j:-1:2]
    d4 = (inputimg[i - 1:-2:2, j + 1::2] + inputimg[i + 1::2, j - 1:-2:2])/2.0  - inputimg[i:-1:2, j:-1:2]
    d5 = (inputimg[i - 1:-2:2, j:-1:2] + inputimg[i:-1:2, j - 1:-2:2] -inputimg[i - 1:-2:2, j - 1:-2:2]) - inputimg[i:-1:2, j:-1:2]
    d6 = (inputimg[i - 1:-2:2, j:-1:2] + inputimg[i:-1:2, j + 1::2] - inputimg[i - 1:-2:2,j + 1::2]) - inputimg[i:-1:2, j:-1:2]
    d7 = (inputimg[i:-1:2, j- 1:-2:2] + inputimg[i + 1::2, j:-1:2] - inputimg[i + 1::2, j - 1:-2:2]) - inputimg[i:-1:2, j:-1:2]
    d8 = (inputimg[i:-1:2, j + 1::2] + inputimg[i + 1::2, j:-1:2] - inputimg[i + 1::2, j + 1::2]) - inputimg[i:-1:2, j:-1:2]

    d = d1 * (np.abs(d1) <= np.abs(d2)) + d2 * (np.abs(d2) < np.abs(d1))
    d = d * (np.abs(d) <= np.abs(d3)) + d3 * (np.abs(d3) < np.abs(d))
    d = d * (np.abs(d) <= np.abs(d4)) + d4 * (np.abs(d4) < np.abs(d))
    d = d * (np.abs(d) <= np.abs(d5)) + d5 * (np.abs(d5) < np.abs(d))
    d = d * (np.abs(d) <= np.abs(d6)) + d6 * (np.abs(d6) < np.abs(d))
    d = d * (np.abs(d) <= np.abs(d7)) + d7 * (np.abs(d7) < np.abs(d))
    d = d * (np.abs(d) <= np.abs(d8)) + d8 * (np.abs(d8) < np.abs(d))

    inputimg_ij[...] +=d
    return inputimg_ij

def save_fig(img_mat, folder, prefix, file_name, t, recon_mode):
    isExists=os.path.exists(folder+'/'+prefix)
    if not isExists:
        os.makedirs(folder+'/'+prefix)

    maxvalue = np.max(img_mat)
    img_mat = img_mat/maxvalue
    
    total_iter = 3
    filteredimg = 255*img_mat
    if recon_mode == 'camera_moving':
        filteredimg = np.power(filteredimg,1/3.5)
    for iter_num in range(total_iter):
        spatial_filter(filteredimg, 1, 1)
        spatial_filter(filteredimg, 2, 2)
        spatial_filter(filteredimg, 1, 2)
        spatial_filter(filteredimg, 2, 1)
    new_im = Image.fromarray(filteredimg)
    misc.imsave(folder+'/'+prefix+'-%s-%s.bmp' % (file_name,t), new_im.transpose(Image.FLIP_TOP_BOTTOM))




