# encoding: utf-8
from brian2 import *
from PIL import Image
import numpy as np
from scipy import misc
from model.model import UnsupervisedSNM
from utils.utils import *
import matplotlib.pyplot as pyplot
import time
import math
import matlab.engine
import os
import scipy.io as sio
import argparse

def main(args):
    args.tau = args.tau*us
    args.vth = args.vth*mV
    args.tauth = args.tauth*us
    args.tauth_recon = args.tauth_recon*us
    args.t_refractory = args.t_refractory*us
    args.taum = args.taum*us
    args.Ee = args.Ee*mV
    args.vr = args.vr*mV
    args.El = args.El*mV
    args.taue = args.taue*us
    args.taupre = args.taupre*us
    args.taupost = args.taupost*us
    args.input_time = args.input_time*us
    args.run_time = args.run_time*args.scale_ts*us

    file_name = 'temp'
    if args.load_data:
        indices=np.load('./indices-%s-%sus.npy'%(file_name,int(args.input_time/us)))
        times=np.load('./times-%s-%sus.npy'%(file_name,int(args.input_time/us)))
        interval = np.load('./interval-%s-%sus.npy'%(file_name,int(args.input_time/us)),allow_pickle=True)
    else:
        indices, times, interval = read_spike_brian2_format(args.height, args.width, args.input_file, args.input_time, args.save_data, file_name)

    model = UnsupervisedSNM(args, interval, indices, times)

    if args.recon_time == 0:
        for i in range(100, args.run_time/25):
                img = model.recon_visual_image(i)
                save_fig(img, args.save_path, file_name, args.recon_mode, i)
    else:
        img = model.recon_visual_image(args.recon_time)
        save_fig(img, args.save_path, 'temp', args.recon_mode, args.recon_time, args.recon_mode)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description='Inference for Spiking Neuron Model')
    parser.add_argument('-i', '--input_file',  type=str,
                        help='path to dat files')
    parser.add_argument('-m', '--recon_mode', type=str, default='camera_moving',
                        help='reconstruction mode: camera_moving, camera_fix_1, or camera_fix_2.\
                        `camera_moving` mode is suitable for scenes, `camera_fix_1, camera_fix_2` are more suitbale for fixed camera scenes.')
    parser.add_argument('-t', '--recon_time', type=int, default=200,
                        help='reconstruction time t, will reconstrut every moment if t set t as 0')
    parser.add_argument('-o', '--save_path',  type=str, default='./',
                        help='path to save results')      
    parser.add_argument('--width', default=400, type=int)
    parser.add_argument('--height', default=250, type=int)
    parser.add_argument('--save_data', default=False, type=bool, help='Store intermediate variables')
    parser.add_argument('--load_data', default=False, type=bool, help='Load intermediate variables')
    parser.add_argument('--scale_ts', default=10, type=int)
    parser.add_argument('--stable_ts', default=0, type=int)
    parser.add_argument('--input_time', default=25*300, type=int)
    parser.add_argument('--run_time', default=25*300, type=int)

    parser.add_argument('--tau', default=10000, type=float, help='us')
    parser.add_argument('--vth', default=1, type=float, help='mV')
    parser.add_argument('--tauth', default=20000, type=float, help='us')
    parser.add_argument('--tauth_recon', default=20000, type=float, help='us')
    parser.add_argument('--t_refractory', default=10, type=float, help='us')
    parser.add_argument('--taum', default=10000, type=float, help='us')
    parser.add_argument('--Ee', default=0, type=float, help='mV')
    parser.add_argument('--vr', default=-600, type=float, help='mV')
    parser.add_argument('--El', default=0, type=float, help='mV')
    parser.add_argument('--taue', default=50000, type=float, help='us')
    parser.add_argument('--gmax', default=0.2, type=float, help='-')
    parser.add_argument('--dApre', default=.01, type=float, help='-')
    parser.add_argument('--taupre', default=10000, type=float, help='us')
    parser.add_argument('--taupost', default=10000, type=float, help='us')

    args = parser.parse_args()
    main(args)
