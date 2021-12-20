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

# three layers: input_layer, input_s1 layer and excitatory_layer, 
# one to one connection between input_s1 layer and s1_layer, 
# and one to N spatial connection between s1_layer and excitatory_layer
# add dynamic refining

class UnsupervisedSNM:
    """This is the Spiking Neuron Model that was presented in the paper:
       "Retina-like Visual Image Reconstruction via Spiking Neural Model", CVPR'20 """

    def __init__(self, config, interval, indices, times):
        #super(UnsupervisedSNM, self).__init__()
        self.recon_mode = config.recon_mode
        self.camera_moving = False
        self.camera_fix_1 = False
        self.camera_fix_2 = False
        if self.recon_mode == 'camera_moving':
            self.camera_moving = True
        elif self.recon_mode == 'camera_fix_1':
            self.camera_fix_1 = True
        elif self.recon_mode == 'camera_fix_2':
            self.camera_fix_2 = True
        else:
            self.camera_moving = True
        
        self.height = config.height
        self.width = config.width
        self.scale_ts = config.scale_ts
        self.stable_ts = config.stable_ts
        self.input_time = config.input_time
        self.run_time = config.run_time
        self.save_data = config.save_data
        self.load_data = config.load_data
        '''==========Neuron parameters========='''
        tau = config.tau
        vth = config.vth
        tauth = config.tauth
        tauth_recon = config.tauth_recon
        t_refractory = config.t_refractory
        taum = config.taum
        Ee = config.Ee
        vr = config.vr
        El = config.El
        taue = config.taue
        gmax = config.gmax
        dApre = config.dApre
        taupre = config.taupre
        taupost = config.taupost

        dApost = -dApre * taupre / taupost * 1.05
        dApost *= gmax
        dApre *= gmax
        self.last_change = [0 for i in range(self.width*self.height)]
        self.n_input = self.height*self.width        
        self.interval = interval
        file_name = 'temp'

        if self.camera_moving != True:
            self.motioncut = self.build_motion_excitation_layer(interval, file_name)
        ng_input = SpikeGeneratorGroup(self.n_input, indices, times * self.scale_ts * us)
        ng_e1, sm_e1 = self.build_spike_refining_layer(self.n_input, False, tau, vth, tauth)
        conn1 = self.build_ordinary_connection(ng_input, ng_e1)
        conn1.connect(j='i')
        ng_e2, vm_e2, vth_e2 = self.build_visual_reconstruction_layer(self.n_input, False, Ee, vr, El, taum, vth, tauth_recon, taue)
        conn2 = self.build_stdp_connection(ng_e1, ng_e2, True, taupre, taupost)
        conn2 = self.build_spatial_connection(conn2, ng_e1, ng_e2, 15)
        run(self.run_time, report='text')
        self.vth_e2 = vth_e2

    def build_motion_excitation_layer(self, interval, file_name):
        if self.load_data:
            output = np.load('./output-%s-%sus.npy'%(file_name,int(self.input_time/us)))
        else:
            sio.savemat('interval-%s-%sus.mat'%(file_name,int(self.input_time/us)), {'itv': interval})
            eng = matlab.engine.start_matlab()
            output = eng.motioncut(file_name,'%d'%(self.input_time/us),float(self.input_time/25/us), self.height, self.width, 0.5) #0.5for rotation
            output = np.array(output)
            eng.quit()
        if self.save_data:
            np.save('./output-%s-%sus.npy'%(file_name,int(self.input_time/us)),output)
        return output

    def build_motion_excitation_layer_simple(self, interval, time):
        output = np.zeros((self.height,self.width,time))
        for i in range(self.height):
            for j in range(self.width):
                s= 0
                for t in range(len(interval[i*self.width + j])-1):
                    s += interval[i*self.width + j][t]
                    if interval[i*self.width + j][t]  == 0: continue
                    if (abs(256/interval[i*self.width + j][t+1] - 256/interval[i*self.width + j][t])) > 5 and (abs(interval[i*self.width + j][t+1]-interval[i*self.width + j][t])>1):
                        output[i,j,s: s+ interval[i*self.width + j][t+1]] = 1
        return output  

    def build_motion_excitation_layer_confid(self, interval, file_name):
        sio.savemat('interval_confid-%s-%sus.mat'%(file_name,int(self.input_time/us)), {'itv': interval})
        eng = matlab.engine.start_matlab()
        output = eng.motioncondifence(file_name,'%d'%(self.input_time/us),float(self.input_time/25/us), 0.5) #0.5for rotation
        output = np.array(output)
        eng.quit()
        output[output>=0.1] = 1
        output[output<0.1] = 0
        #np.save('output-%s-%sus.npy'%(file_name,int(input_time/us)),output)
        return output

    def dynamic_spike_refining(self, a, b, c, t, i, j):
        itv = b
        f = 3
        openflag = 0
        if a < b - 1 and b - 1 > c and openflag == 1:
            tl = min(f*a/b,b/4)
            tr = min(f*c/b,b/4)
            te = b - tl -tr
            t_beforeitv = tl
            if t <= tl:
                if self.last_change[i*self.width+j] > 0:
                    itv = self.last_change[i*self.width+j]
                else:
                    itv = a
            elif t > b - tr:
                itv = c
            else:
                if a*c - c*tl - a*tr == 0:
                    itv = a*c*te
                else:
                    itv = a*c*te/(a*c - c*tl - a*tr)
        elif a > b + 1 and b + 1 < c and openflag == 1:
            tl = min(f*a/b,b/4)
            tr = min(f*c/b,b/4)
            te = b - tl -tr
            t_beforeitv = tl
            if t <= tl:
                if self.last_change[i*self.width+j] > 0:
                    itv = self.last_change[i*self.width+j]
                else:
                    itv = a
            elif t > b - tr:
                itv = c
            else:
                if a*c - c*tl - a*tr == 0:
                    itv = a*c*te
                else:
                    itv = a*c*te/(a*c - c*tl - a*tr)
        elif (a > b+1 and b > c+1) or (a+1 < b and b+1 < c):
            tm = a*(c-b)/(c-a)
            #te = min(tm,b-tm)
            te = 0
            tl = tm - te/2
            tr = b - tm - te/2
            t_beforeitv = tl
            if t <= tl:
                if self.last_change[i*self.width+j] > 0:
                    itv = self.last_change[i*self.width+j]
                else:
                    itv = a
            elif t > b - tr:
                itv = c
            else:
                if a*c - c*tl - a*tr == 0:
                    itv = a*c*te
                else:
                    itv = a*c*te/(a*c - c*tl - a*tr)
        else:
            self.last_change[i*self.width+j] = 0
            # if abs(a-b)<3 and abs(b-c)<3:
            #     itv = ((a+b+c)/3)
            # else:
            itv = b
            t_beforeitv = 0
        return abs(itv), t_beforeitv

    def dynamic_spike_refining_v2(self, a, b, c, t, i, j):
        itv = b
        f = 1
        openflag = 0
        if a < b - 1 and b - 1 > c and openflag == 1:
            tl = min(f*a/b,b/4)
            tr = min(f*c/b,b/4)
            te = b - tl -tr
            t_beforeitv = tl
            if t <= tl:
                if self.last_change[i*self.width+j] > 0:
                    itv = self.last_change[i*self.width+j]
                else:
                    itv = a
            elif t > b - tr:
                itv = c
            else:
                if a*c - c*tl - a*tr == 0:
                    itv = a*c*te
                else:
                    itv = a*c*te/(a*c - c*tl - a*tr)
        elif a > b + 1 and b + 1 < c and openflag == 1:
        
            tl = min(f*b/c,b/4)
            tr = min(f*b/a,b/4)
            te = b - tl -tr
            t_beforeitv = tl
            if t <= tl:
                if self.last_change[i*self.width+j] > 0:
                    itv = self.last_change[i*self.width+j]
                else:
                    itv = a
            elif t > b - tr:
                itv = c
            else:
                if a*c - c*tl - a*tr == 0:
                    itv = a*c*te
                else:
                    itv = a*c*te/(a*c - c*tl - a*tr)
        elif (a > b and b > c) or (a < b and b < c):
            tle = a*min(float(a-b)/(2*(a-c)),float(b-c)/(2*(a-c)))
            tre = c*min(float(a-b)/(2*(a-c)),float(b-c)/(2*(a-c)))
            tm = float(a)*(c-b)/(c-a)
            te = min(tm-tle,b-tm-tre)
            assert(te>=0)
            te = 2 * te
            #te = min(tm,b-tm)
            te = 0
            tl = tm - te/2
            tr = b - tm - te/2
            t_beforeitv = tl
            if t <= tl:
                if self.last_change[i*self.width+j] > 0:
                    itv = self.last_change[i*self.width+j]
                else:
                    itv = a
            elif t > b - tr:
                itv = c
            else:
                if a*c - c*tl - a*tr == 0:
                    itv = a*c*te
                else:
                    itv = a*c*te/(a*c - c*tl - a*tr)
        else:
            self.last_change[i*self.width+j] = 0
            # if abs(a-b)<3 and abs(b-c)<3:
            #     itv = ((a+b+c)/3)
            # else:
            itv = b
            t_beforeitv = 0
        return abs(itv), t_beforeitv

    def build_spike_refining_layer(self, number_of_neurons, record_spike, tau, vth, tauth):

        """
        :param number_of_neurons:
        :param record_rate:
        :param record_spike:
        :return:
        """
        """ =========================== MODEL ========================== """
        eqs_neuron = Equations('''
                    dv/dt = -v/tau : volt (unless refractory) 
                    dv_thresh_i/dt = (vth - v_thresh_i)/tauth : volt 
                    x : meter
                    y : meter
                    ''')
        
        """ ================== THRESHOLD & REFRACTORY ================= """
        eqs_thresh = 'v > v_thresh_i'
        t_refractory = 10*us
        """ =========================== RESET ========================== """
        eqs_reset = '''
                    v = 0*mV
                    v_thresh_i = 0.0001*mV
                    '''

        ng = NeuronGroup(number_of_neurons, 
                                model=eqs_neuron, 
                                reset=eqs_reset, 
                                threshold=eqs_thresh,
                                refractory=t_refractory, method='exact')
        ng.v_thresh_i = 0.001*mV
        #rate_monitor = PopulationRateMonitor(ng) if record_rate else None
        spike_monitor = SpikeMonitor(ng) if record_spike else None
        return ng, spike_monitor

    def build_visual_reconstruction_layer(self, number_of_neurons, record_v, Ee, vr, El, taum, vth, tauth_recon, taue):
        """
        :param number_of_neurons:
        :param record_rate:
        :param record_spike:
        :return:
        """
        """ =========================== MODEL ========================== """
        eqs_neuron = Equations('''
                    dv1/dt = ( ge * (Ee-vr) + El  - v1) / taum : volt
                    dge/dt = -ge / taue : 1
                    dv_thresh_adaptive/dt = (vth - v_thresh_adaptive)/tauth_recon : volt 
                    x : meter
                    y : meter
                    ''')
        """ ================== THRESHOLD & REFRACTORY ================= """
        eqs_thresh = 'v1 > v_thresh_adaptive'
        t_refractory = 10*us
        """ =========================== RESET ========================== """
        eqs_neuron_reset = '''
                    v1 = 0*mV
                    v_thresh_adaptive += 3*mV
                    '''
        ng = NeuronGroup(number_of_neurons,
                        model=eqs_neuron, 
                        reset=eqs_neuron_reset, 
                        threshold=eqs_thresh, 
                        refractory=t_refractory, 
                        method='exact')
        ng.v_thresh_adaptive = 20*mV

        v_thresh_monitor = StateMonitor(ng, 'v_thresh_adaptive', record=True, dt =25*us) 
        v_monitor = StateMonitor(ng, 'v1', record=True, dt =self.height*us) if record_v else None
        return ng, v_monitor, v_thresh_monitor

    def build_stdp_connection(self, input_ng, output_ng, stdp_on, taupre, taupost):

        """ =========================== MODEL ========================== """
        # STDP synaptic
        eqs_stdp = Equations('''w1 : 1
                    weight : 1
                    dApre/dt = -Apre / taupre : 1 (clock-driven)
                    dApost/dt = -Apost / taupost : 1 (clock-driven)''')

        # setting STDP update rule
        # v1+=50*weight*mV
        # ge += w1
        # Apre += dApre
        # w1 = clip(w1 + Apost, 0, gmax)

        #oringinal STDP
            # pre = '''
            #             v1+=500*weight*w1*mV
            #             Apre += dApre
            #             w1 = clip(w1 + Apost, 0.01, gmax)
            #            '''
            # post = '''Apost += dApost
            #             w1 = clip(w1 + Apre, 0.01, gmax)
            #             '''
                        #  v1+=50*weight*mV
                        # ge += w1/9
                        # Apre += dApre/9
        if  stdp_on:
            model = eqs_stdp
            pre = '''
                        v1+=50*weight*mV
                        ge += w1/9
                        Apre += dApre/9
                        w1 = clip(w1 + Apost, 0, gmax)
                    '''
            post = '''Apost += dApost
                        w1 = clip(w1 + Apre, 0, gmax)
                        '''
                #oringinal STDP
            # pre = '''
            #             v1+=500*weight*w1*mV
            #             Apre += dApre
            #             w1 = clip(w1 + Apost, 0.01, gmax)
            #            '''
            # post = '''Apost += dApost
            #             w1 = clip(w1 + Apre, 0.01, gmax)
            #             '''
        else:
            model = eqs_stdp
            pre = 'v1 += 10*weight*mV' #10
            post = ''
        conn = Synapses(input_ng, output_ng, model=model, on_pre=pre, on_post=post)
        
        #mon = StateMonitor(conn, 'w1', record=True, dt =25*us) if record_v else None
        return conn

    def build_ordinary_connection(self, input_ng, output_ng):
        conn = Synapses(input_ng, output_ng, on_pre='''v += 10*mV''')
        return conn

    def build_spatial_connection(self, conn, ng_e1, ng_e2,size):
        grid_dist = 10*umeter
        width = self.width
        height = self.height
        ng_e1.x = '(i // width) * grid_dist - width/2.0 * grid_dist'
        ng_e1.y = '(i % width) * grid_dist - height/2.0 * grid_dist'
        ng_e2.x = '(i // width) * grid_dist - width/2.0 * grid_dist'
        ng_e2.y = '(i % width) * grid_dist - height/2.0 * grid_dist'
        distance = size*umeter
        conn.connect('sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2) < distance')
        conn.weight = 'exp(-10*sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2)/distance)'
        return conn

    def cal_scale(self, interval):
        itv = 0
        for i in range(0, self.height):
            for j in range(0, self.width):
                sum = 0
                for k in range(int(self.run_time/us)):
                    sum = sum + interval[i*self.width + j][k]
                    if sum >= 300:
                        break
                itv =  itv + interval[i*self.width + j][k]
        avgl = itv*1.0/(self.height*self.width)
        return avgl

    def find_indice(self, interval, t):
        sum_ref = 0
        for k in range(int(self.run_time/us)):
            sum_ref = sum_ref + interval[k]
            if sum_ref >= t:
                break
        return sum_ref, k

    def generate_mapping(self, motioncut, v_th_matrix, s, t, maxvalue):
        alpha =  [[] for i in range(maxvalue)]
        for i in range(0, self.height):
            for j in range(0, self.width):
                if motioncut[i][j][t] == 1 or np.sum(motioncut[i][j][s:t])>5:
                    continue
                else:
                    sum_ref, k = self.find_indice(self.interval[i*self.width + j], t)
                    b = self.interval[i*self.width + j][k]
                    alpha[b].append(v_th_matrix[i*self.width+j])
        for i in range(maxvalue):
            alpha[i] = mean(alpha[i])
        return alpha

    def recon_visual_image(self, t):
        v = self.vth_e2
        interval = self.interval
        if self.camera_moving != True:
            motioncut = self.motioncut
        stable_ts = self.stable_ts
        dynamic_refining_flag = True

        w = self.width
        h = self.height
        img_mat = np.zeros((h, w), dtype=np.float)
        #k_store = np.zeros((h, w), dtype=np.float)
        #sum_store = np.zeros((h, w), dtype=np.float)
        #b1 = np.zeros((h, w), dtype=np.float)
        img_mat1 = np.zeros((h, w), dtype=np.float)
        cutmap = np.zeros((h, w), dtype=np.float)   
        sum_ref = 0.0
        num = 0
        v_th_matrix = v.v_thresh_adaptive[:,t*self.scale_ts]/mvolt
        for i in range(0, h):
            for j in range(0, w):
                sum_ref = sum_ref + v_th_matrix[i*w+j]#v[i*w+j].v_thresh_adaptive[t*scale_ts]/mvolt
                num = num + 1
        v_ref = sum_ref / num

        if self.stable_ts > 0:
            st = stable_ts
        else:
            st = t
        if t-1000>=0:
            s = t-1000
        else:
            s = 100 

        if self.camera_fix_1:
            maxvalue = 256
            alpha = self.generate_mapping(motioncut, v_th_matrix, s, t, maxvalue)
            x = []
            y = []
            for i in range(256):
                if alpha[i] > 0:
                    x.append(i)
                    y.append(alpha[i])
            x.append(256)
            y.append(0)
            a11=np.polyfit(x,y,2)
            b11=np.poly1d(a11)

            for i in range(0, h):
                for j in range(0, w):
                    sum_ref, k = self.find_indice(interval[i*w + j], st)
                    itv =  interval[i*w + j][k]
                    sum_ref, k = self.find_indice(interval[i*w + j], t)

                    a = interval[i*w + j][k-1]
                    b = interval[i*w + j][k]
                    c = interval[i*w + j][k+1]
                    
                    if dynamic_refining_flag or motioncut[i][j][t] == 1 or np.sum(motioncut[i][j][s:t])>5: 
                        itv, t_beforeitv = self.dynamic_spike_refining_v2(a, b, c, t - (sum_ref - interval[i*w + j][k]), i, j)
                    else:
                        itv =  interval[i*w + j][k]
                        t_beforeitv = 0
                    if itv==0: itv=1
                    img_mat[i][j] = b11(itv)
                    if img_mat[i][j] < 0:
                        img_mat[i][j] = 0

        if self.camera_fix_2:
            maxvalue = 1000
            alpha = self.generate_mapping(motioncut, v_th_matrix, s, t, maxvalue)
            for i in range(maxvalue):
                alpha[i] = mean(alpha[i])
            itv1 = 0
            alpha_itv1 = 0
            if np.isnan(alpha[1]):
                for i in range(1,maxvalue):
                    if np.isnan(alpha[i]):
                        continue
                    itv1 = i
                    alpha_itv1 = alpha[i]
                    break
                alpha[1] = itv1*alpha_itv1
            if np.isnan(alpha[maxvalue-1]):
                alpha[maxvalue-1] = 1
            #alpha[1] = itv1*alpha_itv1
            if np.isnan(alpha[0]):
                alpha[0] = alpha[1]*1.2
            nanlist = []
            for i in range(maxvalue):
                if np.isnan(alpha[i]):
                    nanlist.append(i)
            idx1 = 0
            idx2 = 0
            for i in range(len(nanlist)):
                idx = nanlist[i]
                for idx1 in range(idx,0,-1):
                    if ~np.isnan(alpha[idx1]):
                        break
                for idx2 in range(idx,maxvalue,1):
                    if ~np.isnan(alpha[idx2]):
                        break
                alpha[idx] = alpha[idx1] - (abs(alpha[idx1] - alpha[idx2]))*(idx - idx1)/(idx2-idx1)

            x = []
            y = []
            for i in range(maxvalue):
                if alpha[i] > 0:
                    x.append(i)
                    y.append(alpha[i])
            x.append(maxvalue)
            y.append(1)
            a11=np.polyfit(x,y,2)
            b11=np.poly1d(a11)

            s = 0
            for i in range(0, h):
                for j in range(0, w):
                    if motioncut[i][j][t] == 1: 
                        sum_ref, k = self.find_indice(interval[i*w + j], t)
                        a = interval[i*w + j][k-1]
                        b = interval[i*w + j][k]
                        try:
                            c = interval[i*w + j][k+1]
                        except:
                            c = b
                        itv, t_beforeitv = self.dynamic_spike_refining_v2(a, b, c, t - (sum_ref - interval[i*w + j][k]), i, j)
                        img_mat[i][j] = alpha[itv]
                        img_mat1[i][j] = alpha[itv]
                        #cutmap[i][j] = 255
                    elif np.sum(motioncut[i][j][s:t])>5:
                        cutmap[i][j] = 255
                        sum_ref, k = self.find_indice(interval[i*w + j], t)
                        a = interval[i*w + j][k-1]
                        b = interval[i*w + j][k]
                        try:
                            c = interval[i*w + j][k+1]
                        except:
                            c = b
                        itv, t_beforeitv = self.dynamic_spike_refining_v2(a, b, c, t - (sum_ref - interval[i*w + j][k]), i, j)
                        img_mat[i][j] = alpha[itv]
                        img_mat1[i][j] = alpha[itv]
                        
                    else:
                        img_mat[i][j] = v_th_matrix[i*w+j]
                        
                        sum1 = 0
                        for k in range(int(self.run_time/us)):
                            sum1 = sum1 + interval[i*w + j][k]
                            if sum1 >= t:
                                break
                        try:
                            a = interval[i*w + j][k-1]
                            b = interval[i*w + j][k]
                            c = interval[i*w + j][k+1]
                            itv, t_beforeitv = self.dynamic_spike_refining_v2(a, b, c, t - (sum1 - interval[i*w + j][k]), i, j)

                            x_list = [a/2, a + t_beforeitv + itv/2.0, a + itv + c/2.0]
                            y_list = [alpha[a], alpha[itv], alpha[c]]
                            w_list = np.polyfit(x_list, y_list, 1)
                            img_mat1[i][j] = np.polyval(w_list, a + t - (sum1 - interval[i*w + j][k]))             
                        except:
                            img_mat1[i][j] = v_th_matrix[i*w+j]
                    if img_mat[i][j] < 0:
                        img_mat[i][j] = 0

        if self.camera_moving:
            for i in range(0, h):
                for j in range(0, w):
                
                    sum_ref = 0
                    for k in range(int(self.run_time/us)):
                        sum_ref = sum_ref + interval[i*w + j][k]
                        if sum_ref >= t:
                            break
                    
                    '''============================dynamic refining=============================='''
                    if dynamic_refining_flag:
                        a = interval[i*w + j][k-1]
                        b = interval[i*w + j][k]
                        try:
                            c = interval[i*w + j][k+1]
                        except:
                            c = b
                        itv, t_beforeitv = self.dynamic_spike_refining_v2(a, b, c, t - (sum_ref - interval[i*w + j][k]), i, j)
                    else:
                        itv =  interval[i*w + j][k]
                    if itv==0: itv=1
                    if a==0: a=1
                    if c==0: c=1
                    if itv != b:
                        img_mat[i][j] = v_ref/itv
                        img_mat1[i][j] = v_ref/itv  
                    else:
                        x_list = [a/2.0, a + t_beforeitv + itv/2.0, a + itv + c/2.0]
                        y_list = [v_ref/a, v_ref/itv, v_ref/c]
                        w_list = np.polyfit(x_list, y_list, 1)
                        img_mat[i][j] = np.polyval(w_list, a + t - (sum_ref - interval[i*w + j][k]))                    
                        img_mat1[i][j] = img_mat[i][j]
                    if img_mat[i][j] < 0:
                        img_mat[i][j] = 0
                    cutmap[i][j] = 255
        return img_mat


