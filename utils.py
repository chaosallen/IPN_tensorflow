import os
import numpy as np
import copy

def check_dir_exist(dir):
    """create directories"""
    if os.path.exists(dir):
        return
    else:
        names = os.path.split(dir)
        dir = ''
        for name in names:
            dir = os.path.join(dir,name)
            if not os.path.exists(dir):
                try:
                    os.mkdir(dir)
                except:
                    pass
        print('dir','\''+dir+'\'','is created.')

def cal_downsampling_size_combine(size,num):
    th = size*11//10 #range of product of pooling size
    seq = cal_single_ds_c(size,num,th)
    if not seq:
        print('Cal failed! Please redefine num!')
    seq.reverse()
    return seq

def cal_channel_num(num):
    assert num>0
    channel = []
    size = [32,64,128,256,512]
    size_num = np.zeros(5,dtype='int')
    if num >= 1:
        size_num[0] = 1
    if num >= 2:
        size_num[1] = 1
    if num >= 3:
        temp1 = (num-2) // 3
        temp2 = (num-2) % 3
        size_num[2:] = size_num[2:] + temp1
        if temp2 >= 1:
            size_num[2] += 1
        if temp2 == 2:
            size_num[3] += 1
    for i, s in enumerate(size):
        for j in range(size_num[i]):
            channel.append(s)
    return channel

def cal_single_ds_c(maximum,num,th,ds_c = [],index=1):
    #maximum:size, num:stride_num, th:the range deviate from size, ds_c:seq, index:the current product
    if index>=maximum:
        if index > th or len(ds_c) != num:
            return
        else:
            final_ds_c = copy.deepcopy(ds_c)
            return final_ds_c

    if len(ds_c) > num:
        return

    if ds_c:
        start = ds_c[-1]
    else:
        start = 2

    final_ds_c = []
    # max size 10000
    for i in range(start,101):
        ds_c.append(i)
        index *= i
        temp = cal_single_ds_c(maximum,num,th,ds_c,index)
        if temp:
            if final_ds_c:
                if cal_product(temp) < cal_product(final_ds_c):
                    final_ds_c = temp
                if cal_product(temp) == cal_product(final_ds_c):
                    if np.sum(np.array(temp)) <  np.sum(final_ds_c):
                        final_ds_c = temp
            else:
                final_ds_c = temp
        ds_c.pop(-1)
        index /= i

    return  final_ds_c

def cal_product(list):
    index = 1
    for l in list:
        index *= l
    return index

#pm
def cal_Dice(img1,img2):
    shape = img1.shape
    I = 0
    U = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img1[i,j] >= 1 and img2[i,j] >= 1:
                I += 1
            if img1[i,j] >= 1 or img2[i,j] >= 1:
                U += 1
    return 2*I/(I+U+1e-5)

def cal_Dice_para(img1,img2):
    shape = img1.shape
    I = 0
    U = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img1[i,j] >= 1 and img2[i,j] >= 1:
                I += 1
            if img1[i,j] >= 1 or img2[i,j] >= 1:
                U += 1
    return I,U


def cal_acc(img1,img2):
    shape = img1.shape
    acc = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img1[i,j] == img2[i,j]:
                acc += 1
    return acc/(shape[0]*shape[1])

def test():
    size = 11
    num = 3
    resize = 1
    seq = cal_downsampling_size_combine(size,num)
    for index in seq:
        resize *= index
    print('{}->{} = '.format(size,resize),end='')
    for i,s in enumerate(seq):
        if i == 0:
            print(str(s),end='')
        else:
            print('x'+str(s),end='')
test()