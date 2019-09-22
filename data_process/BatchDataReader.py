import os
import numpy as np
import scipy.misc as misc
import h5py
import random

class BatchDatset:
    def __init__(self, records_list,datasize,blocksize,channels,batch_size,cube_num,dataclass,saveroot):
        self.saveroot = saveroot
        self.filelist = records_list
        self.datasize = datasize
        self.blocksize = blocksize
        self.channels = channels
        self.batch_size = batch_size
        self.dataclass = dataclass

        self.images = np.zeros((batch_size, blocksize[0], blocksize[1], blocksize[2], channels))
        self.annotations = np.zeros((batch_size, 1, blocksize[1], blocksize[2], 1))
        self.transformkey=0
        self.top = 0
        self.left = 0
        self.isEpoch = False

        if datasize[0]!=blocksize[0]:#if reduce the height of data
            self.transformkey = 1

        #self.cube_num = len(list(self.filelist['label']))
        self.cube_num = cube_num

        self.data = np.zeros((blocksize[0], datasize[1], datasize[2], self.cube_num, channels), dtype=np.uint8)
        self.label = np.zeros((1, datasize[1], datasize[2], self.cube_num), dtype=np.uint8)
        self.read_images()

        self.pos_start = 0

    def read_images(self):
        if not os.path.exists(os.path.join(self.saveroot,self.dataclass+"data.hdf5")):
            print(self.dataclass+"picking ...It will take some minutes")
            modality_num = -1
            for modality in self.filelist.keys():
                if modality != 'label':
                    ctlist=list(self.filelist[modality])
                    modality_num+=1
                    ct_num=-1
                    for ct in ctlist:
                        ct_num+=1
                        scanlist=list(self.filelist[modality][ct])
                        scan_num=-1
                        for scan in scanlist:
                            scan_num+=1
                            self.data[:,:,scan_num,ct_num,modality_num]=np.array(self.image_transform(scan,self.transformkey))
                else:
                    ctlist=list(self.filelist[modality])
                    ct_num=-1
                    for ct in ctlist:
                        ct_num+=1
                        labeladress=self.filelist[modality][ct]
                        self.label[0,:,:,ct_num]=np.array(self.image_transform(labeladress,0))
            f= h5py.File(os.path.join(self.saveroot,self.dataclass+"data.hdf5"), "w")
            f.create_dataset('data',data=self.data)
            f.create_dataset('label',data=self.label)
            f.close
        else:
            print("found pickle !!!")
            f = h5py.File(os.path.join(self.saveroot,self.dataclass+"data.hdf5"), "r")
            self.data = f['data']
            self.label = f['label']
            f.close


    def image_transform(self, filename, key):
        image = misc.imread(filename)
        if key:
            resize_image = misc.imresize(image,[self.blocksize[0], self.datasize[1]], interp='nearest')
        else:
            resize_image = image
        return np.array(resize_image)

    def read_batch_random_train(self):#vessel segmentation
        for batch in range(0,self.batch_size):
            nx=random.randint(self.blocksize[1]/2,self.datasize[1]-self.blocksize[1]/2)
            ny=random.randint(self.blocksize[2]/2,self.datasize[2]-self.blocksize[2]/2)
            startx = nx-int(self.blocksize[1]/2)
            endx = nx+int(self.blocksize[1]/2)
            starty= ny-int(self.blocksize[2]/2)
            endy = ny + int(self.blocksize[2]/2)
            ctnum = random.randint(0, self.cube_num - 1)
            self.images[batch, :, 0:self.blocksize[1], 0:self.blocksize[2]] = self.data[:, startx:endx,starty:endy, ctnum].astype(np.float32)
            self.annotations[batch,0,0:self.blocksize[1],0:self.blocksize[2],0]=self.label[:,startx:endx,starty:endy, ctnum].astype(np.float32)
        return self.images, self.annotations

    def read_batch_normal_train(self):#FAZ segmentation
        sd=50 #Standard Deviation
        for batch in range(self.batch_size):
            nx=int(np.random.normal(self.datasize[1]/2,sd))
            ny=int(np.random.normal(self.datasize[2]/2,sd))
            startx = nx-int(self.blocksize[1]/2)
            endx = nx+int(self.blocksize[1]/2)
            starty= ny-int(self.blocksize[2]/2)
            endy = ny + int(self.blocksize[2]/2)
            if startx<0 or starty<0 or endx>self.datasize[1] or endy>self.datasize[2]:
                startx = int(self.datasize[1] / 2)
                starty = int(self.datasize[2] / 2)
                endx =startx+int(self.blocksize[1])
                endy = starty + int(self.blocksize[2])
            ctnum = random.randint(0, self.cube_num - 1)

            self.images[batch, :, 0:self.blocksize[1], 0:self.blocksize[2]] = self.data[:, startx:endx,starty:endy, ctnum].astype(np.float32)
            self.annotations[batch, 0, 0:self.blocksize[1], 0:self.blocksize[2], 0] = self.label[:, startx:endx,starty:endy, ctnum].astype(np.float32)
        return self.images, self.annotations

    def read_batch_normal_valid(self):
        sd=50 #Standard Deviation

        for batch in range(self.batch_size):
            nx=int(np.random.normal(self.datasize[1]/2,sd))
            ny=int(np.random.normal(self.datasize[2]/2,sd))
            startx = nx-int(self.blocksize[1]/2)
            endx = nx+int(self.blocksize[1]/2)
            starty= ny-int(self.blocksize[2]/2)
            endy = ny + int(self.blocksize[2]/2)
            if startx<0 or starty<0 or endx>self.datasize[1] or endy>self.datasize[2]:
                startx = int(self.datasize[1] / 2)
                starty = int(self.datasize[2] / 2)
                endx = startx+int(self.blocksize[1])
                endy = starty + int(self.blocksize[2])
            #ctnum = random.randint(0, self.scan_num - 1)
            self.images[batch, :, 0:self.blocksize[1], 0:self.blocksize[2]] = self.data[:, startx:endx,starty:endy, self.pos_start].astype(np.float32)
            self.annotations[batch, 0, 0:self.blocksize[1], 0:self.blocksize[2], 0] = self.label[:, startx:endx,starty:endy, self.pos_start].astype(np.float32)
            self.pos_start += 1
            if self.pos_start == self.cube_num:
                self.pos_start = 0
        return self.images, self.annotations

    def read_batch_normal_valid_all(self,batch_size):
        for batch in range(batch_size):
            #print(batch,self.pos_start)
            self.images[batch, :, 0:self.blocksize[1], 0:self.blocksize[2]] = self.data[:, self.top:self.top+self.blocksize[1],self.left:self.left+self.blocksize[2], self.pos_start].astype(np.float32)
            self.annotations[batch, 0, 0:self.blocksize[1], 0:self.blocksize[2], 0] = self.label[:, self.top:self.top+self.blocksize[1],self.left:self.left+self.blocksize[2], self.pos_start].astype(np.float32)
            self.left += self.blocksize[2]
            if self.left + self.blocksize[2] > self.datasize[2]:
                self.left = 0
                self.top += self.blocksize[1]
                if self.top + self.blocksize[1] > self.datasize[1]:
                    self.top = 0
                    self.pos_start += 1
                    if self.pos_start == self.cube_num:
                        self.pos_start = 0

        return self.images, self.annotations



