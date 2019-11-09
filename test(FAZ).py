import tensorflow as tf
import numpy as np
import os
import scipy.misc as misc
import natsort
import time
import utils
import lossfunc
from options.test_options import TestOptions
import model

def main(argv=None):
    opt = TestOptions().parse()
    test_results = os.path.join(opt.saveroot,'test_results')
    utils.check_dir_exist(test_results)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    DATA_SIZE = opt.data_size.split(',')
    DATA_SIZE = [int(DATA_SIZE[0][1:]),int(DATA_SIZE[1]),int(DATA_SIZE[2][:-1])]
    BLOCK_SIZE = opt.block_size.split(',')
    BLOCK_SIZE = [int(BLOCK_SIZE[0][1:]),int(BLOCK_SIZE[1]),int(BLOCK_SIZE[2][:-1])]

    x=tf.placeholder(tf.float32, shape=[None] + BLOCK_SIZE + [opt.input_nc+1], name="input_image")
    y=tf.placeholder(tf.int32, shape=[None, 1, BLOCK_SIZE[1], BLOCK_SIZE[2], 1], name="annotation")
    y_,pred_, variables,sf= model.IPN(x=x,PLM_NUM=opt.PLM_num, LAYER_NUM=opt.layer_num,NUM_OF_CLASS=opt.NUM_OF_CLASS)
    model_loss = lossfunc.cross_entropy(y_,y)

    sess = tf.Session()
    print("Setting up Saver...")
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    restore_path = os.path.join('FAZ_logs','best_model',natsort.natsorted(os.listdir(os.path.join('FAZ_logs','best_model')))[-1])
    o_itr = natsort.natsorted(os.listdir(restore_path))[-1][11:-5]
    saver.restore(sess, os.path.join(restore_path,'model.ckpt-'+o_itr))
    print("Model restored...")

    test_images= np.zeros((1, BLOCK_SIZE[0], BLOCK_SIZE[1], BLOCK_SIZE[2], opt.input_nc+1))
    cube_images= np.zeros((1, BLOCK_SIZE[0], DATA_SIZE[1], DATA_SIZE[2], opt.input_nc+1))
    test_annotations = np.zeros((1,1,BLOCK_SIZE[1],BLOCK_SIZE[2],1))

    modalitylist = os.listdir(os.path.join(opt.dataroot,opt.mode))
    modalitylist = natsort.natsorted(modalitylist)
    print(modalitylist)

    result = np.zeros((DATA_SIZE[1], DATA_SIZE[2]))

    cubelist = os.listdir(os.path.join(opt.dataroot, opt.mode,modalitylist[0]))
    cubelist = natsort.natsorted(cubelist)

    for kk,cube in enumerate(cubelist):
        loss2 = 0
        bscanlist = os.listdir(os.path.join(opt.dataroot, opt.mode, modalitylist[0], cube))
        bscanlist=natsort.natsorted(bscanlist)
        for i,bscan in enumerate(bscanlist):
            for j,modal in enumerate(modalitylist):
                if modal!="label":
                    cube_images[0,:,:,i,j]=np.array(misc.imresize(misc.imread(os.path.join(opt.dataroot,opt.mode,modal,cube,bscan)),[BLOCK_SIZE[0], DATA_SIZE[1]], interp='nearest'))
            cube_images[0, :, :, i, j] = np.array(misc.imresize(misc.imread(os.path.join('FAZ_logs','distancemap', bscan)),[BLOCK_SIZE[0], DATA_SIZE[1]], interp='nearest'))

        for i in range(DATA_SIZE[1] // BLOCK_SIZE[1]):
            for j in range(0, DATA_SIZE[2] // BLOCK_SIZE[2]):
                test_images[0, 0:BLOCK_SIZE[0], 0:BLOCK_SIZE[1], 0:BLOCK_SIZE[2], :] = cube_images[0, :,BLOCK_SIZE[1] * i:BLOCK_SIZE[1] * (i + 1), BLOCK_SIZE[2] * j:BLOCK_SIZE[2] * (j + 1), :]
                score,result0,piece_loss,sf0 = sess.run([y_,pred_,model_loss,sf], feed_dict={x: test_images,y: test_annotations})
                result[BLOCK_SIZE[1] * i:BLOCK_SIZE[1] * (i + 1), BLOCK_SIZE[2] * j:BLOCK_SIZE[2] * (j + 1)] = sf0[0, 0, :,:,1] * 255
        for num in range(1,20):
            nx = int(np.random.normal(DATA_SIZE[1], 50))
            ny = int(np.random.normal(DATA_SIZE[2], 50))
            mx = int(BLOCK_SIZE[1]/2)
            my = int(BLOCK_SIZE[2]/2)
            if nx<=BLOCK_SIZE[1]/2 or nx>=DATA_SIZE[1]-mx:
                nx=int(DATA_SIZE[1]/2)
            if ny<=BLOCK_SIZE[2]/2 or ny>=DATA_SIZE[2]-my:
                ny=int(DATA_SIZE[2]/2)
            test_images[0, 0:BLOCK_SIZE[0], 0:BLOCK_SIZE[1], 0:BLOCK_SIZE[2], :] = cube_images[0, :,(nx-mx):(nx+mx),(ny-my):(ny+my), :]
            score, result0, piece_loss, sf0 = sess.run([y_, pred_, model_loss, sf],
                                                       feed_dict={x: test_images, y: test_annotations})
            result[(nx-mx):(nx+mx),(ny-my):(ny+my)] =result[(nx-mx):(nx+mx),(ny-my):(ny+my)]/2+ sf0[0, 0, :,:, 1] * 255/2

        print("Saved image: ", cube)
        misc.imsave(os.path.join(test_results,cube+"_FAZ_pre.bmp"), result.astype(np.uint8))




if __name__ == "__main__":
    tf.app.run()