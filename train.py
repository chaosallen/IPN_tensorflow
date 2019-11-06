"""
Image Projection Network 1.2
From 3D to 2D image segmentation
Author 'Mingchao Li, Yerui Chen'
"""

import tensorflow as tf
import numpy as np
import lossfunc
import data_process.readData as readData
import data_process.BatchDataReader as BatchDataReader
from options.train_options import TrainOptions
import model
import utils
import shutil
import os
import natsort
from six.moves import xrange

def main(argv=None):
    opt = TrainOptions().parse()
    model_save_path = os.path.join(opt.saveroot,'checkpoints')
    best_model_save_path = os.path.join(opt.saveroot,'best_model')
    utils.check_dir_exist(model_save_path)
    utils.check_dir_exist(best_model_save_path)

    train_data_num = len(os.listdir(os.path.join(opt.dataroot,'train','image1')))
    val_data_num = len(os.listdir(os.path.join(opt.dataroot,'val','image1'))) #validation cube num
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    DATA_SIZE = opt.data_size.split(',')
    DATA_SIZE = [int(DATA_SIZE[0][1:]),int(DATA_SIZE[1]),int(DATA_SIZE[2][:-1])]
    BLOCK_SIZE = opt.block_size.split(',')
    BLOCK_SIZE = [int(BLOCK_SIZE[0][1:]),int(BLOCK_SIZE[1]),int(BLOCK_SIZE[2][:-1])]

    copy_file_suffixs = ['.meta','.index','.data-00000-of-00001']
    x=tf.placeholder(tf.float32, shape=[None, BLOCK_SIZE[0], BLOCK_SIZE[1], BLOCK_SIZE[2], opt.input_nc], name="input_image")
    y=tf.placeholder(tf.int32, shape=[None, 1, BLOCK_SIZE[1], BLOCK_SIZE[2], 1], name="annotation")
    y_,pred_, variables,sf = model.IPN(x=x,PLM_NUM=opt.PLM_num, LAYER_NUM=opt.layer_num,NUM_OF_CLASS=opt.NUM_OF_CLASS)
    # Loss function
    loss = lossfunc.cross_entropy(y_,y)

    # Train initialize
    trainable_var = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(opt.lr)
    grads = optimizer.compute_gradients(loss, trainable_var)
    train_op = optimizer.apply_gradients(grads)

    # Read Data
    print("Start Setup dataset reader")
    train_records, validation_records = readData.read_dataset(opt.dataroot)
    #if you haven't original data but have hdf5 files, please use the following 2 lines' code replaces previoes line, and it will not influence final result
    #training_records = []
    #validation_records = []
    print("Setting up dataset reader")
    train_dataset_reader = BatchDataReader.BatchDatset(train_records,DATA_SIZE,BLOCK_SIZE,opt.input_nc,opt.batch_size,train_data_num,"train",opt.saveroot)
    validation_dataset_reader = BatchDataReader.BatchDatset(validation_records,DATA_SIZE,BLOCK_SIZE,opt.input_nc,opt.batch_size,val_data_num,"validation",opt.saveroot)

    sess = tf.Session()
    print("Setting up Saver...")
    sess.run(tf.global_variables_initializer())
    o_itr = 0

    saver = tf.train.Saver()
    if opt.useRestore:
        best_dice = natsort.natsorted(os.listdir(best_model_save_path))[-1]
        restore_path = os.path.join(best_model_save_path,best_dice)
        o_itr = int(natsort.natsorted(os.listdir(restore_path))[-1][11:-5])
        saver.restore(sess, os.path.join(restore_path,'model.ckpt-'+str(o_itr)))
        best_valid_dice = float(best_dice)
    else:
        best_valid_dice = 0

    o_itr += 1

    ##################### other paras initialization #########################
    train_loss_sum = 0
    train_acc_sum = 0
    train_Dice_sum = 0
    move_count = 0
    ##########################################################################

    for itr in xrange(o_itr,opt.max_iteration):
        train_images,train_annotations = train_dataset_reader.read_batch_normal_train()
        feed_dict_train = {x: train_images, y: train_annotations}
        sess.run(train_op, feed_dict=feed_dict_train)

        train_loss = sess.run(loss, feed_dict = feed_dict_train)
        train_loss_sum += train_loss
        train_pred = sess.run([pred_], feed_dict = {x: train_images})
        train_pred = np.squeeze(train_pred[0],axis=1)
        train_annotations = np.squeeze(train_annotations,axis=1)
        train_annotations = np.squeeze(train_annotations,axis=-1)
        for i in range(opt.batch_size):
            train_Dice_sum += utils.cal_Dice(train_pred[i],train_annotations[i])
            train_acc_sum += utils.cal_acc(train_pred[i],train_annotations[i])

        if itr % opt.print_info_freq == 0:
            print("Step:{}, Train_loss:{}, Train_acc:{}, Train_Dice:{}".format(itr,round(train_loss_sum/opt.print_info_freq,4),   \
                                                                               round(train_acc_sum/opt.print_info_freq/opt.batch_size,4),   \
                                                                               train_Dice_sum/opt.print_info_freq/opt.batch_size))
            train_loss_sum = 0
            train_acc_sum = 0
            train_Dice_sum = 0
            ############################  validation  ################################
            val_loss = 0
            val_acc = 0
            val_Dice_sum = 0
            block_nums = (DATA_SIZE[1] // BLOCK_SIZE[1]) * (DATA_SIZE[2] // BLOCK_SIZE[2])
            val_num = block_nums * val_data_num
            I = 0
            U = 0
            count = 0
            temp_index = val_num//opt.batch_size

            for kk in range(temp_index):
                valid_images, valid_annotations = validation_dataset_reader.read_batch_normal_valid_all(opt.batch_size)
                feed_dict_valid = {x: valid_images, y: valid_annotations}
                valid_loss = sess.run(loss, feed_dict = feed_dict_valid)
                val_loss += valid_loss

                valid_pred = sess.run([pred_], feed_dict = {x: valid_images})
                valid_pred = np.squeeze(valid_pred[0],axis=1)
                valid_annotations = np.squeeze(valid_annotations,axis=1)
                valid_annotations = np.squeeze(valid_annotations,axis=-1)
                for i in range(opt.batch_size):
                    count += 1
                    t1, t2 = utils.cal_Dice_para(valid_pred[i],valid_annotations[i])
                    I += t1
                    U += t2
                    if count % block_nums == 0:
                        val_Dice_sum += 2*I/(I+U+1e-5)
                        I = 0
                        U = 0

                    val_acc += utils.cal_acc(valid_pred[i],valid_annotations[i])


            if val_num > temp_index * opt.batch_size:
                valid_images, valid_annotations = validation_dataset_reader.read_batch_normal_valid_all(val_num - temp_index * opt.batch_size)
                feed_dict_valid = {x: valid_images, y: valid_annotations}
                valid_loss = sess.run(loss, feed_dict = feed_dict_valid)
                val_loss += valid_loss

                valid_pred = sess.run([pred_], feed_dict = {x: valid_images})
                valid_pred = np.squeeze(valid_pred[0],axis=1)
                valid_annotations = np.squeeze(valid_annotations,axis=1)
                valid_annotations = np.squeeze(valid_annotations,axis=-1)
                temp_index += 1


            for i in range(val_num - temp_index * opt.batch_size):
                t1, t2 = utils.cal_Dice_para(valid_pred[i],valid_annotations[i])
                I += t1
                U += t2

                val_acc += utils.cal_acc(valid_pred[i],valid_annotations[i])

            val_Dice_sum += 2*I/(I+U+1e-5)

            print("Step:{}, Valid_loss:{}, Valid_acc:{}, Valid_Dice:{}".format(itr,round(val_loss/temp_index,4),  \
                                                                               round(val_acc/val_num,4),val_Dice_sum/val_data_num))

            ##########################  save (best) model  ###################################
            saver.save(sess, os.path.join(model_save_path,"model.ckpt"), itr)
            m_Dice = val_Dice_sum/val_data_num

            if m_Dice > best_valid_dice:
                temp = '{:.5f}'.format(m_Dice)
                os.mkdir(os.path.join(best_model_save_path,temp))
                for suffix in copy_file_suffixs:
                    temp2 = "model.ckpt-{}".format(itr)+suffix
                    shutil.copy(os.path.join(model_save_path,temp2),os.path.join(best_model_save_path,temp,temp2))
                shutil.copy(os.path.join(model_save_path,'checkpoint'),os.path.join(best_model_save_path,temp,'checkpoint'))

                model_names = natsort.natsorted(os.listdir(best_model_save_path))
                #print(len(model_names))
                if len(model_names) == 4:
                    shutil.rmtree(os.path.join(best_model_save_path,model_names[0]))
                best_valid_dice = m_Dice


if __name__ == "__main__":
    tf.app.run()
