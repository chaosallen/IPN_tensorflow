"""
Create A Data Dictionary
"""

import os
import natsort

def read_dataset(data_dir):
    datasetlist={'train':{},'val':{}}
    datalist=list(datasetlist.keys())
    for dlist in datalist:#train/validation
        modalitylist=os.listdir(os.path.join(data_dir, dlist))
        modalitylist=natsort.natsorted(modalitylist)
        for modal in modalitylist:#model1/model2/labels
            datasetlist[dlist].update({modal:{}})
            if modal!='label':
                ctlist=os.listdir(os.path.join(data_dir, dlist, modal))
                ctlist=natsort.natsorted(ctlist)
                for ct in ctlist:#ct1/ct2/ct3
                    datasetlist[dlist][modal].update({ct:{}})
                    scanlist=os.listdir(os.path.join(data_dir, dlist, modal,ct))
                    scanlist=natsort.natsorted(scanlist)
                    for i in range(0,len(scanlist)):#1.bmp/2.bmp/.../n.bmp
                        scanlist[i]=os.path.join(data_dir, dlist, modal,ct,scanlist[i])
                    datasetlist[dlist][modal][ct]=scanlist
            else:
                ctlist=os.listdir(os.path.join(data_dir, dlist, modal))
                ctlist=natsort.natsorted(ctlist)
                for ct in ctlist:
                    datasetlist[dlist][modal].update({ct: {}})
                    labeladdress = os.path.join(data_dir, dlist, modal, ct)
                    datasetlist[dlist][modal][ct] = labeladdress
    train_records = datasetlist['train']
    validation_records = datasetlist['val']
    return train_records, validation_records
