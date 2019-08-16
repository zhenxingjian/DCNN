import numpy as np
import nibabel as nib
# import tensorflow as tf
import pdb
import dipy

import csv
import os
import time

import pandas as pd

from keras.preprocessing.sequence import pad_sequences
from dipy.reconst.shore import ShoreModel
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere

def vec2mtx(spd_data):
    '''
    input the SPD image, which is N x M x P x 1 x 6
    output is the SPD_matrix image which is N x M x P x 3 x 3
    '''
    img_shape = spd_data.shape
    N = img_shape[0]
    M = img_shape[1]
    P = img_shape[2]
    spd_data = np.reshape(spd_data,[N*M*P,6])
    mtx = np.zeros([N*M*P,3,3],dtype = np.float32)
    mtx[:,1,0] = spd_data[:,1]
    mtx[:,2,0] = spd_data[:,3]
    mtx[:,2,1] = spd_data[:,4]
    mtx = mtx + np.transpose(mtx,[0,2,1])
    mtx[:,0,0] = spd_data[:,0]
    mtx[:,1,1] = spd_data[:,2]
    mtx[:,2,2] = spd_data[:,5]
    mtx = np.reshape(mtx,[N,M,P,3,3])
    return mtx

def read_fiber(path):
    # print path
    track_img = nib.load(path)
    track_data = track_img.get_data()
    Pos = []
    length = int(np.max(track_data))
    flag = 0
    for i in range (1,length+1):
        temp_pos = np.where(track_data == i)
        # print temp_pos
        # print flag
        # pdb.set_trace()
        if temp_pos[0].size == 0:
            flag = flag + 1
        elif temp_pos[0].size > 1 and flag:
            temp_flag = flag
            for i in range(min(flag+1 , temp_pos[0].size)):
                temp_flag = temp_flag - 1
                Pos.append([temp_pos[0][i],temp_pos[1][i],temp_pos[2][i]])
            flag = temp_flag + 1
        else:
            Pos.append([temp_pos[0][0],temp_pos[1][0],temp_pos[2][0]])

    Pos2 = []
    flag2 = 0
    for i in range (length,0,-1):
        temp_pos = np.where(track_data == i)
        # print temp_pos
        # print flag2
        # pdb.set_trace()
        if temp_pos[0].size == 0:
            flag2 = flag2 + 1
        elif temp_pos[0].size > 1 and flag2:
            temp_flag = flag2
            for i in range(min(flag2+1 , temp_pos[0].size)):
                temp_flag = temp_flag - 1
                Pos2.append([temp_pos[0][i],temp_pos[1][i],temp_pos[2][i]])
            flag2 = temp_flag + 1
        else:
            Pos2.append([temp_pos[0][0],temp_pos[1][0],temp_pos[2][0]])
    Pos2.reverse()
    # pdb.set_trace()
    # print Pos
    # print Pos2 
    if len(Pos) < len(Pos2):
        Pos = Pos2
    print len(Pos)
    Pos = np.asarray(Pos,dtype = np.int32)
    # pdb.set_trace()
    return Pos

def prepare_data(csv_file,label_name): 

    Names = []
    labels = []
    Visit_ = []

    # csv_file = 'data/ad_data/adrc-subject-list.csv'
    with open(csv_file, 'rb') as f:
        reader = csv.reader(f)
        count = 0

        for row in reader:
            if count == 0:
                # label_idx = row.index(label_name)  # normal should use this, APOE is different
                label_idx1 = 3
                label_idx2 = 4
                count = count + 1
                continue
            else:
                Names.append(row[0])
                labels.append(row[label_idx1] == '4' or row[label_idx2] == '4')
                # labels.append(1*(row[2]=="Male"))
                count = count + 1
    # Labels = np.zeros([len(labels) , 2])
    # for labelid in range(len(labels)):
    #     Labels[ labelid , labels[labelid] ] = 1
    # pdb.set_trace()

    spd_data_folder = "data/ad_data/data/"
    spd_data_name = "/cor_DTI_SPD.nii"
    dMRI_data_name = "/dMRI.nii.gz"

    track_names = ["fmajor_PP.avg33_mni_bbr_track_image.nii.gz",
                   "fminor_PP.avg33_mni_bbr_track_image.nii.gz",
                   "lh.atr_PP.avg33_mni_bbr_track_image.nii.gz",
                   "lh.cab_PP.avg33_mni_bbr_track_image.nii.gz",
                   "lh.ccg_PP.avg33_mni_bbr_track_image.nii.gz",
                   "lh.cst_AS.avg33_mni_bbr_track_image.nii.gz",
                   "lh.ilf_AS.avg33_mni_bbr_track_image.nii.gz",
                   "lh.slfp_PP.avg33_mni_bbr_track_image.nii.gz",
                   "lh.slft_PP.avg33_mni_bbr_track_image.nii.gz",
                   "lh.unc_AS.avg33_mni_bbr_track_image.nii.gz",
                   "rh.atr_PP.avg33_mni_bbr_track_image.nii.gz",
                   "rh.cab_PP.avg33_mni_bbr_track_image.nii.gz",
                   "rh.ccg_PP.avg33_mni_bbr_track_image.nii.gz",
                   "rh.cst_AS.avg33_mni_bbr_track_image.nii.gz",
                   "rh.ilf_AS.avg33_mni_bbr_track_image.nii.gz",
                   "rh.slfp_PP.avg33_mni_bbr_track_image.nii.gz",
                   "rh.slft_PP.avg33_mni_bbr_track_image.nii.gz",
                   "rh.unc_AS.avg33_mni_bbr_track_image.nii.gz",
                   ]
    

    Traces = [[] for _ in range(len(track_names))]
    Traces_ODF = [[] for _ in range(len(track_names))]
    for pid in range(len(Names)):
        name = Names[pid]
        for vi in ["_v2","_v3","_v1","_v4"]:
            spd_path = spd_data_folder + name + vi + spd_data_name
            if os.path.isfile(spd_path):
                break
        if not os.path.isfile(spd_path):
            print("There's something wrong in this dataset, the data is not v1/v2/v3.")
            exit()
        Visit_.append(vi)
        spd_img = nib.load(spd_path)
        spd_data = spd_img.get_data()
        mtx_whole = vec2mtx(spd_data)

        dMRI_path = spd_data_folder + name + vi + dMRI_data_name
        dMRI_img = nib.load(dMRI_path)
        dMRI_data = dMRI_img.get_data()

        ############### ODF
        radial_order = 6
        zeta = 700
        lambdaN = 1e-8
        lambdaL = 1e-8
        bval = "./data/ad_data/Bval_vec/dti_bvals.bval"
        bvec = "./data/ad_data/Bval_vec/" + name + vi + "/bvecs_eddy.rotated.bvecs"
        gtab = gtab = gradient_table(bvals = bval, bvecs = bvec)
        asm = ShoreModel(gtab, radial_order=radial_order,
                         zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL)

        ############### ODF


        for track_id in range(len(track_names)):
            track_sub_name = track_names[track_id]
            # for vi in ["_v2","_v3","_v1","_v4"]:  ########## \tab for the rest 3 lines
            track_path = spd_data_folder + name + vi + "/" + track_sub_name
            # if os.path.isfile(track_path):
            #     break
            if os.path.isfile(track_path):
                Pos = read_fiber(track_path)
                # pdb.set_trace()
                minx = min(Pos[:,0])
                maxx = max(Pos[:,0])
                miny = min(Pos[:,1])
                maxy = max(Pos[:,1])
                minz = min(Pos[:,2])
                maxz = max(Pos[:,2])
                # pdb.set_trace()
                asmfit = asm.fit(dMRI_data[minx:maxx+1,miny:maxy+1,minz:maxz+1])
                # pdb.set_trace()
                sphere = get_sphere('symmetric724')
                dMRI_odf = asmfit.odf(sphere)
                # pdb.set_trace()
                Traces_ODF[track_id].append( dMRI_odf[(Pos[:,0]-minx,Pos[:,1]-miny,Pos[:,2]-minz)] )
                Traces[track_id].append( mtx_whole[(Pos[:,0],Pos[:,1],Pos[:,2])] )
            else:
                Traces[track_id].append( [] )
    Traces = np.asarray(Traces)
    # Traces_ODF = None
    Traces_ODF = np.asarray(Traces_ODF)
    Labels = np.asarray(labels)
    # pdb.set_trace()
    return Traces,Traces_ODF,Labels,track_names, Names , Visit_

def read_data(csv_file,label_name,random_seed = 20160924, recalculate = False ):
    # pdb.set_trace()
    if not os.path.isdir("./data/ad_data/processed_data/"):
        os.mkdir("./data/ad_data/processed_data/")
    if not os.path.isdir("./data/ad_data/processed_data/"+label_name):
        os.mkdir("./data/ad_data/processed_data/"+label_name)

    np.random.seed(random_seed)
    if os.path.isfile("./data/ad_data/processed_data/"+label_name+"/spddata.npy") and not recalculate:
        spddata = None#np.load("./data/ad_data/processed_data/"+label_name+"/spddata.npy")
        odfdata = None#np.load("./data/ad_data/processed_data/"+label_name+"/odfdata.npy")
        label = np.load("./data/ad_data/processed_data/"+label_name+"/label.npy")
        return spddata, odfdata , label
    spddata, odfdata , label , track_sub_names , Names ,Visit_ = prepare_data(csv_file,label_name)

    # print (data.shape)
    # print (label.shape)

    Null_pos = []
    maxlength = np.zeros(spddata.shape[0],dtype = np.int32)
    minlength = np.ones(spddata.shape[0],dtype = np.int32)*np.inf
    for personi in range(spddata.shape[1]):
        for tracei in range(spddata.shape[0]):
            temp_data = spddata[tracei,personi]
            # pdb.set_trace()
            if temp_data == []:
                Null_pos.append(personi)
                break
            else:
                if maxlength[tracei] < temp_data.shape[0]:
                    maxlength[tracei] = temp_data.shape[0]
                if minlength[tracei] > temp_data.shape[0]:
                    minlength[tracei] = temp_data.shape[0]

    minlength = minlength.astype ( np.int32 )

    # pdb.set_trace()

    for del_pos in reversed(Null_pos):
        spddata = np.delete(spddata,del_pos,1)
        odfdata = np.delete(odfdata,del_pos,1)
        label = np.delete(label,del_pos,0)
        Names.pop(del_pos)
        Visit_.pop(del_pos)

    num_people = spddata.shape[1]
    people_shuffle = range(num_people)
    np.random.shuffle(people_shuffle)
    np.save("./data/ad_data/processed_data/"+label_name+"/people_shuffle.npy",people_shuffle)
    # print (people_shuffle)
    # pdb.set_trace()
    spddata = spddata[:,people_shuffle]
    odfdata = odfdata[:,people_shuffle]
    label = label[people_shuffle]

    newspddata = [[] for _ in range(spddata.shape[0])]
    newodfdata = [[] for _ in range(odfdata.shape[0])]

    for tracei in range(spddata.shape[0]):
        for personi in range(spddata.shape[1]):
            # print personi
            temp_data = spddata[tracei,personi]

            newspddata[tracei].append(  pad_sequences([temp_data], maxlen=minlength[tracei], truncating='post', dtype='float32')[0] )

            temp_data = odfdata[tracei,personi]

            newodfdata[tracei].append(  pad_sequences([temp_data], maxlen=minlength[tracei], truncating='post', dtype='float32')[0] )

    spddata = np.asarray(newspddata)
    odfdata = np.asarray(newodfdata)
    # pdb.set_trace() 

    for tracei in range(spddata.shape[0]):
        subdata = np.asarray(newspddata[tracei])
        np.save("./data/ad_data/processed_data/"+label_name+"/spdTrack" + str(tracei) + ".npy" , subdata)
        subdata = np.asarray(newodfdata[tracei])
        np.save("./data/ad_data/processed_data/"+label_name+"/odfTrack" + str(tracei) + ".npy" , subdata)

        # print (subdata.shape)
    fileObject = open("./data/ad_data/processed_data/"+label_name+"/Track_info.txt", 'w')
    fileObject.write("Track_names maxlength minlength")
    fileObject.write('\n')
    for tracei in range(spddata.shape[0]):
        track_sub_name = track_sub_names[tracei]
        fileObject.write(track_sub_name+" ")
        fileObject.write(str(maxlength[tracei])+" "+str(minlength[tracei]))
        fileObject.write('\n')
    fileObject.close()

    # print (data.shape)
    # print (label.shape)

    # pdb.set_trace()

    np.save("./data/ad_data/processed_data/"+label_name+"/spddata.npy",spddata)
    np.save("./data/ad_data/processed_data/"+label_name+"/odfdata.npy",odfdata)
    # odfdata = None
    np.save("./data/ad_data/processed_data/"+label_name+"/label.npy",label)
    return spddata,odfdata,label

def load_length(path):
    Length = []
    Name = []
    if not os.path.isfile(path):
        print ("Run readdata first")
        return None
    with open(path, 'r') as f:
        data = f.readlines()
        for line in data:
            if "#" in line or "maxlength" in line:
                continue
            else:
                pos = line.split()[1:3]
                name = line.split()[0]
                Length.append(pos)
                Name.append(name)
    Length = np.asarray(Length,dtype = np.int32)
    return Length, Name



if __name__ == '__main__':
    # tic = time.time()
    # prepare_data("data/ad_data/ADRC_CSF_biomarker_cutoffs_usabel.csv","lp1_csf_biomarker_group")
    # print time.time() - tic
    # pdb.set_trace()
    # read_fiber('./data/ad_data/data/adrc00074_v2/fmajor_PP.avg33_mni_bbr_track_image.nii.gz')
    tic = time.time()
    # read_data("data/ad_data/ADRC_CSF_biomarker_cutoffs_test.csv","lp1_csf_biomarker_group",recalculate = True)
    read_data("data/ad_data/APOE.csv","APOE",recalculate = True)
    # read_data("data/ad_data/ADRC_CSF_data_batch1frombatch2.csv","pv_ttau_by_ab",recalculate = True)
    # read_data("data/ad_data/ADRC_NFL.csv","NFL",recalculate = True)

    print time.time() - tic
    pdb.set_trace()
    # print (label.shape)
    # print (data.shape)















