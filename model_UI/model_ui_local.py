from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
from tkinter import *

import os
import numpy as np
    
import matplotlib.pyplot as plt
import pickle
import sys
import pandas as pd
import random
import glob
import nibabel as nib
from keras.layers import GlobalAveragePooling3D
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Concatenate, BatchNormalization, Activation, Conv3D, UpSampling3D,MaxPooling3D, GlobalAveragePooling3D, Dense, Dropout, Flatten
from keras import optimizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras import regularizers
from keras import losses
from keras import backend as K
from keras.callbacks import History
from keras.layers.merge import Concatenate
import keras
import keras_metrics as km
from keras.models import load_model
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config = config))
from keras.utils.generic_utils import get_custom_objects
from keras.models import load_model
import h5py as hp



def nii_file():
    nii_file = filedialog.askdirectory(parent=screen, title='Nii File Location ( 「choose this file」 / 「PTID」 / * / 「FOLDER」 / 「EXAMDATE」 / * / *.nii )')
    if not nii_file:
            messagebox.showinfo("Mistake",'No Choose Nii File')  
    else:
        xlsx_path = filedialog.askopenfilename(parent=screen, title='xlsx file')   # 選擇檔案後回傳檔案路徑與名稱
        if not xlsx_path:
            messagebox.showinfo("Mistake",'No Choose xlsx File')
        else:
            nii_file_name = os.path.split(nii_file)[1]
            globname_GM = glob.glob(nii_file +'/*_S_*/*_*_*/*/*/*/smwmGM.nii')
            messagebox.showinfo("Success",'Already Start Saving...' )

            nii_path_list = [['PTID','TIME','FOLDER']]
            PTID_list = []
            TIME_list = []

            globlist_GM = []

            for name_GM in globname_GM:
                name_GM = name_GM.replace("\\","/")
                list_GM = [name_GM]
                globlist_GM.extend(list_GM)

            img_new = np.zeros([1,145,121,121,3], dtype='float32')  #建立5維

            #### GM
            for saveimg_GM in globlist_GM:
                saveimg_GM_path = os.path.split(saveimg_GM)[0]
                nii_path_GM = saveimg_GM_path +'/'
                img_GM_name  = os.path.split(saveimg_GM)[1]  #儲存檔案的檔名
                img_GM_name = os.path.join('.',saveimg_GM_path , img_GM_name) #('.')用於自動增加 / 或 \ 

                img_GM = nib.load(img_GM_name).get_fdata() #载入
                img_GM = np.array(img_GM)
                img_GM = img_GM.astype('float32')   #存浮點
                img_GM = np.swapaxes(img_GM, 0, 1)
                img_GM = np.flip(img_GM, 0)
                img_GM = np.squeeze(img_GM)    #刪除shape為1的維度
                img_GM = np.expand_dims(img_GM,axis=0)
                img_GM = np.expand_dims(img_GM,axis=-1)

                nii_path_GM = nii_path_GM.replace("\\","/")
                nii_split = nii_path_GM.split("/")   
            #######################################################    
                img_WM_name = img_GM_name.replace("GM" , "WM")
                img_WM = nib.load(img_WM_name).get_fdata() #载入
                img_WM = np.array(img_WM)
                img_WM = img_WM.astype('float32')   #存浮點
                img_WM = np.swapaxes(img_WM, 0, 1)
                img_WM = np.flip(img_WM, 0)
                img_WM = np.squeeze(img_WM)    #刪除shape為1的維度
                img_WM = np.expand_dims(img_WM,axis=0)
                img_WM = np.expand_dims(img_WM,axis=-1)
                img_WM_new = np.concatenate([img_GM,img_WM],axis=0)
            #######################################################
                img_CSF_name = img_GM_name.replace("GM","CSF")
                img_CSF = nib.load(img_CSF_name).get_fdata() #载入
                img_CSF = np.array(img_CSF)
                img_CSF = img_CSF.astype('float32')   #存浮點
                img_CSF = np.swapaxes(img_CSF, 0, 1)
                img_CSF = np.flip(img_CSF, 0)
                img_CSF = np.squeeze(img_CSF)    #刪除shape為1的維度
                img_CSF = np.expand_dims(img_CSF,axis=0)
                img_CSF = np.expand_dims(img_CSF,axis=-1)
                img_CSF_new = np.concatenate([img_WM_new,img_CSF],axis=0)
                img_CSF_new = np.swapaxes(img_CSF_new,0,-1)
                img_new = np.concatenate([img_new,img_CSF_new],axis=0)
            #######################################################
                # PTID = nii_split[6]   # O
                # TIME = nii_split[9]  # O
                # DX = nii_split[8]   # O

                PTID = nii_split[-6]
                TIME = nii_split[-3]
                DX = nii_split[-4] 

                TIME_split = TIME.split('_')
                TIME_SPLIT = TIME_split[0]    
                nii_path_list.append([PTID,TIME_SPLIT,DX])
                #測試將受測者編號做成list 以日後進行搜尋
                PTID_list.append(PTID)
                TIME_list.append(TIME_SPLIT)

            NII_PATH_LIST = pd.DataFrame(nii_path_list)

            NII_PATH_LIST.rename(columns = {0:'PTID',1:'EXAMDATE',2:'FOLDER'},inplace = True)
            NII_dataframe = NII_PATH_LIST.drop(NII_PATH_LIST.index[[0]])
            NII_dataframe = NII_dataframe.drop_duplicates(subset = ['PTID','EXAMDATE','FOLDER'])
            list_ID = list(NII_dataframe['PTID'])
            list_TIME = list(NII_dataframe['EXAMDATE'])
            list_KEYWORD = list(NII_dataframe['FOLDER'])

            ###########################################################################################
            df = pd.read_excel(str(xlsx_path),usecols="A,B,C,D,E,F,G,H,I,J,K" )
            final_df_ID = pd.DataFrame()

            for t,w,r in zip(list_ID, list_TIME, list_KEYWORD):
                filt_ID = (df["PTID"] == t) & (df["EXAMDATE"] == w) & (df["FOLDER"] == r)

                if final_df_ID.empty:
                    final_df_ID = df.loc[filt_ID]    #loc這裡為插入一行的作用
                else:
                    final_df_ID = pd.concat([final_df_ID, df.loc[filt_ID]])

            ###################  儲存pickle檔案  ######################################################
            final_df_ID.rename(columns = {0:'PTID',1:'EXAMDATE',2:'VISCODE',3:'FOLDER',4:'TRUE_AGE',5:'PTGENDER',
                                        6:'APOE4',7:'CDRSB',8:'CDRGL',9:'DX',10:'Race'},inplace = True)

            with open(nii_file + '/' + nii_file_name +'.pickle', 'wb') as f:
                pickle.dump(final_df_ID, f)
            ###########################################################################################
            np.save(nii_file +'/' + nii_file_name +'.npy', img_new) #保存為.npy  使用時標註解開
            ##########################################################################################
            x = np.load(nii_file + '/' + nii_file_name +'.npy')
            x = x[1:,:,:,:]
            np.save(nii_file + '/' + nii_file_name +'.npy',x)
            messagebox.showinfo("Saving Successful", ('Pickle and Npy Saving at'+ nii_file ))
###########################################################################################
###########################################################################################
###########################################################################################
def show_value(event=None):
    var_class = var.get()
    if var_class == 'BrainAge':
        pickle_path = filedialog.askopenfilename(parent=screen, title='Pickle File')
        pickle_path_name = os.path.split(pickle_path)[0]
        if not pickle_path:
            messagebox.showinfo("Mistake", 'No Choose Pickle File')
        else:
            pickle_path = pickle_path.replace("\\","/")
            file_path = filedialog.askopenfilename(parent=screen, title='Npy File')# 選擇檔案後回傳檔案路徑與名稱
            if not file_path:
                messagebox.showinfo("Mistake", 'No Choose Npy File')
            else:
                file_path = file_path.replace("\\","/")

                with open(str(pickle_path), 'rb') as f:
                    pickle_file = pickle.load(f)
                pickle_file = pd.DataFrame(pickle_file)
                pickle_file['PTGENDER'].replace("Female", 0, inplace = True) # 將"Famale"換成數值0 
                pickle_file['PTGENDER'].replace("Male", 1, inplace = True) # 將"male"換成數值1
                pickle_file['PTGENDER'].replace(2, 0, inplace = True) # JANDI將 2 (Female)  換成數值0
                npy_input = np.load(str(file_path),allow_pickle=True)
                messagebox.showinfo("Loading Success",'Loading Success Please Click "OK" and Waiting the Prediction Result')
                ###############################    model    ####################
                ############################### 去除race 4  ####################
                pickup_4_val = [i for i,j in enumerate(pickle_file.loc[:]['Race']) if j == 4]
                npy_input = np.delete(npy_input , pickup_4_val ,0)
                pickle_file = pickle_file.drop(pickle_file.index[pickup_4_val])
                #########################  去除race 5  ##############################
                pickup_5_val = [i for i,j in enumerate(pickle_file.loc[:]['Race']) if j == 5]
                npy_input = np.delete(npy_input , pickup_5_val ,0)
                pickle_file = pickle_file.drop(pickle_file.index[pickup_5_val])
                #########################  去除race 6  ##############################
                pickup_6_val = [i for i,j in enumerate(pickle_file.loc[:]['Race']) if j == 6]
                npy_input = np.delete(npy_input , pickup_6_val ,0)
                pickle_file = pickle_file.drop(pickle_file.index[pickup_6_val])
                ####################################################################
                ttt = np.zeros([1,11],dtype="float32")
                fir = np.concatenate((ttt,pickle_file),axis = 0)
                fir = fir[1:,:]

                race_onehot =[]
                for ii in fir:
                    if ii[10] == 1:
                        race_onehot.append([1,0,0])
                    elif ii[10] == 2:
                        race_onehot.append([0,1,0])
                    elif ii[10] == 3:
                        race_onehot.append([0,0,1])
                fir = np.concatenate((fir,race_onehot),axis = 1)

                pred_pickle_input = []    # age PTGENDER 
                fit_y = []
                #選取pickle檔中需要的資料
                for j in fir:
                    c = [j[5],j[11],j[12],j[13]] # PTGENDER
                    pred_pickle_input.append(c)
                pred_pickle_input = np.asarray(pred_pickle_input).astype('float32')    
                for y in fir:
                    r = y[4]    # age
                    fit_y.append(r)
                fit_y = np.asarray(fit_y).astype('float32')
                fit_y = (fit_y-50)/10

                pred_npy_input = np.zeros([1,145,121,121,3],dtype="float32")
                pred_npy_input = np.concatenate((pred_npy_input,npy_input) ,axis = 0)
                pred_npy_input = pred_npy_input[1:,:,:,:]
                pred_npy_input = np.concatenate((np.expand_dims(pred_npy_input[:,:,:,:,0],axis=-1),np.expand_dims(pred_npy_input[:,:,:,:,2],axis=-1)),axis=-1)

                pred_pickle_input =np.array(pred_pickle_input)

                input_pickle = Input(shape=(pred_pickle_input.shape[1],))
                input_img = Input(shape=(pred_npy_input.shape[1], pred_npy_input.shape[2],pred_npy_input.shape[3],pred_npy_input.shape[4]))

                class Swish(Activation):
                    def __init__(self, activation, **kwargs):
                        super(Swish, self).__init__(activation, **kwargs)
                        self.__name__ = 'swish'
                def swish(x):
                    beta = 0.5
                    return (K.sigmoid(beta*x) * x)
                
                get_custom_objects().update({'swish': Swish(swish)})
                customObjects = {'Swish':Swish,'swish':swish}

                model = load_model('AF_JADNI_AFdata_2conv5_16to256_cbam_4d3d_n96-64_swish05_lr-5_l2005_b16_swAGExRACE-00100-0.35738-32.07687.h5', custom_objects=customObjects)
                pre_class = model.predict([pred_npy_input, pred_pickle_input] , batch_size = 8)


                df = pd.DataFrame(pre_class)
                df = df*10 + 50

                df_da = pd.DataFrame(np.concatenate((pd.DataFrame(fir) , pd.DataFrame(df)),axis =1))

                df_da.rename(columns = {0:'PTID',1:'EXAMDATE',2:'VISCODE',3:'FOLDER',4:'TRUE_AGE',
                                        5:'PTGENDER',6:'APOE4',7:'CDRSB',8:'CDRGL',9:'DX',
                                        10:'Race',11:'race1',12:'race2',13:'race3',14:'predict_age'},inplace = True)

                df_da.to_csv(pickle_path_name + '/' +'BrainAge.csv',index = False)
                messagebox.showinfo("Prediction Result", "Predict Success")

    elif var_class == 'NIFD_2ndclass':
        pickle_path = filedialog.askopenfilename(parent=screen, title='Pickle File')
        pickle_path_name = os.path.split(pickle_path)[0]
        if not pickle_path:
            messagebox.showinfo("Mistake", 'No Choose Pickle File')
        else:
            pickle_path = pickle_path.replace("\\","/")
            file_path = filedialog.askopenfilename(parent=screen, title='Npy File')# 選擇檔案後回傳檔案路徑與名稱
            if not file_path:
                messagebox.showinfo("Mistake", 'No Choose Npy File')
            else:
                file_path = file_path.replace("\\","/")

                with open(str(pickle_path), 'rb') as f:
                    pickle_file = pickle.load(f)
                pickle_file = pd.DataFrame(pickle_file)
                pickle_file['PTGENDER'].replace("Female", 0, inplace = True) # 將"Famale"換成數值0 
                pickle_file['PTGENDER'].replace("Male", 1, inplace = True) # 將"male"換成數值1
                pickle_file['PTGENDER'].replace(2, 0, inplace = True) # JANDI將 2 (Female)  換成數值0
                messagebox.showinfo("Loading Success",'Loading Success Please Click "OK" and Waiting the Prediction Result')
                ######################## model ##########################
                ttt = np.zeros([1,11],dtype="float32")
                fir = np.concatenate((ttt,pickle_file),axis = 0)
                fir = fir[1:,:]
                pred_pickle = []

                for i in fir:
                    j = i[4:6]
                    pred_pickle.append(j)

                pred_pickle = np.asarray(pred_pickle).astype('float32')# X 與 z 要為相同的資料型態
                #正規化
                ss_val = ((pred_pickle[:,0]-50)/10) 
                ee_val = pred_pickle[:,1]
                pred_pickle_input = [ss_val,ee_val]
                pred_pickle_input = np.asarray(pred_pickle_input).astype('float32')
                pred_pickle_input = pred_pickle_input.T

                fir = pd.DataFrame(fir)
                fir.rename(columns = {9:'DX'}, inplace = True)
                fit_y = []
                for i in range(len(fir['DX'])):
                    if fir['DX'][i]=='BV':
                        fit_y.append([1,0,0,0])
                    elif fir['DX'][i]=='SV':
                        fit_y.append([0,1,0,0])
                    elif fir['DX'][i]=='PNFA':
                        fit_y.append([0,0,1,0])
                    elif fir['DX'][i]=='L_SD':
                        fit_y.append([0,0,0,1])


                npy_input = np.load(str(file_path),allow_pickle=True)
                pred_npy_input = np.zeros([1,145,121,121,3],dtype="float32")

                pred_npy_input = np.concatenate((pred_npy_input,npy_input) ,axis = 0)
                pred_npy_input = pred_npy_input[1:,:,:,:]
                pred_npy_input = np.concatenate((np.expand_dims(pred_npy_input[:,:,:,:,0],axis=-1),np.expand_dims(pred_npy_input[:,:,:,:,2],axis=-1)),axis=-1)

                # fit_y_input = np.asarray(fit_y_input)
                pred_pickle_input =np.array(pred_pickle_input)

                input_pickle = Input(shape=(pred_pickle_input.shape[1],))
                input_img = Input(shape=(pred_npy_input.shape[1], pred_npy_input.shape[2],pred_npy_input.shape[3],pred_npy_input.shape[4]))

                def dice_coef_smooth1(y_true, y_pred, smooth=1):
                    intersection = K.sum(y_true * y_pred)
                    return (2. * intersection + smooth) / (K.sum(y_true * y_true) + K.sum(y_pred * y_pred) + smooth)

                def dice_coef(y_true, y_pred, smooth=1e-5):
                    intersection = K.sum(y_true * y_pred)
                    return (2. * intersection + smooth) / (K.sum(y_true * y_true) + K.sum(y_pred * y_pred) + smooth)

                def dice_coef_loss(y_true, y_pred):
                    return -K.log(dice_coef(y_true, y_pred))

                metrics = [km.f1_score(),
                           km.precision(),
                           km.recall(),
                           'accuracy',
                           dice_coef]


                customObjects = {'dice_coef':dice_coef,'dice_coef_smooth1':dice_coef_smooth1,'dice_coef_loss':dice_coef_loss,
                                 'binary_f1_score':km.f1_score(),'binary_precision':km.precision(),'binary_recall':km.recall()}

                model = load_model('N4class_conv5_16_4d16_flatten_mix_p5_cbam135-0125_dice_lr-3_l2005-00020-0.88800.h5', custom_objects=customObjects)

                pre_class = model.predict([pred_npy_input, pred_pickle_input] , batch_size = 8)


                df = pd.DataFrame(pre_class)
                df_da = []
                for j in range(len(df)):
                    if (df.loc[j][0] > df.loc[j][1]) & (df.loc[j][0] > df.loc[j][2])& (df.loc[j][0] > df.loc[j][3]):
                        df_da.append('BV')
                    elif (df.loc[j][1] > df.loc[j][0]) & (df.loc[j][1] > df.loc[j][2])& (df.loc[j][1] > df.loc[j][3]):
                        df_da.append('SV')
                    elif (df.loc[j][2] > df.loc[j][0]) & (df.loc[j][2] > df.loc[j][1])& (df.loc[j][2] > df.loc[j][3]):
                        df_da.append('PNFA')
                    elif (df.loc[j][3] > df.loc[j][0]) & (df.loc[j][3] > df.loc[j][1])& (df.loc[j][3] > df.loc[j][2]):
                        df_da.append('L_SD')

                df_da = pd.DataFrame(df_da)
                df_da = pd.DataFrame(np.concatenate((pd.DataFrame(pickle_file),pd.DataFrame(fit_y),pd.DataFrame(pre_class) , df_da),axis =1))

                df_da.rename(columns = {0:'PTID',1:'EXAMDATE',2:'VISCODE',3:'FOLDER',4:'TRUE_AGE',
                                        5:'PTGENDER',6:'APOE4',7:'CDRSB',8:'CDRGL',9:'DX',
                                        10:'Race',11:'true1',12:'true2',13:'true3',14:'true4',
                                        15:'pred1',16:'pred2',17:'_pred3',18:'pred4',
                                        19:'NIFDclass_predict'},inplace = True)

                df_da.to_csv(pickle_path_name + '/' +'NIFD_2ndclass.csv',index = False)
                messagebox.showinfo("Prediction Result", "Predict Success")
    ####################################################################
    ####################################################################
    elif var_class == 'ADNI&NIFD_classification':
        pickle_path = filedialog.askopenfilename(parent=screen, title='Pickle File')
        pickle_path_name = os.path.split(pickle_path)[0]
        if not pickle_path:
            messagebox.showinfo("Mistake", 'No Choose Pickle File')
        else:
            pickle_path = pickle_path.replace("\\","/")
            file_path = filedialog.askopenfilename(parent=screen, title='Npy File')# 選擇檔案後回傳檔案路徑與名稱
            if not file_path:
                messagebox.showinfo("Mistake", 'No Choose Npy File')
            else:
                file_path = file_path.replace("\\","/")

                with open(str(pickle_path), 'rb') as f:
                    pickle_file = pickle.load(f)
                pickle_file = pd.DataFrame(pickle_file)
                pickle_file['PTGENDER'].replace("Female", 0, inplace = True) # 將"Famale"換成數值0 
                pickle_file['PTGENDER'].replace("Male", 1, inplace = True) # 將"male"換成數值1
                pickle_file['PTGENDER'].replace(2, 0, inplace = True) # JANDI將 2 (Female)  換成數值0
                messagebox.showinfo("Loading Success",'Loading Success Please Click "OK" and Waiting the Prediction Result')

                ttt = np.zeros([1,11],dtype="float32")
                fir = np.concatenate((ttt,pickle_file),axis = 0)
                fir = fir[1:,:]
                pred_pickle = []

                for i in fir:
                    j = i[4:6]
                    pred_pickle.append(j)

                pred_pickle = np.asarray(pred_pickle).astype('float32')# X 與 z 要為相同的資料型態
                #正規化
                ss_val = ((pred_pickle[:,0]-50)/10) 
                ee_val = pred_pickle[:,1]
                pred_pickle_input = [ss_val,ee_val]
                pred_pickle_input = np.asarray(pred_pickle_input).astype('float32')
                pred_pickle_input = pred_pickle_input.T

                fir = pd.DataFrame(fir)
                fir.rename(columns = {9:'DX'}, inplace = True)
                fit_y = []
                for i in range(len(fir['DX'])):
                    if fir['DX'][i]=='Dementia':
                        fit_y.append([1,0])
                    elif fir['DX'][i]=='MCI':
                        fit_y.append([1,0])
                    elif fir['DX'][i]=='BV':
                        fit_y.append([0,1])
                    elif fir['DX'][i]=='SV':
                        fit_y.append([0,1])
                    elif fir['DX'][i]=='PNFA':
                        fit_y.append([0,1])
                    elif fir['DX'][i]=='L_SD':
                        fit_yl.append([0,1])

                fit_xxx = []
                for i in range(len(fir['DX'])):
                    if fir['DX'][i]=='Dementia':
                        fit_xxx.append('ADNI')
                    elif fir['DX'][i]=='MCI':
                        fit_xxx.append('ADNI')
                    elif fir['DX'][i]=='BV':
                        fit_xxx.append('NIFD')
                    elif fir['DX'][i]=='SV':
                        fit_xxx.append('NIFD')
                    elif fir['DX'][i]=='PNFA':
                        fit_xxx.append('NIFD')
                    elif fir['DX'][i]=='L_SD':
                        fit_xxx.append('NIFD') 

                npy_input = np.load(str(file_path),allow_pickle=True)
                pred_npy_input = np.zeros([1,145,121,121,3],dtype="float32")

                pred_npy_input = np.concatenate((pred_npy_input,npy_input) ,axis = 0)
                pred_npy_input = pred_npy_input[1:,:,:,:]
                pred_npy_input = np.concatenate((np.expand_dims(pred_npy_input[:,:,:,:,0],axis=-1),np.expand_dims(pred_npy_input[:,:,:,:,2],axis=-1)),axis=-1)

                # fit_y_input = np.asarray(fit_y_input)
                pred_pickle_input =np.array(pred_pickle_input)

                input_pickle = Input(shape=(pred_pickle_input.shape[1],))
                input_img = Input(shape=(pred_npy_input.shape[1], pred_npy_input.shape[2],pred_npy_input.shape[3],pred_npy_input.shape[4]))

                def dice_coef_smooth1(y_true, y_pred, smooth=1):
                    intersection = K.sum(y_true * y_pred)
                    return (2. * intersection + smooth) / (K.sum(y_true * y_true) + K.sum(y_pred * y_pred) + smooth)

                def dice_coef(y_true, y_pred, smooth=1e-5):
                    intersection = K.sum(y_true * y_pred)
                    return (2. * intersection + smooth) / (K.sum(y_true * y_true) + K.sum(y_pred * y_pred) + smooth)

                def dice_coef_loss(y_true, y_pred):
                    return -K.log(dice_coef(y_true, y_pred))

                metrics = [km.f1_score(),
                           km.precision(),
                           km.recall(),
                           'accuracy',
                           dice_coef]
                customObjects = {'dice_coef':dice_coef,'dice_coef_smooth1':dice_coef_smooth1,'dice_coef_loss':dice_coef_loss,
                                 'binary_f1_score':km.f1_score(),'binary_precision':km.precision(),'binary_recall':km.recall()}

                model = load_model('AN_conv5_16_4d32_flatten_mix_p5_cbam135-0125_dice_lr-5_l2005-00082-0.89583.h5', custom_objects=customObjects)

                pre_class = model.predict([pred_npy_input, pred_pickle_input] , batch_size = 8)


                df = pd.DataFrame(pre_class)
                df_da = []
                for j in range(len(df)):
                    if (df.loc[j][0] > df.loc[j][1]):
                        df_da.append('ADNI')
                    elif (df.loc[j][1] > df.loc[j][0]):
                        df_da.append('NIFD')

                df_da = pd.DataFrame(np.concatenate((pd.DataFrame(pickle_file),pd.DataFrame(fit_y),pd.DataFrame(pre_class),
                                                     pd.DataFrame(fit_xxx),pd.DataFrame(df_da)),axis =1))
                df_da.rename(columns = {0:'PTID',1:'EXAMDATE',2:'VISCODE',3:'FOLDER',4:'TRUE_AGE',5:'PTGENDER',
                                        6:'APOE4',7:'CDRSB',8:'CDRGL',9:'DX',10:'Race',11:'true1',12:'true2',
                                        13:'pred1',14:'pred2',15:'true_class',16:'pred_class'},inplace = True)

                df_da.to_csv(pickle_path_name + '/' +'ADNI&NIFD_classification.csv',index = False)
                messagebox.showinfo("Prediction Result","Predict Success")
    #######################################################################################
    #######################################################################################
    elif var_class == 'CDRSB_6class':
        pickle_path = filedialog.askopenfilename(parent=screen, title='Pickle File')
        pickle_path_name = os.path.split(pickle_path)[0]
        if not pickle_path:
            messagebox.showinfo("Mistake", 'No Choose Pickle File')
        else:
            pickle_path = pickle_path.replace("\\","/")
            file_path = filedialog.askopenfilename(parent=screen, title='Npy File')# 選擇檔案後回傳檔案路徑與名稱
            if not file_path:
                messagebox.showinfo("Mistake", 'No Choose Npy File')
            else:
                file_path = file_path.replace("\\","/")

                with open(str(pickle_path), 'rb') as f:
                    pickle_file = pickle.load(f)
                pickle_file = pd.DataFrame(pickle_file)
                pickle_file['PTGENDER'].replace("Female", 0, inplace = True) # 將"Famale"換成數值0 
                pickle_file['PTGENDER'].replace("Male", 1, inplace = True) # 將"male"換成數值1
                pickle_file['PTGENDER'].replace(2, 0, inplace = True) # JANDI將 2 (Female)  換成數值0
                messagebox.showinfo("Loading Success",'Loading success Please Click "OK" and Waiting the Prediction Result')

                ttt = np.zeros([1,11],dtype="float32")
                fir = np.concatenate((ttt,pickle_file),axis = 0)
                fir = fir[1:,:]
                pred_pickle = []

                for i in fir:
                    j = i[4:6]
                    pred_pickle.append(j)

                pred_pickle = np.asarray(pred_pickle).astype('float32')# X 與 z 要為相同的資料型態
                #正規化
                ss_val = ((pred_pickle[:,0]-50)/10) 
                ee_val = pred_pickle[:,1]
                pred_pickle_input = [ss_val,ee_val]
                pred_pickle_input = np.asarray(pred_pickle_input).astype('float32')
                pred_pickle_input = pred_pickle_input.T

                fir = pd.DataFrame(fir)
                fit_y = []
                for i in range(len(fir[7])):
                    if (fir[7][i] >= 0.0)&(fir[7][i] <= 0.25):
                        fit_y.append([1,0,0,0,0,0])
                    elif (fir[7][i] > 0.25)&(fir[7][i] <= 2.75):
                        fit_y.append([0,1,0,0,0,0])
                    elif (fir[7][i] > 2.75)&(fir[7][i] <= 4.25):
                        fit_y.append([0,0,1,0,0,0])
                    elif (fir[7][i] > 4.25)&(fir[7][i] <= 9.25):
                        fit_y.append([0,0,0,1,0,0])
                    elif (fir[7][i] > 9.25)&(fir[7][i] <= 15.75):
                        fit_y.append([0,0,0,0,1,0])
                    elif fir[7][i] > 15.75:
                        fit_y.append([0,0,0,0,0,1])

                fit_xxx = []
                for i in range(len(fir[7])):
                    if (fir[7][i] >= 0.0)&(fir[7][i] <= 0.25):
                        fit_xxx.append('Normal')
                    elif (fir[7][i] > 0.25)&(fir[7][i] <= 2.75):
                        fit_xxx.append('Questionable impairment')
                    elif (fir[7][i] > 2.75)&(fir[7][i] <= 4.25):
                        fit_xxx.append('Very mild dementia')
                    elif (fir[7][i] > 4.25)&(fir[7][i] <= 9.25):
                        fit_xxx.append('Mild dementia')
                    elif (fir[7][i] > 9.25)&(fir[7][i] <= 15.75):
                        fit_xxx.append('Moderate dementia')
                    elif fir[7][i] > 15.75:
                        fit_xxx.append('Server dementia') 

                npy_input = np.load(str(file_path),allow_pickle=True)
                pred_npy_input = np.zeros([1,145,121,121,3],dtype="float32")
                pred_npy_input = np.concatenate((pred_npy_input,npy_input) ,axis = 0)
                pred_npy_input = pred_npy_input[1:,:,:,:]
                pred_npy_input = np.concatenate((np.expand_dims(pred_npy_input[:,:,:,:,0],axis=-1),np.expand_dims(pred_npy_input[:,:,:,:,2],axis=-1)),axis=-1)
                pred_npy_input = pred_npy_input[:, 7:135, 12:108, :112, :]
                pred_pickle_input =np.array(pred_pickle_input)

                input_pickle = Input(shape=(pred_pickle_input.shape[1],))
                input_img = Input(shape=(pred_npy_input.shape[1], pred_npy_input.shape[2],pred_npy_input.shape[3],pred_npy_input.shape[4]))
                def dice_coef(y_true, y_pred, smooth=1e-5):
                    intersection = K.sum(y_true * y_pred)
                    return (2. * intersection + smooth) / (K.sum(y_true * y_true) + K.sum(y_pred * y_pred) + smooth)

                def dice_coef_loss(y_true, y_pred):
                    return -K.log(dice_coef(y_true, y_pred))

                metrics = [km.f1_score(),
                           km.precision(),
                           km.recall(),
                           'accuracy',
                           dice_coef]
                customObjects = {'dice_coef':dice_coef,'dice_coef_loss':dice_coef_loss,'binary_f1_score':km.f1_score(),
                                 'binary_precision':km.precision(),'binary_recall':km.recall()}

                model = load_model('6CDRclass_cut_conv5_32_4d32_D04_flatten_D04_mix_p5_cbam135-0125_dice_lr-4_l2003_b12_swplus-00195-0.39379.h5', custom_objects=customObjects)

                pre_class = model.predict([pred_npy_input, pred_pickle_input] , batch_size = 8)


                df = pd.DataFrame(pre_class)
                df_da = []
                for j in range(len(df)):
                    if (df.loc[j][0] > df.loc[j][1]) & (df.loc[j][0] > df.loc[j][2])& (df.loc[j][0] > df.loc[j][3])& (df.loc[j][0] > df.loc[j][4])& (df.loc[j][0] > df.loc[j][5]):
                        df_da.append('Normal')
                    elif (df.loc[j][1] > df.loc[j][0]) & (df.loc[j][1] > df.loc[j][2])& (df.loc[j][1] > df.loc[j][3])& (df.loc[j][1] > df.loc[j][4])& (df.loc[j][1] > df.loc[j][5]):
                        df_da.append('Questionable impairment')
                    elif (df.loc[j][2] > df.loc[j][0]) & (df.loc[j][2] > df.loc[j][1])& (df.loc[j][2] > df.loc[j][3])& (df.loc[j][2] > df.loc[j][4])& (df.loc[j][2] > df.loc[j][5]):
                        df_da.append('Very mild dementia')
                    elif (df.loc[j][3] > df.loc[j][0]) & (df.loc[j][3] > df.loc[j][1])& (df.loc[j][3] > df.loc[j][2])& (df.loc[j][3] > df.loc[j][4])& (df.loc[j][3] > df.loc[j][5]):
                        df_da.append('Mild dementia')
                    elif (df.loc[j][4] > df.loc[j][0]) & (df.loc[j][4] > df.loc[j][1])& (df.loc[j][4] > df.loc[j][2])& (df.loc[j][4] > df.loc[j][3])& (df.loc[j][4] > df.loc[j][5]):
                        df_da.append('Moderate dementia')
                    elif (df.loc[j][5] > df.loc[j][0]) & (df.loc[j][5] > df.loc[j][1])& (df.loc[j][5] > df.loc[j][2])& (df.loc[j][5] > df.loc[j][3])& (df.loc[j][5] > df.loc[j][4]):
                        df_da.append('Server dementia')

                df_da = pd.DataFrame(np.concatenate((pd.DataFrame(pickle_file),pd.DataFrame(fit_y),
                                                     pd.DataFrame(pre_class),pd.DataFrame(fit_xxx),pd.DataFrame(df_da)),axis =1))

                df_da.rename(columns = {0:'PTID',1:'EXAMDATE',2:'VISCODE',3:'FOLDER',4:'TRUE_AGE',5:'PTGENDER',
                                        6:'APOE4',7:'CDRSB',8:'CDRGL',9:'DX',10:'Race',
                                        11:'true1',12:'true2',13:'true3',14:'true4',15:'true5',16:'true6',
                                        17:'pred1',18:'pred2',19:'pred3',20:'pred4',21:'pred5',22:'pred6',
                                        23:'true_class',24:'pred_class'},inplace = True)

                df_da.to_csv(pickle_path_name + '/' + 'CDRSB_6class.csv',index = False)
                messagebox.showinfo("Prediction Result", "Predict Success")

    elif var_class == 'CDRSB_6reg':
        pickle_path = filedialog.askopenfilename(parent=screen, title='Pickle File')
        pickle_path_name = os.path.split(pickle_path)[0]
        if not pickle_path:
            messagebox.showinfo("Mistake", 'No Choose Pickle File')
        else:
            pickle_path = pickle_path.replace("\\","/")
            file_path = filedialog.askopenfilename(parent=screen, title='Npy File')# 選擇檔案後回傳檔案路徑與名稱
            if not file_path:
                messagebox.showinfo("Mistake", 'No Choose Npy File')
            else:
                file_path = file_path.replace("\\","/")

                with open(str(pickle_path), 'rb') as f:
                    pickle_file = pickle.load(f)
                pickle_file = pd.DataFrame(pickle_file)
                pickle_file['PTGENDER'].replace("Female", 0, inplace = True) # 將"Famale"換成數值0 
                pickle_file['PTGENDER'].replace("Male", 1, inplace = True) # 將"male"換成數值1
                pickle_file['PTGENDER'].replace(2, 0, inplace = True) # JANDI將 2 (Female)  換成數值0
                messagebox.showinfo("Loading Success",'Loading success Please Click "OK" and Waiting the Prediction Result')

                ttt = np.zeros([1,11],dtype="float32")
                fir = np.concatenate((ttt,pickle_file),axis = 0)
                fir = fir[1:,:]
                pred_pickle = []

                for i in fir:
                    j = i[4:6]
                    pred_pickle.append(j)

                pred_pickle = np.asarray(pred_pickle).astype('float32')# X 與 z 要為相同的資料型態
                #正規化
                ss_val = ((pred_pickle[:,0]-50)/10) 
                ee_val = pred_pickle[:,1]
                pred_pickle_input = [ss_val,ee_val]
                pred_pickle_input = np.asarray(pred_pickle_input).astype('float32')
                pred_pickle_input = pred_pickle_input.T

                fir = pd.DataFrame(fir)
                fit_y = []
                for i in range(len(fir[7])):
                    if (fir[7][i] >= 0.0)&(fir[7][i] <= 0.25):
                        fit_y.append(1)
                    elif (fir[7][i] > 0.25)&(fir[7][i] <= 2.75):
                        fit_y.append(2)
                    elif (fir[7][i] > 2.75)&(fir[7][i] <= 4.25):
                        fit_y.append(3)
                    elif (fir[7][i] > 4.25)&(fir[7][i] <= 9.25):
                        fit_y.append(4)
                    elif (fir[7][i] > 9.25)&(fir[7][i] <= 15.75):
                        fit_y.append(5)
                    elif fir[7][i] > 15.75:
                        fit_y.append(6)
                fit_xxx = []
                for i in range(len(fir[7])):
                    if (fir[7][i] >= 0.0)&(fir[7][i] <= 0.25):
                        fit_xxx.append('Normal')
                    elif (fir[7][i] > 0.25)&(fir[7][i] <= 2.75):
                        fit_xxx.append('Questionable impairment')
                    elif (fir[7][i] > 2.75)&(fir[7][i] <= 4.25):
                        fit_xxx.append('Very mild dementia')
                    elif (fir[7][i] > 4.25)&(fir[7][i] <= 9.25):
                        fit_xxx.append('Mild dementia')
                    elif (fir[7][i] > 9.25)&(fir[7][i] <= 15.75):
                        fit_xxx.append('Moderate dementia')
                    elif fir[7][i] > 15.75:
                        fit_xxx.append('Server dementia') 

                npy_input = np.load(str(file_path),allow_pickle=True)
                pred_npy_input = np.zeros([1,145,121,121,3],dtype="float32")
                pred_npy_input = np.concatenate((pred_npy_input,npy_input) ,axis = 0)
                pred_npy_input = pred_npy_input[1:,:,:,:]
                pred_npy_input = np.concatenate((np.expand_dims(pred_npy_input[:,:,:,:,0],axis=-1),np.expand_dims(pred_npy_input[:,:,:,:,2],axis=-1)),axis=-1)
                pred_npy_input = pred_npy_input[:, 7:135, 12:108, :112, :]
                pred_pickle_input =np.array(pred_pickle_input)

                input_pickle = Input(shape=(pred_pickle_input.shape[1],))
                input_img = Input(shape=(pred_npy_input.shape[1], pred_npy_input.shape[2],pred_npy_input.shape[3],pred_npy_input.shape[4]))
                class Swish(Activation):
                    def __init__(self, activation, **kwargs):
                        super(Swish, self).__init__(activation, **kwargs)
                        self.__name__ = 'swish'

                def swish(x):
                    beta = 0.5
                    return (K.sigmoid(beta*x) * x)

                get_custom_objects().update({'swish': Swish(swish)})

                customObjects = {'Swish':Swish,'swish':swish}

                model = load_model('6CDRclass1to6_transfer_learningCDR053_4-00178-1.12959.h5', custom_objects=customObjects)

                pre_class = model.predict([pred_npy_input, pred_pickle_input] , batch_size = 8)
                pre_class = pd.DataFrame(np.round(pre_class,0))
                df_da = []
                for ii in range(len(pre_class)):
                    if (pre_class.loc[ii] <= 1).all():
                        df_da.append('Normal')
                    elif (pre_class.loc[ii] == 2).all():
                        df_da.append('Questionable impairment')
                    elif(pre_class.loc[ii] == 3).all():
                        df_da.append('Very mild dementia')
                    elif (pre_class.loc[ii] == 4).all():
                        df_da.append('Mild dementia')
                    elif (pre_class.loc[ii] == 5).all():
                        df_da.append('Moderate dementia')
                    elif (pre_class.loc[ii] >= 6).all():
                        df_da.append('Severe dementia')

                df_da = pd.DataFrame(df_da)
                df_da = pd.DataFrame(np.concatenate((pd.DataFrame(pickle_file),pd.DataFrame(fit_y),
                                                     pre_class,pd.DataFrame(fit_xxx),pd.DataFrame(df_da)),axis =1))

                df_da.rename(columns = {0:'PTID',1:'EXAMDATE',2:'VISCODE',3:'FOLDER',4:'TRUE_AGE',5:'PTGENDER',
                                        6:'APOE4',7:'CDRSB',8:'CDRGL',9:'DX',10:'Race',
                                        11:'true1',12:'pred1',13:'true_class',14:'pred_class'},inplace = True)

                df_da.to_csv(pickle_path_name + '/' +'CDRSB_6reg.csv',index = False)
                messagebox.showinfo("Prediction Result", "Predict Success")
        
    elif var_class == '3111class':
        pickle_path = filedialog.askopenfilename(parent=screen, title='Pickle File')
        pickle_path_name = os.path.split(pickle_path)[0]
        if not pickle_path:
            messagebox.showinfo("Mistake", 'No Choose Pickle File')
        else:
            pickle_path = pickle_path.replace("\\","/")
            file_path = filedialog.askopenfilename(parent=screen, title='Npy File')# 選擇檔案後回傳檔案路徑與名稱
            if not file_path:
                messagebox.showinfo("Mistake", 'No Choose Npy File')
            else:
                file_path = file_path.replace("\\","/")

                with open(str(pickle_path), 'rb') as f:
                    pickle_file = pickle.load(f)
                pickle_file = pd.DataFrame(pickle_file)
                pickle_file['PTGENDER'].replace("Female", 0, inplace = True) # 將"Famale"換成數值0 
                pickle_file['PTGENDER'].replace("Male", 1, inplace = True) # 將"male"換成數值1
                pickle_file['PTGENDER'].replace(2, 0, inplace = True) # JANDI將 2 (Female)  換成數值0
                messagebox.showinfo("Loading Success", 'Loading Success Please Click "OK" and Waiting the Prediction Result')

                ttt = np.zeros([1,11],dtype="float32")
                fir = np.concatenate((ttt,pickle_file),axis = 0)
                fir = fir[1:,:]
                pred_pickle = []

                for i in fir:
                    j = i[4:6]
                    pred_pickle.append(j)

                pred_pickle = np.asarray(pred_pickle).astype('float32')# X 與 z 要為相同的資料型態
                #正規化
                ss_val = ((pred_pickle[:,0]-50)/10) 
                ee_val = pred_pickle[:,1]
                pred_pickle_input = [ss_val,ee_val]
                pred_pickle_input = np.asarray(pred_pickle_input).astype('float32')
                pred_pickle_input = pred_pickle_input.T

                fir = pd.DataFrame(fir)
                fit_y = []
                for i in range(len(fir[7])):
                    if (fir[7][i] >= 0.0)&(fir[7][i] <= 4.0):
                        fit_y.append([1,0,0,0])
                    elif (fir[7][i] > 4.0)&(fir[7][i] <= 9.0):
                        fit_y.append([0,1,0,0])
                    elif (fir[7][i] > 9.0)&(fir[7][i] <= 15.5):
                        fit_y.append([0,0,1,0])
                    elif (fir[7][i] > 15.5):
                        fit_y.append([0,0,0,1])

                fit_xxx = []
                for i in range(len(fir[7])):
                    if (fir[7][i] >= 0.0)&(fir[7][i] <= 4.0):
                        fit_xxx.append('1st')
                    elif (fir[7][i] > 4.0)&(fir[7][i] <= 9.0):
                        fit_xxx.append('2nd')
                    elif (fir[7][i] > 9.0)&(fir[7][i] <= 15.5):
                        fit_xxx.append('3rd')
                    elif (fir[7][i] > 15.5):
                        fit_xxx.append('4th') 

                npy_input = np.load(str(file_path),allow_pickle=True)
                pred_npy_input = np.zeros([1,145,121,121,3],dtype="float32")
                pred_npy_input = np.concatenate((pred_npy_input,npy_input) ,axis = 0)
                pred_npy_input = pred_npy_input[1:,:,:,:]
                pred_npy_input = np.concatenate((np.expand_dims(pred_npy_input[:,:,:,:,0],axis=-1),np.expand_dims(pred_npy_input[:,:,:,:,2],axis=-1)),axis=-1)
                pred_npy_input = pred_npy_input[:, 7:135, 12:108, :112, :]
                pred_pickle_input =np.array(pred_pickle_input)

                input_pickle = Input(shape=(pred_pickle_input.shape[1],))
                input_img = Input(shape=(pred_npy_input.shape[1], pred_npy_input.shape[2],pred_npy_input.shape[3],pred_npy_input.shape[4]))
                def dice_coef(y_true, y_pred, smooth=1e-5):
                    intersection = K.sum(y_true * y_pred)
                    return (2. * intersection + smooth) / (K.sum(y_true * y_true) + K.sum(y_pred * y_pred) + smooth)
                def dice_coef_loss(y_true, y_pred):
                    return -K.log(dice_coef(y_true, y_pred))

                metrics = [km.f1_score(),
                           km.precision(),
                           km.recall(),
                           'accuracy',
                           dice_coef]


                customObjects = {'dice_coef':dice_coef,'dice_coef_loss':dice_coef_loss,'binary_f1_score':km.f1_score(),
                                 'binary_precision':km.precision(),'binary_recall':km.recall()}

                model = load_model('3111class_cut_conv5_32-32-64-64-64-32_4d32_D01_flatten_D01_mix_p5_cbam135-0125_dice_lr-6_xl2005_b12_swplus_last2-1x25-00016-0.70025.h5', custom_objects=customObjects)

                pre_class = model.predict([pred_npy_input, pred_pickle_input] , batch_size = 8)
                df = pd.DataFrame(pre_class)
                df_da = []
                for j in range(len(df)):
                    if (df.loc[j][0] > df.loc[j][1]) & (df.loc[j][0] > df.loc[j][2])& (df.loc[j][0] > df.loc[j][3]):
                        df_da.append("1st")
                    elif (df.loc[j][1] > df.loc[j][0]) & (df.loc[j][1] > df.loc[j][2])& (df.loc[j][1] > df.loc[j][3]):
                        df_da.append("2nd")
                    elif (df.loc[j][2] > df.loc[j][0]) & (df.loc[j][2] > df.loc[j][1])& (df.loc[j][2] > df.loc[j][3]):
                        df_da.append("3rd")
                    elif (df.loc[j][3] > df.loc[j][0]) & (df.loc[j][3] > df.loc[j][1])& (df.loc[j][3] > df.loc[j][2]):
                        df_da.append("4th")

                df_da = pd.DataFrame(np.concatenate((pd.DataFrame(pickle_file),pd.DataFrame(fit_y),pd.DataFrame(pre_class),
                                                     pd.DataFrame(fit_xxx),pd.DataFrame(df_da)),axis =1))

                df_da.rename(columns = {0:'PTID',1:'EXAMDATE',2:'VISCODE',3:'FOLDER',4:'TRUE_AGE',5:'PTGENDER',
                                        6:'APOE4',7:'CDRSB',8:'CDRGL',9:'DX',10:'Race',
                                        11:'true1',12:'true2',13:'true3',14:'true4',
                                        15:'pred1',16:'pred2',17:'pred3',18:'pred4',
                                        19:'true_class',20:'pred_class'},inplace = True)

                df_da.to_csv(pickle_path_name + '/' +'3111class.csv',index = False)
                messagebox.showinfo("Prediction Result", "Predict Success")
        
    elif var_class == '2112class':
        pickle_path = filedialog.askopenfilename(parent=screen, title='Pickle File')
        pickle_path_name = os.path.split(pickle_path)[0]
        if not pickle_path:
            messagebox.showinfo("Mistake", 'No Choose Pickle File')
        else:
            pickle_path = pickle_path.replace("\\","/")
            file_path = filedialog.askopenfilename(parent=screen, title='Npy File')# 選擇檔案後回傳檔案路徑與名稱
            if not file_path:
                messagebox.showinfo("Mistake", 'No Choose Npy File')
            else:
                file_path = file_path.replace("\\","/")

                with open(str(pickle_path), 'rb') as f:
                    pickle_file = pickle.load(f)
                pickle_file = pd.DataFrame(pickle_file)
                pickle_file['PTGENDER'].replace("Female", 0, inplace = True) # 將"Famale"換成數值0 
                pickle_file['PTGENDER'].replace("Male", 1, inplace = True) # 將"male"換成數值1
                pickle_file['PTGENDER'].replace(2, 0, inplace = True) # JANDI將 2 (Female)  換成數值0
                messagebox.showinfo("Loading Success", 'Loading Success Please Click "OK" and Waiting the Prediction Result')

                ttt = np.zeros([1,11],dtype="float32")
                fir = np.concatenate((ttt,pickle_file),axis = 0)
                fir = fir[1:,:]
                pred_pickle = []
                for i in fir:
                    j = i[4:6]
                    pred_pickle.append(j)

                pred_pickle = np.asarray(pred_pickle).astype('float32')# X 與 z 要為相同的資料型態
                #正規化
                ss_val = ((pred_pickle[:,0]-50)/10) 
                ee_val = pred_pickle[:,1]
                pred_pickle_input = [ss_val,ee_val]
                pred_pickle_input = np.asarray(pred_pickle_input).astype('float32')
                pred_pickle_input = pred_pickle_input.T

                fir = pd.DataFrame(fir)
                fit_y = []
                for i in range(len(fir[7])):
                    if (fir[7][i] >= 0.0)&(fir[7][i] <= 0.25):
                        fit_y.append([1,0,0,0])
                    elif (fir[7][i] > 0.25)&(fir[7][i] <= 4.0):
                        fit_y.append([0,1,0,0])
                    elif (fir[7][i] > 4.0)&(fir[7][i] <= 9.0):
                        fit_y.append([0,0,1,0])
                    elif (fir[7][i] > 9.0):
                        fit_y.append([0,0,0,1])

                fit_xxx = []
                for i in range(len(fir[7])):
                    if (fir[7][i] >= 0.0)&(fir[7][i] <= 0.25):
                        fit_xxx.append('1st')
                    elif (fir[7][i] > 0.25)&(fir[7][i] <= 4.0):
                        fit_xxx.append('2nd')
                    elif (fir[7][i] > 4.0)&(fir[7][i] <= 9.0):
                        fit_xxx.append('3rd')
                    elif (fir[7][i] > 9.0):
                        fit_xxx.append('4th')

                npy_input = np.load(str(file_path),allow_pickle=True)
                pred_npy_input = np.zeros([1,145,121,121,3],dtype="float32")
                pred_npy_input = np.concatenate((pred_npy_input,npy_input) ,axis = 0)
                pred_npy_input = pred_npy_input[1:,:,:,:]
                pred_npy_input = np.concatenate((np.expand_dims(pred_npy_input[:,:,:,:,0],axis=-1),np.expand_dims(pred_npy_input[:,:,:,:,2],axis=-1)),axis=-1)
                pred_npy_input = pred_npy_input[:, 7:135, 12:108, :112, :]
                pred_pickle_input =np.array(pred_pickle_input)

                input_pickle = Input(shape=(pred_pickle_input.shape[1],))
                input_img = Input(shape=(pred_npy_input.shape[1], pred_npy_input.shape[2],pred_npy_input.shape[3],pred_npy_input.shape[4]))
                def dice_coef(y_true, y_pred, smooth=1e-5):
                    intersection = K.sum(y_true * y_pred)
                    return (2. * intersection + smooth) / (K.sum(y_true * y_true) + K.sum(y_pred * y_pred) + smooth)
                def dice_coef_loss(y_true, y_pred):
                    return -K.log(dice_coef(y_true, y_pred))

                metrics = [km.f1_score(),
                           km.precision(),
                           km.recall(),
                           'accuracy',
                           dice_coef]
                
                customObjects = {'dice_coef':dice_coef,'dice_coef_loss':dice_coef_loss,'binary_f1_score':km.f1_score(),
                                 'binary_precision':km.precision(),'binary_recall':km.recall()}

                model = load_model('2112class_cut_conv5_32_4d32_D05_flatten_D05_mix_p5_cbam135-0125_dice_lr-4_l2005_b12_swplus-00038-0.60118.h5', custom_objects=customObjects)

                pre_class = model.predict([pred_npy_input, pred_pickle_input] , batch_size = 8)
                df = pd.DataFrame(pre_class)
                df_da = []
                for j in range(len(df)):
                    if (df.loc[j][0] > df.loc[j][1]) & (df.loc[j][0] > df.loc[j][2])& (df.loc[j][0] > df.loc[j][3]):
                        df_da.append("1st")
                    elif (df.loc[j][1] > df.loc[j][0]) & (df.loc[j][1] > df.loc[j][2])& (df.loc[j][1] > df.loc[j][3]):
                        df_da.append("2nd")
                    elif (df.loc[j][2] > df.loc[j][0]) & (df.loc[j][2] > df.loc[j][1])& (df.loc[j][2] > df.loc[j][3]):
                        df_da.append("3rd")
                    elif (df.loc[j][3] > df.loc[j][0]) & (df.loc[j][3] > df.loc[j][1])& (df.loc[j][3] > df.loc[j][2]):
                        df_da.append("4th")

                df_da = pd.DataFrame(df_da)
                df_da = pd.DataFrame(np.concatenate((pd.DataFrame(pickle_file),pd.DataFrame(fit_y),pd.DataFrame(pre_class),
                                                     pd.DataFrame(fit_xxx),pd.DataFrame(df_da)),axis =1))

                df_da.rename(columns = {0:'PTID',1:'EXAMDATE',2:'VISCODE',3:'FOLDER',4:'TRUE_AGE',5:'PTGENDER',
                                        6:'APOE4',7:'CDRSB',8:'CDRGL',9:'DX',10:'Race',
                                        11:'true1',12:'true2',13:'true3',14:'true4',
                                        15:'pred1',16:'pred2',17:'pred3',18:'pred4',
                                        19:'true_class',20:'pred_class'},inplace = True)

                df_da.to_csv(pickle_path_name + '/' +'2112class.csv',index = False)
                messagebox.showinfo("Prediction Result", "Predict Success")
        
    elif var_class == 'CN_MCI_other':
        pickle_path = filedialog.askopenfilename(parent=screen, title='Pickle File')
        pickle_path_name = os.path.split(pickle_path)[0]
        if not pickle_path:
            messagebox.showinfo("Mistake", 'No Choose Pickle File')
        else:
            pickle_path = pickle_path.replace("\\","/")
            file_path = filedialog.askopenfilename(parent=screen, title='Npy File')# 選擇檔案後回傳檔案路徑與名稱
            if not file_path:
                messagebox.showinfo("Mistake", 'No Choose Npy File')
            else:
                file_path = file_path.replace("\\","/")

                with open(str(pickle_path), 'rb') as f:
                    pickle_file = pickle.load(f)
                pickle_file = pd.DataFrame(pickle_file)
                pickle_file['PTGENDER'].replace("Female", 0, inplace = True) # 將"Famale"換成數值0 
                pickle_file['PTGENDER'].replace("Male", 1, inplace = True) # 將"male"換成數值1
                pickle_file['PTGENDER'].replace(2, 0, inplace = True) # JANDI將 2 (Female)  換成數值0
                messagebox.showinfo("Loading Success", 'Loading Success Please Click "OK" and Waiting the Prediction Result')

                ttt = np.zeros([1,11],dtype="float32")
                fir = np.concatenate((ttt,pickle_file),axis = 0)
                fir = fir[1:,:]
                pred_pickle = []

                for i in fir:
                    j = i[4:6]
                    pred_pickle.append(j)

                pred_pickle = np.asarray(pred_pickle).astype('float32')# X 與 z 要為相同的資料型態
                #正規化
                ss_val = ((pred_pickle[:,0]-50)/10) 
                ee_val = pred_pickle[:,1]
                pred_pickle_input = [ss_val,ee_val]
                pred_pickle_input = np.asarray(pred_pickle_input).astype('float32')
                pred_pickle_input = pred_pickle_input.T

                fir = pd.DataFrame(fir)
                fit_y = []
                for i in range(len(fir[7])):
                    if (fir[7][i] >= 0)&(fir[7][i] <= 0.25):
                        fit_y.append([1,0,0])
                    elif (fir[7][i] > 0.25)&(fir[7][i] <= 3):
                        fit_y.append([0,1,0])
                    elif fir[7][i] > 3:
                        fit_y.append([0,0,1])

                fit_xxx = []
                for i in range(len(fir[7])):
                    if (fir[7][i] >= 0)&(fir[7][i] <= 0.25):
                        fit_xxx.append('CN')
                    elif (fir[7][i] > 0.25)&(fir[7][i] <= 3):
                        fit_xxx.append('MCI')
                    elif fir[7][i] > 3:
                        fit_xxx.append('other')


                npy_input = np.load(str(file_path),allow_pickle=True)
                pred_npy_input = np.zeros([1,145,121,121,3],dtype="float32")
                pred_npy_input = np.concatenate((pred_npy_input,npy_input) ,axis = 0)
                pred_npy_input = pred_npy_input[1:,:,:,:]
                pred_npy_input = np.concatenate((np.expand_dims(pred_npy_input[:,:,:,:,0],axis=-1),np.expand_dims(pred_npy_input[:,:,:,:,2],axis=-1)),axis=-1)
                pred_pickle_input =np.array(pred_pickle_input)

                input_pickle = Input(shape=(pred_pickle_input.shape[1],))
                input_img = Input(shape=(pred_npy_input.shape[1], pred_npy_input.shape[2],pred_npy_input.shape[3],pred_npy_input.shape[4]))
                def dice_coef(y_true, y_pred, smooth=1e-5):
                    intersection = K.sum(y_true * y_pred)
                    return (2. * intersection + smooth) / (K.sum(y_true * y_true) + K.sum(y_pred * y_pred) + smooth)
                def dice_coef_loss(y_true, y_pred):
                    return -K.log(dice_coef(y_true, y_pred))

                metrics = [km.f1_score(),
                           km.precision(),
                           km.recall(),
                           'accuracy',
                           dice_coef]


                customObjects = {'dice_coef':dice_coef,'dice_coef_loss':dice_coef_loss,'binary_f1_score':km.f1_score(),
                                 'binary_precision':km.precision(),'binary_recall':km.recall()}

                model = load_model('CN_MCI_other_conv5_32_4d32_D04_flatten_D04_mix_p5_cbam135-0125_dice_lr-4_l2005_set_b12-00034-0.64484.h5', custom_objects=customObjects)

                pre_class = model.predict([pred_npy_input, pred_pickle_input] , batch_size = 8)
                df = pd.DataFrame(pre_class)
                df_da = []
                for j in range(len(df)):
                    if (df.loc[j][0] > df.loc[j][1]) & (df.loc[j][0] > df.loc[j][2]):
                        df_da.append('CN')
                    elif (df.loc[j][1] > df.loc[j][0]) & (df.loc[j][1] > df.loc[j][2]):
                        df_da.append('MCI')
                    elif (df.loc[j][2] > df.loc[j][0]) & (df.loc[j][2] > df.loc[j][1]):
                        df_da.append('other')

                df_da = pd.DataFrame(df_da)
                df_da = pd.DataFrame(np.concatenate((pd.DataFrame(pickle_file),pd.DataFrame(fit_y),pd.DataFrame(pre_class),
                                                     pd.DataFrame(fit_xxx),pd.DataFrame(df_da)),axis =1))
                df_da.rename(columns = {0:'PTID',1:'EXAMDATE',2:'VISCODE',3:'FOLDER',4:'TRUE_AGE',5:'PTGENDER',
                                        6:'APOE4',7:'CDRSB',8:'CDRGL',9:'DX',10:'Race',
                                        11:'true1',12:'true2',13:'true3',
                                        14:'pred1',15:'pred2',16:'pred3',
                                        17:'true_class',18:'pred_class'},inplace = True)

                df_da.to_csv(pickle_path_name + '/' +'CN_MCI_other.csv',index = False)
                messagebox.showinfo("Prediction Result", "Predict Success")
    else:
        messagebox.showinfo("Mistake", "Model Not Choose Success")

screen = Tk()
screen.geometry("400x500")
screen.resizable(0,0)
screen.title('Predict Model')
heading = Label(text ="Welcome" ,font=('Arial',12,'bold'),bg ="grey",fg ="yellow" ,width = "500" ,height = "3")
heading.pack()

save_npy_text = Label(text ="Nii File Translate Npy File",font=('Arial',12,'bold'),bg = "grey")
save_npy_text.place(x = 15 , y = 70)        
choosemodel_text = Label(text ="Choose Testing Model",font=('Arial',12,'bold'),bg = "grey")
choosemodel_text.place(x = 15 , y = 180)    

var=StringVar()
var.set("Please choose you want to testing model")
values = ("BrainAge", "NIFD_2ndclass", "ADNI&NIFD_classification" , "CDRSB_6class" , "CDRSB_6reg" , "3111class" , "2112class" , "CN_MCI_other")
combobox=ttk.Combobox(screen, values=values, width = 36,textvariable=var)
combobox.bind('<<ComboboxSelected>>', show_value)
combobox.place(x = 15 ,y = 210)

save_npy = Button(screen ,text="Nii Package Location" ,font=('Arial', 12,'bold'),command = nii_file)
save_npy.place(x= 15 , y=100)

quit = Button(screen ,text = "Quit" , width ="15",font=('Arial',12,'bold'),command = screen.destroy)
quit.place(x = 15,y = 450)
screen.mainloop()




