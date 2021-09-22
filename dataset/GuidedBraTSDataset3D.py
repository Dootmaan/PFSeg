# BraTS2020
import torch
import glob
import SimpleITK as sitk
import torch.utils.data
import os
from scipy import ndimage
import numpy as np
import cv2
import random
from config import config

img_size=config.input_img_size
crop_size=config.crop_size  # z,x,y
# img_size=[64,96,96]
# crop_size=[64,96,96]
guide_img_size=[i//2 for i in crop_size]

print('Using patch size:', crop_size)
print('Using guidance size:', guide_img_size)

class GuidedBraTSDataset3D(torch.utils.data.Dataset):
    def __init__(self,path,mode='train',augment=True):
        self.data=[]
        self.label_seg=[]
        self.label_sr=[]
        self.mode=mode
        self.aug=augment
        images=sorted(glob.glob(os.path.join(path, "*/*/*_t2.nii.gz")))
        labels=sorted(glob.glob(os.path.join(path, "*/*/*_seg.nii.gz")))

        # Whether shuffle the dataset. Default not because we need to evaluate test dice for different models.
        # bundle = list(zip(images, labels))
        # random.shuffle(bundle)
        # images[:], labels[:] = zip(*bundle)

        train_frac, val_frac, test_frac = 0.6, 0.2, 0.2
        n_train = int(train_frac * len(images)) + 1
        n_val = int(val_frac * len(images)) + 1
        n_test = min(len(images) - n_train - n_val, int(test_frac * len(images)))
        
        # Accelarate by loading all data into memory
        if mode=='train':
            print("train:",n_train, "folder:",path)
            images=images[:n_train]
            labels=labels[:n_train]
            for i in range(len(images)):
                print('Adding train sample:',images[i])
                image=sitk.ReadImage(images[i])
                image_arr=sitk.GetArrayFromImage(image)
                lesion=sitk.ReadImage(labels[i])
                lesion_arr=sitk.GetArrayFromImage(lesion)
                lesion_arr[lesion_arr>1]=1  # 只做WT分割\
                img,label_seg,label_sr=self.cropMR(image_arr,lesion_arr)
                label_seg[label_seg<0.5]=0.
                label_seg[label_seg>=0.5]=1.
                self.data.append(img)
                self.label_seg.append(label_seg)
                self.label_sr.append(label_sr)
            
        elif mode=='val':
            print("val:", n_val, "folder:",path)
            images=images[n_train:n_train+n_val]
            labels=labels[n_train:n_train+n_val]
            for i in range(len(images)):
                print('Adding val sample:',images[i])
                image=sitk.ReadImage(images[i])
                image_arr=sitk.GetArrayFromImage(image)
                lesion=sitk.ReadImage(labels[i])
                lesion_arr=sitk.GetArrayFromImage(lesion)
                lesion_arr[lesion_arr>1]=1  # 只做WT分割
                img,label_seg,label_sr=self.cropMR(image_arr,lesion_arr)
                label_seg[label_seg<0.5]=0.
                label_seg[label_seg>=0.5]=1.
                self.data.append(img)
                self.label_seg.append(label_seg)
                self.label_sr.append(label_sr)

        elif mode=='test':
            print("test:", n_test, "folder:",path)
            images=images[n_train+n_val:n_train+n_val+n_test]
            labels=labels[n_train+n_val:n_train+n_val+n_test]
            for i in range(len(images)):
                print('Adding test sample:',images[i])
                image=sitk.ReadImage(images[i])
                image_arr=sitk.GetArrayFromImage(image)
                lesion=sitk.ReadImage(labels[i])
                lesion_arr=sitk.GetArrayFromImage(lesion)
                lesion_arr[lesion_arr>1]=1  # 只做WT分割
                img,label_seg,label_sr=self.cropMR(image_arr,lesion_arr)
                label_seg[label_seg<0.5]=0.
                label_seg[label_seg>=0.5]=1.
                self.data.append(img)
                self.label_seg.append(label_seg)
                self.label_sr.append(label_sr)

        elif mode=='all':
            print("all:", len(images), "folder:",path)
            for i in range(len(images)):
                # if i<0.98*len(images):
                #     continue
                print('Adding all sample:',images[i])
                image=sitk.ReadImage(images[i])
                image_arr=sitk.GetArrayFromImage(image)
                lesion=sitk.ReadImage(labels[i])
                lesion_arr=sitk.GetArrayFromImage(lesion)
                lesion_arr[lesion_arr>1]=1  # 只做WT分割
                img,label_seg,label_sr=self.cropMR(image_arr,lesion_arr)
                label_seg[label_seg<0.5]=0.
                label_seg[label_seg>=0.5]=1.
                self.data.append(img)
                self.label_seg.append(label_seg)
                self.label_sr.append(label_sr)


    def normalization(self,image_array):
        max = image_array.max()
        min = image_array.min()
        #归一化
        image_array = 1.0*(image_array - min)/(max - min)
        #image_array = image_array.astype(int)#整型
        return image_array
    
    def cropMR(self,img,mask):
        # result=np.where(img!=0)
        # z_list=result[0]
        # x_list=result[1]
        # y_list=result[2]

        # x_max=x_list.max()
        # x_min=x_list.min()
        # y_max=y_list.max()
        # y_min=y_list.min()
        # z_max=z_list.max()
        # z_min=z_list.min()
        img=img[:,24:-24,24:-24]
        mask=mask[:,24:-24,24:-24]

        return self.normalization(ndimage.interpolation.zoom(img,[img_size[0]/155,0.5,0.5],order=1)),self.normalization(ndimage.interpolation.zoom(mask,[2*img_size[0]/155,1,1],order=0)),self.normalization(ndimage.interpolation.zoom(img,[2*img_size[0]/155,1,1],order=1))

    def crop_guidance(self,label_sr):
        # start_z=random.randint(24,2*img_size[0]-guide_img_size[0]-24)
        # start_x=random.randint(24,2*img_size[1]-guide_img_size[1]-24)
        # start_y=random.randint(24,2*img_size[2]-guide_img_size[2]-24)

        # mask=np.zeros(label_sr.shape).astype(np.float64)
        # mask[start_z:(start_z+guide_img_size[0]),start_x:(start_x+guide_img_size[1]),start_y:(start_y+guide_img_size[2])]=1
        # return label_sr[start_z:(start_z+guide_img_size[0]),start_x:(start_x+guide_img_size[1]),start_y:(start_y+guide_img_size[2])],mask

        D,M,N=label_sr.shape
        D=int(D/2)
        M=int(M/2)
        N=int(N/2)

        mask=np.zeros(label_sr.shape).astype(np.float64)
        mask[D-guide_img_size[0]//2:D+guide_img_size[0]//2,M-guide_img_size[1]//2:M+guide_img_size[1]//2,N-guide_img_size[2]//2:N+guide_img_size[2]//2]=1
        return label_sr[D-guide_img_size[0]//2:D+guide_img_size[0]//2,M-guide_img_size[1]//2:M+guide_img_size[1]//2,N-guide_img_size[2]//2:N+guide_img_size[2]//2],mask
    
    def augment(self,img,label_seg,label_sr):
        # raw_img=img.copy()
        # raw_label_seg=label_seg.copy()
        # raw_label_sr=label_sr.copy()
        if random.random()<0.5:  #Flip
            if random.random()<0.5:
                for i in range(img.shape[0]):
                    img[i,:,:]=cv2.flip(img[i,:,:],0)
                for i in range(label_seg.shape[0]):
                    label_seg[i,:,:]=cv2.flip(label_seg[i,:,:],0)
                    label_sr[i,:,:]=cv2.flip(label_sr[i,:,:],0)
            else:
                for i in range(img.shape[0]):
                    img[i,:,:]=cv2.flip(img[i,:,:],1)
                for i in range(label_seg.shape[0]):
                    label_seg[i,:,:]=cv2.flip(label_seg[i,:,:],1)
                    label_sr[i,:,:]=cv2.flip(label_sr[i,:,:],1)
                    

        if random.random()<0.5:  #Shift
            vertical=np.random.randint(-img.shape[1]//8,img.shape[1]//8)
            horizon=np.random.randint(-img.shape[1]//8,img.shape[1]//8)
            M_img=np.float32([[0,1,horizon],[1,0,vertical]])
            M_label=np.float32([[0,1,2*horizon],[1,0,2*vertical]])
            for i in range(img.shape[0]):
                img[i,:,:]=cv2.warpAffine(img[i,:,:],M_img,(img.shape[1],img.shape[2]))
            for i in range(label_seg.shape[0]):
                label_seg[i,:,:]=cv2.warpAffine(label_seg[i,:,:],M_label,(label_seg.shape[1],label_seg.shape[2]))
                label_sr[i,:,:]=cv2.warpAffine(label_sr[i,:,:],M_label,(label_sr.shape[1],label_sr.shape[2]))
        
        if random.random()<0.5: #Rotate
            degree=np.random.randint(0,360)
            M_img = cv2.getRotationMatrix2D(((img.shape[1]-1)/2.0,(img.shape[2]-1)/2.0),degree,1)
            M_label=cv2.getRotationMatrix2D(((label_seg.shape[1]-1)/2.0,(label_seg.shape[2]-1)/2.0),degree,1)
            for i in range(img.shape[0]):
                img[i,:,:]=cv2.warpAffine(img[i,:,:],M_img,(img.shape[1],img.shape[2]))
            for i in range(label_seg.shape[0]):
                label_seg[i,:,:]=cv2.warpAffine(label_seg[i,:,:],M_label,(label_seg.shape[1],label_seg.shape[2]))
                label_sr[i,:,:]=cv2.warpAffine(label_sr[i,:,:],M_label,(label_sr.shape[1],label_sr.shape[2]))

        #random crop
        start_z=random.randint(0,img_size[0]-crop_size[0])
        start_x=random.randint(0,img_size[1]-crop_size[1])
        start_y=random.randint(0,img_size[2]-crop_size[2])

        label_seg[label_seg<0.5]=0.
        label_seg[label_seg>=0.5]=1.

        return img[start_z:(start_z+crop_size[0]),start_x:(start_x+crop_size[1]),start_y:(start_y+crop_size[2])],label_seg[2*start_z:2*(start_z+crop_size[0]),2*start_x:2*(start_x+crop_size[1]),2*start_y:2*(start_y+crop_size[2])],label_sr[2*start_z:2*(start_z+crop_size[0]),2*start_x:2*(start_x+crop_size[1]),2*start_y:2*(start_y+crop_size[2])]

    def __len__(self):
        return len(self.label_seg)

    def __getitem__(self,index):
        if index > self.__len__():
            print("Index exceeds length!")
            return None
            
        if self.mode=='train' or self.mode=='all':
            if self.aug:
                data,label_seg,label_sr= self.augment(self.data[index],self.label_seg[index],self.label_sr[index])
                guidance,mask=self.crop_guidance(self.label_sr[index])
                return data,label_seg,label_sr,guidance,mask
            else:
                guidance,mask=self.crop_guidance(self.label_sr[index])
                return self.data[index],self.label_seg[index],self.label_sr[index],guidance,mask
        else:
            guidance,mask=self.crop_guidance(self.label_sr[index])
            return self.data[index],self.label_seg[index],self.label_sr[index],guidance,mask

if __name__=='__main__':
    dataset=GuidedBraTSDataset3D('/newdata/why/BraTS20',mode='all')
    test=dataset.__getitem__(0)
    cv2.imwrite('img.png',test[0][32,:,:]*255)
    cv2.imwrite('seg.png',test[1][32,:,:]*255)
    cv2.imwrite('sr.png',test[2][32,:,:]*255)
    print(test)

