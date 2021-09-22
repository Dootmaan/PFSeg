import torch as pt
import numpy as np
from model.PFSeg import PFSeg3D
from medpy.metric.binary import jc,hd95
from dataset.GuidedBraTSDataset3D import GuidedBraTSDataset3D
# from loss.FALoss3D import FALoss3D
import cv2
from loss.TaskFusionLoss import TaskFusionLoss
from loss.DiceLoss import BinaryDiceLoss
from config import config
import argparse
from tqdm import tqdm
# from tensorboardX import SummaryWriter

crop_size=config.crop_size
size=crop_size[2]
img_size=config.input_img_size

parser = argparse.ArgumentParser(description='Patch-free 3D Medical Image Segmentation.')
parser.add_argument('-dataset_path',type=str,default='/newdata/why/BraTS20',help='path to dataset')
parser.add_argument('-model_save_to',type=str,default='.',help='path to output')
parser.add_argument('-bs', type=int, default=1, help='input batch size')
parser.add_argument('-epoch', type=int, default=100, help='number of epochs')
parser.add_argument('-lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('-w_sr', type=float, default=0.5, help='w_sr of the lossfunc')
parser.add_argument('-w_tf', type=float, default=0.5, help='w_tf of the lossfunc')
parser.add_argument('-load_pretrained',type=str,default='',help='load a pretrained model')
parser.add_argument('-v', help="increase output verbosity", action="store_true")

args = parser.parse_args()

dataset_path=args.dataset_path
lr=args.lr
epoch=args.epoch
batch_size=args.bs
model_path=args.model_save_to
w_sr=args.w_sr
w_tf=args.w_tf
pretrained_model=args.load_pretrained

print(args)

model=PFSeg3D(in_channels=1,out_channels=1).cuda()
if pt.cuda.device_count()>1:
    if batch_size<pt.cuda.device_count():
        batch_size=pt.cuda.device_count()
        print('Batch size has to be larger than GPU#. Set to {:d} instead.'.format(batch_size))
    model=pt.nn.DataParallel(model)
if not pretrained_model=='':
    model.load_state_dict(pt.load(pretrained_model,map_location = 'cpu'))

trainset=GuidedBraTSDataset3D(dataset_path,mode='train')
valset=GuidedBraTSDataset3D(dataset_path,mode='val')
testset=GuidedBraTSDataset3D(dataset_path,mode='test')

train_dataset=pt.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,drop_last=True)
val_dataset=pt.utils.data.DataLoader(valset,batch_size=1,shuffle=True,drop_last=True)
test_dataset=pt.utils.data.DataLoader(testset,batch_size=1,shuffle=True,drop_last=True)

lossfunc_sr=pt.nn.MSELoss()
lossfunc_seg=pt.nn.BCELoss()
lossfunc_dice=BinaryDiceLoss()
lossfunc_pf=TaskFusionLoss()
optimizer = pt.optim.Adam(model.parameters(), lr=lr)
# # scheduler = pt.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
scheduler=pt.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',patience=20)

def ValModel():
    model.eval()
    dice_sum=0
    hd_sum=0
    jc_sum=0
    weight_map=np.zeros((1,1,2*img_size[0],2*img_size[1],2*img_size[2]))
    for a in range(0,img_size[0]-crop_size[0]+1,crop_size[0]//2):   # overlap0.5
        for b in range(0,img_size[1]-crop_size[1]+1,crop_size[1]//2):
            for c in range(0,img_size[2]-crop_size[2]+1,crop_size[2]//2):
                weight_map[:,:,(2*a):(2*(a+crop_size[0])),(2*b):(2*(b+crop_size[1])),(2*c):(2*(c+crop_size[2]))]+=1

    weight_map=1./weight_map
    for i,data in enumerate(val_dataset):
        output_list=np.zeros((1,1,2*img_size[0],2*img_size[1],2*img_size[2]))
        label_list=np.zeros((1,1,2*img_size[0],2*img_size[1],2*img_size[2]))

        (inputs,labels,_,guidance,mask)=data
        labels3D = pt.autograd.Variable(labels).type(pt.FloatTensor).cuda().unsqueeze(1)
        guidance = pt.autograd.Variable(guidance).type(pt.FloatTensor).cuda().unsqueeze(1)
        mask = pt.autograd.Variable(mask).type(pt.FloatTensor).cuda().unsqueeze(1)
        for a in range(0,img_size[0]-crop_size[0]+1,crop_size[0]//2):   # overlap0.5
            for b in range(0,img_size[1]-crop_size[1]+1,crop_size[1]//2):
                for c in range(0,img_size[2]-crop_size[2]+1,crop_size[2]//2):
                    inputs3D = pt.autograd.Variable(inputs[:,a:(a+crop_size[0]),b:(b+crop_size[1]),c:(c+crop_size[2])]).type(pt.FloatTensor).cuda().unsqueeze(1)
                    with pt.no_grad():
                        outputs3D,_ = model(inputs3D,guidance)
                    outputs3D=np.array(outputs3D.cpu().data.numpy())
                    output_list[:,:,(2*a):(2*(a+crop_size[0])),(2*b):(2*(b+crop_size[1])),(2*c):(2*(c+crop_size[2]))]+=outputs3D

        label_list=np.array(labels3D.cpu().data.numpy())
        output_list=np.array(output_list)*weight_map

        output_list[output_list<0.5]=0
        output_list[output_list>=0.5]=1
        
        pr_sum = output_list.sum()
        gt_sum = label_list.sum()
        pr_gt_sum = np.sum(output_list[label_list == 1])
        dice = 2 * pr_gt_sum / (pr_sum + gt_sum)
        dice_sum += dice

        if args.v:
            final_img=np.zeros(shape=(2*img_size[1],2*2*img_size[2]))
            final_img[:,:2*img_size[2]]=output_list[0,0,64,:,:]*255
            final_img[:,2*img_size[2]:]=label_list[0,0,64,:,:]*255
            cv2.imwrite('ValPhase_BraTS.png',final_img)
            print("dice:",dice)

        hausdorff=hd95(output_list.squeeze(0).squeeze(0),label_list.squeeze(0).squeeze(0))
        jaccard=jc(output_list.squeeze(0).squeeze(0),label_list.squeeze(0).squeeze(0))

        hd_sum+=hausdorff
        jc_sum+=jaccard

    print("Finished. Total dice: ",dice_sum/len(val_dataset),'\n')
    print("Finished. Avg Jaccard: ",jc_sum/len(val_dataset))
    print("Finished. Avg hausdorff: ",hd_sum/len(val_dataset))
    return dice_sum/len(val_dataset)


def TestModel():
    model.eval()
    dice_sum=0
    hd_sum=0
    jc_sum=0
    weight_map=np.zeros((1,1,2*img_size[0],2*img_size[1],2*img_size[2]))
    for a in range(0,img_size[0]-crop_size[0]+1,crop_size[0]//2):   # overlap0.5
        for b in range(0,img_size[1]-crop_size[1]+1,crop_size[1]//2):
            for c in range(0,img_size[2]-crop_size[2]+1,crop_size[2]//2):
                weight_map[:,:,(2*a):(2*(a+crop_size[0])),(2*b):(2*(b+crop_size[1])),(2*c):(2*(c+crop_size[2]))]+=1

    weight_map=1./weight_map
    for i,data in enumerate(test_dataset):
        output_list=np.zeros((1,1,2*img_size[0],2*img_size[1],2*img_size[2]))
        label_list=np.zeros((1,1,2*img_size[0],2*img_size[1],2*img_size[2]))

        (inputs,labels,_,guidance,mask)=data
        labels3D = pt.autograd.Variable(labels).type(pt.FloatTensor).cuda().unsqueeze(1)
        guidance = pt.autograd.Variable(guidance).type(pt.FloatTensor).cuda().unsqueeze(1)
        mask = pt.autograd.Variable(mask).type(pt.FloatTensor).cuda().unsqueeze(1)
        for a in range(0,img_size[0]-crop_size[0]+1,crop_size[0]//2):   # overlap0.5
            for b in range(0,img_size[1]-crop_size[1]+1,crop_size[1]//2):
                for c in range(0,img_size[2]-crop_size[2]+1,crop_size[2]//2):
                    inputs3D = pt.autograd.Variable(inputs[:,a:(a+crop_size[0]),b:(b+crop_size[1]),c:(c+crop_size[2])]).type(pt.FloatTensor).cuda().unsqueeze(1)
                    with pt.no_grad():
                        outputs3D,_ = model(inputs3D,guidance)
                    outputs3D=np.array(outputs3D.cpu().data.numpy())
                    output_list[:,:,(2*a):(2*(a+crop_size[0])),(2*b):(2*(b+crop_size[1])),(2*c):(2*(c+crop_size[2]))]+=outputs3D

        label_list=np.array(labels3D.cpu().data.numpy())

        output_list=np.array(output_list)*weight_map

        output_list[output_list<0.5]=0
        output_list[output_list>=0.5]=1

        final_img=np.zeros(shape=(2*img_size[1],2*2*img_size[2]))
        final_img[:,:2*img_size[2]]=output_list[0,0,64,:,:]*255
        final_img[:,2*img_size[2]:]=label_list[0,0,64,:,:]*255
        cv2.imwrite('TestPhase_BraTS.png',final_img)
        
        pr_sum = output_list.sum()
        gt_sum = label_list.sum()
        pr_gt_sum = np.sum(output_list[label_list == 1])
        dice = 2 * pr_gt_sum / (pr_sum + gt_sum)
        dice_sum += dice

        hausdorff=hd95(output_list.squeeze(0).squeeze(0),label_list.squeeze(0).squeeze(0))
        jaccard=jc(output_list.squeeze(0).squeeze(0),label_list.squeeze(0).squeeze(0))

        hd_sum+=hausdorff
        jc_sum+=jaccard

    print("Finished. Test Total dice: ",dice_sum/len(test_dataset),'\n')
    print("Finished. Test Avg Jaccard: ",jc_sum/len(test_dataset))
    print("Finished. Test Avg hausdorff: ",hd_sum/len(test_dataset))
    return dice_sum/len(test_dataset)

best_dice=0
iterator=tqdm(train_dataset, ncols=100)
for x in range(epoch):
    model.train()
    loss_sum=0
    print('\n==>Epoch',x,': lr=',optimizer.param_groups[0]['lr'],'==>\n')

    for data in iterator:
        (inputs,labels_seg,labels_sr,guidance,mask)=data
        optimizer.zero_grad()

        inputs = pt.autograd.Variable(inputs).type(pt.FloatTensor).cuda().unsqueeze(1)
        guidance = pt.autograd.Variable(guidance).type(pt.FloatTensor).cuda().unsqueeze(1)
        mask = pt.autograd.Variable(mask).type(pt.FloatTensor).cuda().unsqueeze(1)
        labels_seg = pt.autograd.Variable(labels_seg).type(pt.FloatTensor).cuda().unsqueeze(1)
        labels_sr = pt.autograd.Variable(labels_sr).type(pt.FloatTensor).cuda().unsqueeze(1)
        outputs_seg,outputs_sr = model(inputs,guidance)
        loss_seg = lossfunc_seg(outputs_seg, labels_seg)
        loss_sr = lossfunc_sr(outputs_sr, labels_sr)
        loss_pf = lossfunc_pf(outputs_seg,outputs_sr,labels_seg*labels_sr)
        loss_guide=lossfunc_sr(mask*outputs_sr,mask*labels_sr)

        loss=lossfunc_dice(outputs_seg,labels_seg)+loss_seg+w_sr*(loss_sr+loss_guide)+w_tf*loss_pf

        loss.backward()
        optimizer.step()

        loss_sum+=loss.item()

        if args.v:
            final_img=np.zeros(shape=(2*size,2*size*5))
            iterator.set_postfix(loss=loss.item(),loss_seg=loss_seg.item(),loss_sr=loss_sr.item())
            final_img[:,0:(2*size)]=outputs_seg.cpu().data.numpy()[0,0,size//2,:,:]*255
            final_img[:,(2*size):(4*size)]=outputs_sr.cpu().data.numpy()[0,0,size//2,:,:]*255
            final_img[:,(4*size):(6*size)]=labels_seg.cpu().data.numpy()[0,0,size//2,:,:]*255
            final_img[:,(6*size):(8*size)]=labels_sr.cpu().data.numpy()[0,0,size//2,:,:]*255
            final_img[:,(8*size):]=cv2.resize(inputs.cpu().data.numpy()[0,0,size//4,:,:],((2*size),(2*size)))*255
            cv2.imwrite('combine.png',final_img)

    print('==>End of epoch',x,'==>\n')

    print('===VAL===>')
    dice=ValModel()
    scheduler.step(dice)
    if dice>best_dice:
        best_dice=dice
        print('New best dice! Model saved to',model_path+'/PFSeg_3D_BraTS_patch-free_bs'+str(batch_size)+'_best.pt')
        pt.save(model.state_dict(), model_path+'/PFSeg_3D_BraTS_patch-free_bs'+str(batch_size)+'_best.pt')
        print('===TEST===>')
        TestModel()

print('\nBest Dice:',best_dice)