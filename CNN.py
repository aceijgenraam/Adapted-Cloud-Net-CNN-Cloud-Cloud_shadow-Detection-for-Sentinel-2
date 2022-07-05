# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 17:06:49 2021

@author: Bram
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
#%% model part 1

def improve_ff_block4(input_tensor1, input_tensor2 ,input_tensor3, input_tensor4, pure_ff):
    """It improves the skip connection by using previous layers feature maps
       TO DO: shrink all of ff blocks in one function/class
    """

    for ix in range(1):
        if ix == 0:
            x1 = input_tensor1
        x1 = torch.cat([x1, input_tensor1], axis=1)
    pool=nn.MaxPool2d(2)
    x1 = pool(x1)

    for ix in range(3):
        if ix == 0:
            x2 = input_tensor2
        x2 = torch.cat([x2, input_tensor2], axis=1)
    pool=nn.MaxPool2d(4)
    x2 = pool(x2)

    for ix in range(7):
        if ix == 0:
            x3 = input_tensor3
        x3 = torch.cat([x3, input_tensor3], axis=1)
    pool=nn.MaxPool2d(8)
    x3 = pool(x3)

    for ix in range(15):
        if ix == 0:
            x4 = input_tensor4
        x4 = torch.cat([x4, input_tensor4], axis=1)
    pool=nn.MaxPool2d(16)
    x4 = pool(x4)

    x_final=F.relu(x1+x2+x3+x4+pure_ff)
    return x_final

def improve_ff_block3(input_tensor1, input_tensor2 ,input_tensor3, pure_ff):
    """It improves the skip connection by using previous layers feature maps
       TO DO: shrink all of ff blocks in one function/class
    """

    for ix in range(1):
        if ix == 0:
            x1 = input_tensor1
        x1 = torch.cat([x1, input_tensor1], axis=1)
    pool=nn.MaxPool2d(2)
    x1 = pool(x1)

    for ix in range(3):
        if ix == 0:
            x2 = input_tensor2
        x2 = torch.cat([x2, input_tensor2], axis=1)
    pool=nn.MaxPool2d(4)
    x2 = pool(x2)

    for ix in range(7):
        if ix == 0:
            x3 = input_tensor3
        x3 = torch.cat([x3, input_tensor3], axis=1)
    pool=nn.MaxPool2d(8)
    x3 = pool(x3)

    x_final=F.relu(x1+x2+x3+pure_ff)
    return x_final

def improve_ff_block2(input_tensor1, input_tensor2 , pure_ff):
    """It improves the skip connection by using previous layers feature maps
       TO DO: shrink all of ff blocks in one function/class
    """

    for ix in range(1):
        if ix == 0:
            x1 = input_tensor1
        x1 = torch.cat([x1, input_tensor1], axis=1)
    pool=nn.MaxPool2d(2)
    x1 = pool(x1)

    for ix in range(3):
        if ix == 0:
            x2 = input_tensor2
        x2 = torch.cat([x2, input_tensor2], axis=1)
    pool=nn.MaxPool2d(4)
    x2 = pool(x2)

    x_final=F.relu(x1+x2+pure_ff)
    return x_final

def improve_ff_block1(input_tensor1, pure_ff):
    """It improves the skip connection by using previous layers feature maps
       TO DO: shrink all of ff blocks in one function/class
    """

    for ix in range(1):
        if ix == 0:
            x1 = input_tensor1
        x1 = torch.cat([x1, input_tensor1], axis=1)
    pool=nn.MaxPool2d(2)
    x1 = pool(x1)

    x_final=F.relu(x1+pure_ff)
    return x_final     
#%% model part 2_1

class Nett(nn.Module):    
    def __init__(self):
        super(Nett,self).__init__()          
        self.conv1=nn.Conv2d(4, 16,kernel_size=3,padding=(3-1)//2)
        
        self.conv2=nn.Conv2d(16, 32, 3,stride=1,padding=(3-1)//2)
        self.conv3=nn.Conv2d(32, 32, 3, stride=1,padding=(3-1)//2)
        self.conv4=nn.Conv2d(16, 32//2,kernel_size=(3-2), stride=1,padding=(3-2-1)//2)
  
        self.conv5=nn.Conv2d(32, 64, 3,stride=1,padding=(3-1)//2)
        self.conv6=nn.Conv2d(64, 64,3, stride=1,padding=(3-1)//2)
        self.conv7=nn.Conv2d(32, 64//2,kernel_size=(3-2), stride=1,padding=(3-2-1)//2)

        self.conv8=nn.Conv2d(64, 128, 3,stride=1,padding=(3-1)//2)
        self.conv9=nn.Conv2d(128, 128,3, stride=1,padding=(3-1)//2)
        self.conv10=nn.Conv2d(64, 128//2,kernel_size=(3-2), stride=1,padding=(3-2-1)//2)
        
        self.conv11=nn.Conv2d(128, 256, 3,stride=1,padding=(3-1)//2)
        self.conv12=nn.Conv2d(256, 256,3, stride=1,padding=(3-1)//2)
        self.conv13=nn.Conv2d(128, 256//2,kernel_size=(3-2), stride=1,padding=(3-2-1)//2)
        
        self.conv14=nn.Conv2d(256, 512,3, stride=1,padding=(3-1)//2)
        self.conv15=nn.Conv2d(512, 512,3, stride=1,padding=(3-1)//2)
        self.conv16=nn.Conv2d(512, 512,3, stride=1,padding=(3-1)//2)
        self.conv17=nn.Conv2d(256, 512//2,kernel_size=(3-2), stride=1,padding=(3-2-1)//2)
        self.conv18=nn.Conv2d(512, 512,kernel_size=(3-2), stride=1,padding=(3-2-1)//2)
    
        self.conv19=nn.Conv2d(512, 1024,3, stride=1,padding=(3-1)//2)
        self.conv20=nn.Conv2d(1024, 1024,3, stride=1,padding=(3-1)//2)
        self.drop=nn.Dropout2d(p=0.15)
        self.conv21=nn.Conv2d(512, 1024//2,kernel_size=(3-2), stride=1,padding=(3-2-1)//2)
    
        self.conv22=nn.ConvTranspose2d(1024, 512, kernel_size=2,stride=(2,2),padding=(2-1)//2)
        self.conv23=nn.Conv2d(1024, 512,3, stride=1,padding=(3-1)//2)
        self.conv24=nn.Conv2d(512, 512,3, stride=1,padding=(3-1)//2)
        self.conv25=nn.Conv2d(512, 512,3, stride=1,padding=(3-1)//2)

        self.conv26=nn.ConvTranspose2d(512, 256, kernel_size=2,stride=(2,2),padding=(2-1)//2)
        self.conv27=nn.Conv2d(512, 256,3, stride=1,padding=(3-1)//2)
        self.conv28=nn.Conv2d(256, 256,3, stride=1,padding=(3-1)//2)

        self.conv29=nn.ConvTranspose2d(256, 128, kernel_size=2,stride=(2,2),padding=(2-1)//2)
        self.conv30=nn.Conv2d(256, 128,3, stride=1,padding=(3-1)//2)
        self.conv31=nn.Conv2d(128, 128,3, stride=1,padding=(3-1)//2)

        self.conv32=nn.ConvTranspose2d(128, 64, kernel_size=2,stride=(2,2),padding=(2-1)//2)
        self.conv33=nn.Conv2d(128, 64,3, stride=1,padding=(3-1)//2)
        self.conv34=nn.Conv2d(64, 64,3, stride=1,padding=(3-1)//2)        

        self.conv35=nn.ConvTranspose2d(64, 32, kernel_size=2,stride=(2,2),padding=(2-1)//2)
        self.conv36=nn.Conv2d(64, 32,3, stride=1,padding=(3-1)//2)
        self.conv37=nn.Conv2d(32, 32,3, stride=1,padding=(3-1)//2)

        self.conv38=nn.Conv2d(32, 1,kernel_size=1)
        
    def forward(self,x):
        x_0=F.relu(self.conv1(x))
        pool=nn.MaxPool2d(2)   
        
        x_1_1=F.relu(self.conv2(x_0))
        x_1_1=F.relu(self.conv3(x_1_1))
        x_1_2=F.relu(self.conv4(x_0))
        x_1_2 = torch.cat([x_0, x_1_2], axis=1)
        x_1=F.relu(x_1_1+x_1_2)
        pool_1=pool(x_1)
        
        x_2_1=F.relu(self.conv5(pool_1))
        x_2_1=F.relu(self.conv6(x_2_1))
        x_2_2=F.relu(self.conv7(pool_1))
        x_2_2 = torch.cat([pool_1, x_2_2], axis=1)
        x_2=F.relu(x_2_1+x_2_2)
        pool_2=pool(x_2)
        
        x_3_1=F.relu(self.conv8(pool_2))
        x_3_1=F.relu(self.conv9(x_3_1))
        x_3_2=F.relu(self.conv10(pool_2))
        x_3_2 = torch.cat([pool_2, x_3_2], axis=1)
        x_3=F.relu(x_3_1+x_3_2)
        pool_3=pool(x_3)
        
        x_4_1=F.relu(self.conv11(pool_3))
        x_4_1=F.relu(self.conv12(x_4_1))
        x_4_2=F.relu(self.conv13(pool_3))
        x_4_2 = torch.cat([pool_3, x_4_2], axis=1)
        x_4=F.relu(x_4_1+x_4_2)
        pool_4=pool(x_4)
        
        x_5_1=F.relu((self.conv14(pool_4)))
        x_5_2=F.relu((self.conv15(x_5_1)))
        x_5_1=F.relu((self.conv16(x_5_2)))
        x_5_3=F.relu((self.conv17(pool_4)))
        x_5_3 = torch.cat([pool_4, x_5_3], axis=1)
        x_5_4=F.relu((self.conv18(x_5_2)))
        x_5=F.relu(x_5_1+x_5_3+x_5_4)
        pool_5=pool(x_5)
        
        x_6_1=F.relu((self.conv19(pool_5)))
        x_6_1=F.relu(self.drop(self.conv20(x_6_1)))
        x_6_2=F.relu((self.conv21(pool_5)))
        x_6_2 = torch.cat([pool_5, x_6_2], axis=1)
        x_6=F.relu(x_6_1+x_6_2)
        
        x_T7=self.conv22(x_6)
        prevup7 = improve_ff_block4(input_tensor1=x_4, input_tensor2=x_3, input_tensor3=x_2, input_tensor4=x_1, pure_ff=x_5)
        up7 = torch.cat([x_T7, prevup7], axis=1)
        x_7_1=F.relu((self.conv23(up7)))
        x_7_1=F.relu((self.conv24(x_7_1)))
        x_7_1=F.relu((self.conv25(x_7_1)))
        x_7 = F.relu(x_T7+ x_7_1+ x_5)
        
        x_T8=self.conv26(x_7)
        prevup8 = improve_ff_block3(input_tensor1=x_3, input_tensor2=x_2, input_tensor3=x_1, pure_ff=x_4)
        up8 = torch.cat([x_T8, prevup8], axis=1)
        x_8_1=F.relu((self.conv27(up8)))
        x_8_1=F.relu((self.conv28(x_8_1)))
        x_8 = F.relu(x_T8+ x_8_1+ x_4)
        
        x_T9=self.conv29(x_8)
        prevup9 =improve_ff_block2(input_tensor1=x_2, input_tensor2=x_1, pure_ff=x_3)
        up9 = torch.cat([x_T9, prevup9], axis=1)
        x_9_1=F.relu((self.conv30(up9)))
        x_9_1=F.relu((self.conv31(x_9_1)))
        x_9 = F.relu(x_T9+ x_9_1+ x_3)
        
        x_T10=self.conv32(x_9)
        prevup10 = improve_ff_block1(input_tensor1=x_1, pure_ff=x_2)
        up10 = torch.cat([x_T10, prevup10], axis=1)
        x_10_1=F.relu((self.conv33(up10)))
        x_10_1=F.relu((self.conv34(x_10_1)))
        x_10 = F.relu(x_T10+ x_10_1+ x_2)
        
        x_T11=self.conv35(x_10)
        up11 = torch.cat([x_T11, x_1], axis=1)
        x_11_1=F.relu((self.conv36(up11)))
        x_11_1=F.relu((self.conv37(x_11_1)))
        x_11 = F.relu(x_T11+ x_11_1+ x_1)
        
        pred_final=torch.sigmoid(self.conv38(x_11))
        #pred_final=pred_final.permute(0,2,3,1)
        #pred_final=torch.reshape(pred_final,(Batches*X_training.shape[2]*X_training.shape[3],1))
        return pred_final    

model_clouds = Nett()

#%% model part 2_2

class Nett(nn.Module):    
    def __init__(self):
        super(Nett,self).__init__()          
        self.conv1=nn.Conv2d(6, 16,kernel_size=3,padding=(3-1)//2)
        
        self.conv2=nn.Conv2d(16, 32, 3,stride=1,padding=(3-1)//2)
        self.conv3=nn.Conv2d(32, 32, 3, stride=1,padding=(3-1)//2)
        self.conv4=nn.Conv2d(16, 32//2,kernel_size=(3-2), stride=1,padding=(3-2-1)//2)
  
        self.conv5=nn.Conv2d(32, 64, 3,stride=1,padding=(3-1)//2)
        self.conv6=nn.Conv2d(64, 64,3, stride=1,padding=(3-1)//2)
        self.conv7=nn.Conv2d(32, 64//2,kernel_size=(3-2), stride=1,padding=(3-2-1)//2)

        self.conv8=nn.Conv2d(64, 128, 3,stride=1,padding=(3-1)//2)
        self.conv9=nn.Conv2d(128, 128,3, stride=1,padding=(3-1)//2)
        self.conv10=nn.Conv2d(64, 128//2,kernel_size=(3-2), stride=1,padding=(3-2-1)//2)
        
        self.conv11=nn.Conv2d(128, 256, 3,stride=1,padding=(3-1)//2)
        self.conv12=nn.Conv2d(256, 256,3, stride=1,padding=(3-1)//2)
        self.conv13=nn.Conv2d(128, 256//2,kernel_size=(3-2), stride=1,padding=(3-2-1)//2)
        
        self.conv14=nn.Conv2d(256, 512,3, stride=1,padding=(3-1)//2)
        self.conv15=nn.Conv2d(512, 512,3, stride=1,padding=(3-1)//2)
        self.conv16=nn.Conv2d(512, 512,3, stride=1,padding=(3-1)//2)
        self.conv17=nn.Conv2d(256, 512//2,kernel_size=(3-2), stride=1,padding=(3-2-1)//2)
        self.conv18=nn.Conv2d(512, 512,kernel_size=(3-2), stride=1,padding=(3-2-1)//2)
    
        self.conv19=nn.Conv2d(512, 1024,3, stride=1,padding=(3-1)//2)
        self.conv20=nn.Conv2d(1024, 1024,3, stride=1,padding=(3-1)//2)
        self.drop=nn.Dropout2d(p=0.15)
        self.conv21=nn.Conv2d(512, 1024//2,kernel_size=(3-2), stride=1,padding=(3-2-1)//2)
    
        self.conv22=nn.ConvTranspose2d(1024, 512, kernel_size=2,stride=(2,2),padding=(2-1)//2)
        self.conv23=nn.Conv2d(1024, 512,3, stride=1,padding=(3-1)//2)
        self.conv24=nn.Conv2d(512, 512,3, stride=1,padding=(3-1)//2)
        self.conv25=nn.Conv2d(512, 512,3, stride=1,padding=(3-1)//2)

        self.conv26=nn.ConvTranspose2d(512, 256, kernel_size=2,stride=(2,2),padding=(2-1)//2)
        self.conv27=nn.Conv2d(512, 256,3, stride=1,padding=(3-1)//2)
        self.conv28=nn.Conv2d(256, 256,3, stride=1,padding=(3-1)//2)

        self.conv29=nn.ConvTranspose2d(256, 128, kernel_size=2,stride=(2,2),padding=(2-1)//2)
        self.conv30=nn.Conv2d(256, 128,3, stride=1,padding=(3-1)//2)
        self.conv31=nn.Conv2d(128, 128,3, stride=1,padding=(3-1)//2)

        self.conv32=nn.ConvTranspose2d(128, 64, kernel_size=2,stride=(2,2),padding=(2-1)//2)
        self.conv33=nn.Conv2d(128, 64,3, stride=1,padding=(3-1)//2)
        self.conv34=nn.Conv2d(64, 64,3, stride=1,padding=(3-1)//2)        

        self.conv35=nn.ConvTranspose2d(64, 32, kernel_size=2,stride=(2,2),padding=(2-1)//2)
        self.conv36=nn.Conv2d(64, 32,3, stride=1,padding=(3-1)//2)
        self.conv37=nn.Conv2d(32, 32,3, stride=1,padding=(3-1)//2)

        self.conv38=nn.Conv2d(32, 1,kernel_size=1)
        
    def forward(self,x):
        x_0=F.relu(self.conv1(x))
        pool=nn.MaxPool2d(2)   
        
        x_1_1=F.relu(self.conv2(x_0))
        x_1_1=F.relu(self.conv3(x_1_1))
        x_1_2=F.relu(self.conv4(x_0))
        x_1_2 = torch.cat([x_0, x_1_2], axis=1)
        x_1=F.relu(x_1_1+x_1_2)
        pool_1=pool(x_1)
        
        x_2_1=F.relu(self.conv5(pool_1))
        x_2_1=F.relu(self.conv6(x_2_1))
        x_2_2=F.relu(self.conv7(pool_1))
        x_2_2 = torch.cat([pool_1, x_2_2], axis=1)
        x_2=F.relu(x_2_1+x_2_2)
        pool_2=pool(x_2)
        
        x_3_1=F.relu(self.conv8(pool_2))
        x_3_1=F.relu(self.conv9(x_3_1))
        x_3_2=F.relu(self.conv10(pool_2))
        x_3_2 = torch.cat([pool_2, x_3_2], axis=1)
        x_3=F.relu(x_3_1+x_3_2)
        pool_3=pool(x_3)
        
        x_4_1=F.relu(self.conv11(pool_3))
        x_4_1=F.relu(self.conv12(x_4_1))
        x_4_2=F.relu(self.conv13(pool_3))
        x_4_2 = torch.cat([pool_3, x_4_2], axis=1)
        x_4=F.relu(x_4_1+x_4_2)
        pool_4=pool(x_4)
        
        x_5_1=F.relu((self.conv14(pool_4)))
        x_5_2=F.relu((self.conv15(x_5_1)))
        x_5_1=F.relu((self.conv16(x_5_2)))
        x_5_3=F.relu((self.conv17(pool_4)))
        x_5_3 = torch.cat([pool_4, x_5_3], axis=1)
        x_5_4=F.relu((self.conv18(x_5_2)))
        x_5=F.relu(x_5_1+x_5_3+x_5_4)
        pool_5=pool(x_5)
        
        x_6_1=F.relu((self.conv19(pool_5)))
        x_6_1=F.relu(self.drop(self.conv20(x_6_1)))
        x_6_2=F.relu((self.conv21(pool_5)))
        x_6_2 = torch.cat([pool_5, x_6_2], axis=1)
        x_6=F.relu(x_6_1+x_6_2)
        
        x_T7=self.conv22(x_6)
        prevup7 = improve_ff_block4(input_tensor1=x_4, input_tensor2=x_3, input_tensor3=x_2, input_tensor4=x_1, pure_ff=x_5)
        up7 = torch.cat([x_T7, prevup7], axis=1)
        x_7_1=F.relu((self.conv23(up7)))
        x_7_1=F.relu((self.conv24(x_7_1)))
        x_7_1=F.relu((self.conv25(x_7_1)))
        x_7 = F.relu(x_T7+ x_7_1+ x_5)
        
        x_T8=self.conv26(x_7)
        prevup8 = improve_ff_block3(input_tensor1=x_3, input_tensor2=x_2, input_tensor3=x_1, pure_ff=x_4)
        up8 = torch.cat([x_T8, prevup8], axis=1)
        x_8_1=F.relu((self.conv27(up8)))
        x_8_1=F.relu((self.conv28(x_8_1)))
        x_8 = F.relu(x_T8+ x_8_1+ x_4)
        
        x_T9=self.conv29(x_8)
        prevup9 =improve_ff_block2(input_tensor1=x_2, input_tensor2=x_1, pure_ff=x_3)
        up9 = torch.cat([x_T9, prevup9], axis=1)
        x_9_1=F.relu((self.conv30(up9)))
        x_9_1=F.relu((self.conv31(x_9_1)))
        x_9 = F.relu(x_T9+ x_9_1+ x_3)
        
        x_T10=self.conv32(x_9)
        prevup10 = improve_ff_block1(input_tensor1=x_1, pure_ff=x_2)
        up10 = torch.cat([x_T10, prevup10], axis=1)
        x_10_1=F.relu((self.conv33(up10)))
        x_10_1=F.relu((self.conv34(x_10_1)))
        x_10 = F.relu(x_T10+ x_10_1+ x_2)
        
        x_T11=self.conv35(x_10)
        up11 = torch.cat([x_T11, x_1], axis=1)
        x_11_1=F.relu((self.conv36(up11)))
        x_11_1=F.relu((self.conv37(x_11_1)))
        x_11 = F.relu(x_T11+ x_11_1+ x_1)
        
        pred_final=torch.sigmoid(self.conv38(x_11))
        #pred_final=pred_final.permute(0,2,3,1)
        #pred_final=torch.reshape(pred_final,(Batches*X_training.shape[2]*X_training.shape[3],1))
        return pred_final    

model_shadows = Nett()