from tkinter import END
# from train import fitness#后续要修改！！！！！！！！！！！
# import pandas as pd
import numpy as np
# from numba import jit
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
from math import ceil
import random
# ops={}
# aa=lambda stride, affine: nn.MaxPool2d(3, stride=stride, padding=1)
# for i in range(1,10):
#     # ops[i]= nn.MaxPool2d(3, stride=1, padding=1)
#     ops[i]=lambda stride, affine: nn.MaxPool2d(3, stride=stride, padding=1)
#     # ops[i]=lambda stride, affine: nn.Conv2d(3, stride=stride, padding=1),

# #查询匿名参数的参数个数方法
# ops[1].__code__.co_argcount
# #查询匿名参数的参数名称
# ops[1].__code__.co_varnames
# @jit()
#由于没有办法使用匿名函数的方式进行reshape操作，因此定义一个Reshape类来帮助操作
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.size(0),)+self.shape)


a={}
ops={}
OPS={}
K=1
stride_set=(range(1,7))#ending不可达
padding_set=(range(0,4))#ending不可达
C_out_set=(range(1,33))#ending不可达
kernel_size_set=(range(1,6,2))
for ks in kernel_size_set:
  for cout in C_out_set:
    for pad in range(0,ks):
      for st in range(1,ks+1):
        # for dila in range(1,3):
          ops[K]=((lambda C_in,cout=cout,st=st,pad=pad,ks=ks:nn.Conv2d(in_channels=C_in,out_channels=cout  ,kernel_size=ks, stride=st, padding= pad)))
          # if cout==2 and st==3 and pad==2 and ks==3:
          #   print(K)
          K+=1
k_linear=K
in_features_set=range(100,301)
out_features_set=range(100,1800)
# for inft in in_features_set:
for outft in out_features_set:
    ops[K]=lambda in_f,outft=outft:nn.Linear(in_features=in_f, out_features=outft)
    # if outft==800:
    #        print(K)
    K+=1
  
# 'linear' in str(ops[20000].__class__)
#先产生一个随机个体，试一下对其解码的方式
##结构记录317，1，797
#50,1446
#2,1446

p = np.hstack((np.ones([1,k_linear])*0.5/k_linear ,np.ones([1,K-k_linear-1])*0.5/(K-k_linear-1)))
p=p[0]

# linear_line=5000
# batch_size=2
# tupianchicun=70
# batch_size,
# tupianchicun,控制输入图尺寸
# linear_line,卷积后能接受的最大全连接层数。
# shuru_c,控制输入通道数
# max_output_linear_size，全连接层的最大输出尺寸。
#out_cc,out_size_1,out_size_2,输出图通道，尺寸1，尺寸2
def getmodel(X,batch_size,tupianchicun1,tupianchicun2,linear_line,shuru_c,max_output_linear_size,out_cc,out_size_1,out_size_2,mid_c,mid_size1,mid_size2):
  #                 [2,          70,             73,        8000,    256,        5000,              2,          35,    35,       2,    70,          73]
  model={}
  Y=[]
  Result_name=[]
  X=[X]
  for i in range(len(X)):
    data_input = Variable(torch.randn([batch_size, shuru_c, tupianchicun1, tupianchicun2])) # 这里假设输入图片是96x96
    oux=data_input
  
    model[i]=nn.ModuleList()#[ops[X[i][0]]]
    input_c=shuru_c
    temp_cin=input_c
    temp_fin=shuru_c*tupianchicun1*tupianchicun2
    #判断是线性层，还是卷积层
    if 'in_f' in ops[X[i][0]].__code__.co_varnames:
        n_oux=copy.deepcopy(oux)
        linear_n=shuru_c*tupianchicun1*tupianchicun2

 ##########################################################################################################################################################################
        oux=nn.Conv2d(temp_cin,ceil(temp_cin/4),1,1)(oux)###需要计算卷积操作输出的尺寸
        model[i].append(nn.Conv2d(temp_cin,ceil(temp_cin/4),1,1)) ###需要记录这个卷积操作
        temp_cin = ceil(temp_cin/4)
 ##########################################################################################################################################################################
        #在卷积操作降维之后，使用线性层。此时需要计算通道数

        temp_x=oux
        model[i].append(nn.Flatten())
          # model[i].append(nn.Linear(in_features=oux.size()[1]*oux.size()[2]*oux.size()[3], out_features=oux.size()[1]*oux.size()[2]*oux.size()[3]))
        model[i].append(nn.Linear(temp_x.size()[1]*temp_x.size()[2]*temp_x.size()[3],max_output_linear_size))
        model[i].append(ops[X[i][0]](in_f=max_output_linear_size))


        oux=nn.Flatten()(oux)
        oux=nn.Linear(temp_x.size()[1]*temp_x.size()[2]*temp_x.size()[3],max_output_linear_size)(oux)
        oux=ops[X[i][0]](in_f=max_output_linear_size)(oux)

        temp_fin=ops[X[i][0]](in_f=max_output_linear_size).out_features
    else:
        model[i].append(ops[X[i][0]](C_in=input_c))
        model[i].append(nn.ReLU(inplace=True))
        oux=ops[X[i][0]](C_in=input_c)(oux)
        oux=nn.ReLU(inplace=True)(oux)
        temp_cin=ops[X[i][0]](C_in=input_c).out_channels
    
    for index,j in enumerate(X[i][1:len(X[i][:])]):
      if 'in_f' in ops[j].__code__.co_varnames:#当前是全连接层
        if 'in_f' not in ops[X[i][index]].__code__.co_varnames:#之前是卷积层，触发修正的情况
          #由于需要计算当前全连接层的大小，因此会随机产生一个模拟数据帮助计算，当前全连接的大小。
          # data_input = Variable(torch.randn([4, 256, 60, 60])) # 这里假设输入图片是96x96

          # oux=data_input 
          # for k,mode in  enumerate(model[i]):
          #   oux=mode(oux)
##########################################################################################################################################################################
          temp_x=oux
          if temp_x.size()[1]*temp_x.size()[2]*temp_x.size()[3]>linear_line:
              oux=nn.Conv2d(temp_cin,ceil(temp_cin/4),1,1)(oux)###需要计算卷积操作输出的尺寸
              model[i].append(nn.Conv2d(temp_cin,ceil(temp_cin/4),1,1)) ###需要记录这个卷积操作
              temp_cin = ceil(temp_cin/4)
 ##########################################################################################################################################################################

          temp_x=oux
          model[i].append(nn.Flatten())
          # model[i].append(nn.Linear(in_features=oux.size()[1]*oux.size()[2]*oux.size()[3], out_features=oux.size()[1]*oux.size()[2]*oux.size()[3]))
          model[i].append(ops[j](in_f=temp_x.size()[1]*temp_x.size()[2]*temp_x.size()[3]))
          oux=nn.Flatten()(oux)
          oux=ops[j](in_f=temp_x.size()[1]*temp_x.size()[2]*temp_x.size()[3])(oux)
          temp_fin=ops[j](in_f=temp_fin).out_features
        else:

          model[i].append(ops[j](in_f=temp_fin))
          oux=ops[j](in_f=temp_fin)(oux)
          temp_fin=ops[j](in_f=temp_fin).out_features
      else:####卷积层，可能会出现当前的卷积核大于输入图片尺寸
        # if 'Linear' in str(ops[X[i][index]].type):
        if 'in_f'  in ops[X[i][index]].__code__.co_varnames:
          model[i].append(nn.Linear(in_features=temp_fin, out_features=mid_c*mid_size1*mid_size2))
          model[i].append(Reshape(mid_c,mid_size1,mid_size2))#上下的2位置的参数是需要一样的，含义是通道数（输入的）
          model[i].append(ops[j](C_in=mid_c))
          model[i].append(nn.ReLU(inplace=True))
          oux=nn.Linear(in_features=temp_fin, out_features=mid_c*mid_size1*mid_size2)(oux)
          oux=Reshape(mid_c,mid_size1,mid_size2)(oux)
          oux=ops[j](C_in=mid_c)(oux)
          oux=nn.ReLU(inplace=True)(oux)
          temp_cin=ops[j](C_in=input_c).out_channels
        else:
          try:
            oux=ops[j](C_in=temp_cin)(oux)
            model[i].append(ops[j](C_in=temp_cin))
            model[i].append(nn.ReLU(inplace=True))
            oux=nn.ReLU(inplace=True)(oux)
            temp_cin=ops[j](C_in=input_c).out_channels
          except:
            #首先是展平这个较小的特征图，然后再随机接一个全连接层，再之后再加上reshape操作，再加上原本的卷积操作。
            temp_x=oux
            temp_j=random.randint(k_linear,K-1)#随机加上的全连接层的尺寸
            model[i].append(nn.Flatten())
            model[i].append(nn.Linear(in_features=temp_x.size()[1]*temp_x.size()[2]*temp_x.size()[3], out_features=mid_c*mid_size1*mid_size2))#ops[temp_j](in_f=oux.size()[1]*oux.size()[2]*oux.size()[3])
            model[i].append(Reshape(mid_c,mid_size1,mid_size2))###可能会报错，尺寸问题！！！！！！！！
            model[i].append(ops[j](C_in=mid_c))
            model[i].append(nn.ReLU(inplace=True))
            
            oux=nn.Flatten()(oux)
            oux=nn.Linear(in_features=temp_x.size()[1]*temp_x.size()[2]*temp_x.size()[3], out_features=mid_c*mid_size1*mid_size2)(oux)#ops[temp_j](in_f=temp_x.size()[1]*temp_x.size()[2]*temp_x.size()[3])(oux)
            oux=Reshape(mid_c,mid_size1,mid_size2)(oux)
            oux=ops[j](C_in=mid_c)(oux)
            oux=nn.ReLU(inplace=True)(oux)

            temp_cin=ops[j](C_in=input_c).out_channels

    # model[i].append()#最后在加上一个输出层
    if 'ReLU' in str(model[i][len(model[i])-1].type):
##########################################################################################################################################################################
          temp_x=oux
          while temp_x.size()[1]*temp_x.size()[2]*temp_x.size()[3]>linear_line:
            oux=nn.Conv2d(temp_cin,ceil(temp_cin/4),1,1)(oux)###需要计算卷积操作输出的尺寸
            model[i].append(nn.Conv2d(temp_cin,ceil(temp_cin/4),1,1)) ###需要记录这个卷积操作
            temp_cin = ceil(temp_cin/4)
            temp_x=oux
 #######################################################################################################################################################################
          temp_x=oux         
          model[i].append(nn.Flatten())
          model[i].append(nn.Linear(in_features=temp_x.size()[1]*temp_x.size()[2]*temp_x.size()[3], out_features=out_cc*out_size_1*out_size_2))
          model[i].append(Reshape(out_cc,out_size_1,out_size_2))#上下的2位置的参数是需要一样的，含义是通道数（输入的）
    else:
          temp_x=oux
          model[i].append(nn.Linear(in_features=temp_x.size()[1], out_features=out_cc*out_size_1*out_size_2))
          model[i].append(Reshape(out_cc,out_size_1,out_size_2))#上下的2位置的参数是需要一样的，含义是通道数（输入的）
    # temp_y,result_name=fitness(model[i])
    # Y.append(copy.deepcopy(temp_y))
    # Result_name.append(str(copy.deepcopy(result_name)))
  return model#,Y,Result_name
