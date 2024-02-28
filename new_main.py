from tkinter import END
from new_train import fitness#后续要修改！！！！！！！！！！！
# import pandas as pd
import numpy as np
# from numba import jit
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
from math import ceil
import random
import sys
import os
import re
# 参数定义
N1=8   #第一级编码的个数
N2=8   #第二级编码的个数，每个一个一级编码都有N2个个体
r_max=8 #单行最大模块数
r_min=2 #单行最小模块数
G=100#算法迭代次数
N=40#种群数
row_nums=3#串行数
inc_line_k_min_k=4
crossover_rate_frist = 1
mutation_rate_frist = 0.5
mating_pool_num=30#交配池大小，需设置为偶数
kk_best= 0#保留最好的k_best个个体，剩下锦标赛选择
k_best_first = 1
stride_set=(range(1,7))#ending不可达
padding_set=(range(0,4))#ending不可达
C_out_set=(range(1,33))#ending不可达
kernel_size_set=(range(1,6,2))
out_features_set=range(100,1800)

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.size(0),)+self.shape)
class ResizeModule(nn.Module):

  def __init__(self):
    super(ResizeModule, self).__init__()

  def forward(self, x):
    out_size = (x.shape[2]*2, x.shape[3]*2)
    x_up = nn.functional.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
    return x_up

def text_save(filename, data):#filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()

def getmodel(X,batch_size,tupianchicun1,tupianchicun2,linear_line,shuru_c,max_output_linear_size,out_cc,out_size_1,out_size_2,mid_c,mid_size1,mid_size2):
  #                 [2,          70,             73,        8000,    256,        5000,              2,          35,    35,       2,    70,          73]
  conv2lin_case=0
  conv2conv_case=0
  lin2conv_case=0
  lin2lin_case=0
  
  model={}
  Y=[]
  Result_name=[]
  device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
  X=[X]
  for i in range(len(X)):
    data_input = Variable(torch.zeros(size=([batch_size, shuru_c, tupianchicun1, tupianchicun2]), device=device)) # 这里假设输入图片是96x96
    oux=data_input
  
    model[i]=nn.ModuleList()#[ops[X[i][0]]]
    model[i]=model[i].to(device)
    input_c=shuru_c
    temp_cin=input_c
    temp_fin=shuru_c*tupianchicun1*tupianchicun2
    #判断是线性层，还是卷积层
    if X[i][0]=='f':
        n_oux=copy.deepcopy(oux)
        linear_n=shuru_c*tupianchicun1*tupianchicun2

 ##########################################################################################################################################################################
        oux=nn.Conv2d(temp_cin,ceil(temp_cin/4),1,1).to(device)(oux)###需要计算卷积操作输出的尺寸
        model[i].append(nn.Conv2d(temp_cin,ceil(temp_cin/4),1,1)) ###需要记录这个卷积操作
        temp_cin = ceil(temp_cin/4)
 ##########################################################################################################################################################################
        #在卷积操作降维之后，使用线性层。此时需要计算通道数

        temp_x=oux
        model[i].append(nn.Flatten())
          # model[i].append(nn.Linear(in_features=oux.size()[1]*oux.size()[2]*oux.size()[3], out_features=oux.size()[1]*oux.size()[2]*oux.size()[3]))
        model[i].append(nn.Linear(temp_x.size()[1]*temp_x.size()[2]*temp_x.size()[3],max_output_linear_size))
        model[i].append(nn.Linear(max_output_linear_size,X[i][1]))
        # model[i].append(ops[X[i][0]](in_f=max_output_linear_size))


        oux=nn.Flatten().to(device)(oux)
        oux=nn.Linear(temp_x.size()[1]*temp_x.size()[2]*temp_x.size()[3],max_output_linear_size).to(device)(oux)
        oux=nn.Linear(max_output_linear_size,X[i][1]).to(device)(oux)
        k=2
        last_k=0
        temp_fin=X[i][1]
        
    elif X[i][0]=='c':
        model[i].append(nn.Conv2d(input_c,X[i][1],X[i][2],X[i][3],X[i][4]))
        # model[i].append(ops[X[i][0]](C_in=input_c))
        model[i].append(nn.ReLU(inplace=True))
        oux=nn.Conv2d(input_c,X[i][1],X[i][2],X[i][3],X[i][4]).to(device)(oux)
        oux=nn.ReLU(inplace=True).to(device)(oux)
        temp_cin=X[i][1]
        k=5
        last_k=0
    else:
        raise ValueError('首位非指示符')
    
    
    while k<len(X[i]):
      if X[i][k]=='f':#当前是全连接层
        if X[i][last_k]=='c':#之前是卷积层，触发修正的情况
          #由于需要计算当前全连接层的大小，因此会随机产生一个模拟数据帮助计算，当前全连接的大小。
          # data_input = Variable(torch.randn([4, 256, 60, 60])) # 这里假设输入图片是96x96

          # oux=data_input 
          # for k,mode in  enumerate(model[i]):
          #   oux=mode.to(device)(oux)
          # e 原处理
          if conv2lin_case==0:
  ##########################################################################################################################################################################
            temp_x=oux
            if temp_x.size()[1]*temp_x.size()[2]*temp_x.size()[3]>linear_line:
                oux=nn.Conv2d(temp_cin,ceil(temp_cin/4),1,1).to(device)(oux)###需要计算卷积操作输出的尺寸
                model[i].append(nn.Conv2d(temp_cin,ceil(temp_cin/4),1,1)) ###需要记录这个卷积操作
                temp_cin = ceil(temp_cin/4)
  ##########################################################################################################################################################################

            temp_x=oux
            model[i].append(nn.Flatten())
            # model[i].append(nn.Linear(in_features=oux.size()[1]*oux.size()[2]*oux.size()[3], out_features=oux.size()[1]*oux.size()[2]*oux.size()[3]))
            model[i].append(nn.Linear(temp_x.size()[1]*temp_x.size()[2]*temp_x.size()[3],X[i][k+1]))
            # model[i].append(ops[j](in_f=temp_x.size()[1]*temp_x.size()[2]*temp_x.size()[3]))
            oux=nn.Flatten().to(device)(oux)
            oux=nn.Linear(temp_x.size()[1]*temp_x.size()[2]*temp_x.size()[3],X[i][k+1]).to(device)(oux)
            temp_fin=X[i][k+1]
            last_k=k
            k+=2
          # a处理
          elif conv2lin_case==1:
            oux=nn.AdaptiveAvgPool2d(1).to(device)(oux)
            model[i].append(nn.AdaptiveAvgPool2d(1))
            model[i].append(nn.Flatten())
            temp_x=oux
            model[i].append(nn.Linear(temp_x.size()[1]*temp_x.size()[2]*temp_x.size()[3],X[i][k+1]))
            oux=nn.Flatten().to(device)(oux)
            oux=nn.Linear(temp_x.size()[1]*temp_x.size()[2]*temp_x.size()[3],X[i][k+1]).to(device)(oux)
            temp_fin=X[i][k+1]
            last_k=k
            k+=2
          # b处理
          elif conv2lin_case==2:
            model[i].append(nn.Conv2d(temp_cin, 1, [oux.size()[2],1],1,0))
            model[i].append(nn.Flatten())
            oux=nn.Conv2d(temp_cin, 1, [oux.size()[2],1],1,0).to(device)(oux)
            temp_x=oux
            oux=nn.Flatten().to(device)(oux)
            model[i].append(nn.Linear(temp_x.size()[1]*temp_x.size()[2]*temp_x.size()[3],X[i][k+1]))
            oux=nn.Linear(temp_x.size()[1]*temp_x.size()[2]*temp_x.size()[3],X[i][k+1]).to(device)(oux)
            temp_fin=X[i][k+1]
            last_k=k
            k+=2
          # c处理
          elif conv2lin_case==3:
            model[i].append(nn.Conv2d(temp_cin, 1, [1,oux.size()[3]],1,0))
            model[i].append(nn.Flatten())
            oux=nn.Conv2d(temp_cin, 1, [1,oux.size()[3]],1,0).to(device)(oux)
            temp_x=oux
            
            model[i].append(nn.Linear(temp_x.size()[1]*temp_x.size()[2]*temp_x.size()[3],X[i][k+1]))
            oux=nn.Flatten().to(device)(oux)
            oux=nn.Linear(temp_x.size()[1]*temp_x.size()[2]*temp_x.size()[3],X[i][k+1]).to(device)(oux)
            temp_fin=X[i][k+1]
            last_k=k
            k+=2
          # d处理
          elif conv2lin_case==4:
            model[i].append(nn.Conv2d(temp_cin, temp_cin, [oux.size()[2],oux.size()[3]],1,0))
            model[i].append(nn.Flatten())
            oux=nn.Conv2d(temp_cin, temp_cin, [oux.size()[2],oux.size()[3]],1,0).to(device)(oux)
            temp_x=oux
            oux=nn.Flatten().to(device)(oux)
            if oux.size()[1]==1:
                oux
            model[i].append(nn.Linear(oux.size()[1],X[i][k+1]))

            oux=nn.Linear(oux.size()[1],X[i][k+1]).to(device)(oux)
            temp_fin=X[i][k+1]
            last_k=k
            k+=2

        else:
            model[i].append(nn.Linear(temp_fin,X[i][k+1]))
            oux=nn.Linear(temp_fin,X[i][k+1]).to(device)(oux)
            temp_fin=X[i][k+1]
            last_k=k
            k+=2
      elif X[i][k]=='c':####卷积层，可能会出现当前的卷积核大于输入图片尺寸
        # if 'Linear' in str(ops[X[i][index]].type):
        if X[i][last_k]=='f':
            if lin2conv_case==0:
                model[i].append(nn.Linear(in_features=temp_fin, out_features=mid_c*mid_size1*mid_size2))
                model[i].append(Reshape(mid_c,mid_size1,mid_size2))#上下的2位置的参数是需要一样的，含义是通道数（输入的）
                model[i].append(nn.Conv2d(mid_c,X[i][k+1],X[i][k+2],X[i][k+3],X[i][k+4]))
                model[i].append(nn.ReLU(inplace=True))
                oux=nn.Linear(in_features=temp_fin, out_features=mid_c*mid_size1*mid_size2).to(device)(oux)
                oux=Reshape(mid_c,mid_size1,mid_size2).to(device)(oux)
                oux=nn.Conv2d(mid_c,X[i][k+1],X[i][k+2],X[i][k+3],X[i][k+4]).to(device)(oux)
                # ops[j](C_in=mid_c)
                oux=nn.ReLU(inplace=True).to(device)(oux)
                temp_cin=X[i][k+1]
            last_k=k
            k+=5
        else:
            try:
                temp_x=oux
                oux=nn.Conv2d(temp_cin,X[i][k+1],X[i][k+2],X[i][k+3],X[i][k+4]).to(device)(oux)
                oux=nn.ReLU(inplace=True).to(device)(oux)
                # ops[j](C_in=temp_cin).to(device)(oux)
                model[i].append(nn.Conv2d(temp_cin,X[i][k+1],X[i][k+2],X[i][k+3],X[i][k+4]))
                model[i].append(nn.ReLU(inplace=True))
                # oux=nn.ReLU(inplace=True).to(device)(oux)
                temp_cin=X[i][k+1]
                # oux=nn.functional.interpolate(oux, size=(oux.shape[2]*2, oux.shape[3]*2) , mode='bilinear', align_corners=False) 
            # if True:
            except:
                oux=temp_x
                #首先是展平这个较小的特征图，然后再随机接一个全连接层，再之后再加上reshape操作，再加上原本的卷积操作。
                if conv2conv_case==0:
                    temp_x=oux
                    # temp_j=random.randint(k_linear,K-1)#随机加上的全连接层的尺寸
                    model[i].append(nn.Flatten())
                    model[i].append(nn.Linear(in_features=temp_x.size()[1]*temp_x.size()[2]*temp_x.size()[3], out_features=mid_c*mid_size1*mid_size2))#ops[temp_j](in_f=oux.size()[1]*oux.size()[2]*oux.size()[3])
                    model[i].append(Reshape(mid_c,mid_size1,mid_size2))###可能会报错，尺寸问题！！！！！！！！
                    model[i].append(nn.Conv2d(mid_c,X[i][k+1],X[i][k+2],X[i][k+3],X[i][k+4]))
                    model[i].append(nn.ReLU(inplace=True))
                    
                    oux=nn.Flatten().to(device)(oux)
                    oux=nn.Linear(in_features=temp_x.size()[1]*temp_x.size()[2]*temp_x.size()[3], out_features=mid_c*mid_size1*mid_size2).to(device)(oux)#ops[temp_j](in_f=temp_x.size()[1]*temp_x.size()[2]*temp_x.size()[3]).to(device)(oux)
                    oux=Reshape(mid_c,mid_size1,mid_size2).to(device)(oux)
                    oux=nn.Conv2d(mid_c,X[i][k+1],X[i][k+2],X[i][k+3],X[i][k+4]).to(device)(oux)
                    oux=nn.ReLU(inplace=True).to(device)(oux)

                    temp_cin=X[i][k+1]
                elif conv2conv_case==1:
                    oux=ResizeModule().to(device)(oux)
                    model[i].append(ResizeModule())
                    while(oux.size()[2]<X[i][k+2]):
                        oux=ResizeModule().to(device)(oux)
                        model[i].append(ResizeModule())
                    oux=nn.Conv2d(temp_cin,X[i][k+1],X[i][k+2],X[i][k+3],X[i][k+4]).to(device)(oux)
                    model[i].append(nn.Conv2d(temp_cin,X[i][k+1],X[i][k+2],X[i][k+3],X[i][k+4]))
                    model[i].append(nn.ReLU(inplace=True))
                    oux=nn.ReLU(inplace=True).to(device)(oux)
                    temp_cin=X[i][k+1]
            last_k=k
            k+=5
      else:
          raise ValueError("263报错！")

    # model[i].append()#最后在加上一个输出层
    if 'ReLU' in str(model[i][len(model[i])-1].type):
##########################################################################################################################################################################
        #   temp_x=oux
          while oux.size()[1]*oux.size()[2]*oux.size()[3]>linear_line and oux.size()[1]!=1: 
            oux=nn.Conv2d(temp_cin,ceil(temp_cin/4),1,1).to(device)(oux)###需要计算卷积操作输出的尺寸
            model[i].append(nn.Conv2d(temp_cin,ceil(temp_cin/4),1,1)) ###需要记录这个卷积操作
            temp_cin = ceil(temp_cin/4)
            # temp_x=oux
 #######################################################################################################################################################################
          temp_x=oux 
          oux=nn.Flatten().to(device)(oux)        
        #   temp_x.size()[1]*temp_x.size()[2]*temp_x.size()[3]
          model[i].append(nn.Flatten())
          model[i].append(nn.Linear(in_features=oux.size()[1], out_features=out_cc*out_size_1*out_size_2))
          model[i].append(Reshape(out_cc,out_size_1,out_size_2))#上下的2位置的参数是需要一样的，含义是通道数（输入的）
          
          oux=nn.Linear(in_features=oux.size()[1], out_features=out_cc*out_size_1*out_size_2).to(device)(oux)
    else:
          temp_x=oux
          model[i].append(nn.Linear(in_features=temp_x.size()[1], out_features=out_cc*out_size_1*out_size_2))
          model[i].append(Reshape(out_cc,out_size_1,out_size_2))#上下的2位置的参数是需要一样的，含义是通道数（输入的）
    # temp_y,result_name=fitness(model[i])
    # Y.append(copy.deepcopy(temp_y))
    # Result_name.append(str(copy.deepcopy(result_name)))
  return model#,Y,Result_name

def cal_f(x,g,f1,s2,param):
    model=nn.ModuleList()
    # batch_size,
    # tupianchicun,控制输入图尺寸
    # linear_line,卷积后能接受的最大全连接层数。
    # shuru_c,控制输入通道数
    # max_output_linear_size，全连接层的最大输出尺寸。
    #out_cc,out_size_1,out_size_2,输出图通道，尺寸1，尺寸2
    cor=[[2,73,73,3000,256,2000,2,36,36,2,36,36],[2,36,36,3000,2,2000,2,36,36,2,36,36],[2,36,36,3000,4,1200,2,73,73,2,36,36]]
    try:
        for i,n in enumerate(x):
            # if(i==1):
                model_temp=getmodel(n,cor[i][0],cor[i][1],cor[i][2],cor[i][3],cor[i][4],cor[i][5],cor[i][6],cor[i][7],cor[i][8],cor[i][9],cor[i][10],cor[i][11])
            # elif (i==2):
            #     model_temp=getmodel(x,cor[i][0],cor[i][1],cor[i][2],cor[i][3],cor[i][4],cor[i][5],cor[i][6],cor[i][7],cor[i][8],cor[i][9],cor[i][10],cor[i][11])
            # else:
            #     model_temp=getmodel(x,cor[i][0],cor[i][1],cor[i][2],cor[i][3],cor[i][4],cor[i][5],cor[i][6],cor[i][7],cor[i][8],cor[i][9],cor[i][10],cor[i][11])

                model.append(model_temp[0])
    # for i,n in enumerate(x[0]):
    #     model.extend(dict[n])
    # # print(model)
    # if g!=0:
    #     a=fitness(model[0],model[1],model[2],g,f1,s2,param)
    # else:
    # a=random.random()
    # a=fitness(model[0],model[1],model[2],g,f1,s2,param)
    # print(a)
    # return a
        a=fitness(model[0],model[1],model[2],g,f1,s2,param)
        print(a)
        return a
    except:
        s=sys.exc_info()
        print("Error '%s' happened on line %d" % (s[1],s[2].tb_lineno))
        return (0.,'no files')

def crossover_first(parent1, parent2, crossover_rate, mutation_rate):
    """
    Performs crossover and mutation on two binary lists.

    Args:
    - parent1 (list of int): The first parent binary list.
    - parent2 (list of int): The second parent binary list.
    - crossover_rate (float): The probability of crossover (0.0 to 1.0).
    - mutation_rate (float): The probability of mutation (0.0 to 1.0).

    Returns:
    - child (list of int): The offspring binary list.
    """

    # if len(parent1) != len(parent2):
    #     raise ValueError("Parent lists must have the same length")
    child = [[[],[],[]],[[],[],[]]]
    train_key1= [[[],[],[]],[[],[],[]]]
    train_key2= [[[],[],[]],[[],[],[]]]
    del_key=[[],[]]
    for r in range(row_nums):
        crossover_point1 = random.randint(1, len(parent1[r])-1)
        crossover_point2 = random.randint(1,  len(parent2[r])-1)
        child[0][r] = parent1[r][:crossover_point1] + parent2[r][crossover_point2:] 
        child[1][r] = parent2[r][:crossover_point2] + parent1[r][crossover_point1:]
        train_key1[0][r]=list(range(crossover_point1))
        train_key1[1][r]=list(range(crossover_point2,len(parent2[r])))
        train_key2[0][r]=list(range(crossover_point1,len(parent1[r])))
        train_key2[1][r]=list(range(crossover_point2))
    i=0
    for bitt1, bitt2 in zip(parent1, parent2):
        del_key[0].append([])
        del_key[1].append([])
        # Perform crossover with the specified probability
        # 01注释
        for bit1, bit2 in zip( bitt1, bitt2):
            # Perform mutation with the specified probability
            
            if random.random() < mutation_rate and len(child[0][i])>0:
                child[0][i][-1] = 1 - child[0][i][-1]  # Flip the bit
                del_key[0][-1].append(i)
            if random.random() < mutation_rate and len(child[1][i])>0:
                child[1][i][-1] = 1 - child[1][i][-1]  # Flip the bit
                del_key[1][-1].append(i)  
        i+=1

    return child,train_key1,train_key2,del_key


def crossover_second(parent1, parent2, crossover_rate, mutation_rate):
    """
    Performs crossover and mutation on two lists with 'c' or 'f' elements at the same positions.

    Args:
    - parent1 (list): The first parent list.
    - parent2 (list): The second parent list.
    - crossover_rate (float): The probability of crossover (0.0 to 1.0).
    - mutation_rate (float): The probability of mutation (0.0 to 1.0).

    Returns:
    - child1 (list): The first offspring list.
    - child2 (list): The second offspring list.
    """

    if len(parent1) != len(parent2):
        raise ValueError("Parent lists must have the same length")

    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)
    train_key1= [[[],[],[]],[[],[],[]]]
    train_key2= [[[],[],[]],[[],[],[]]]
    del_key=[[],[]]
    # Randomly choose a crossover point where 'c' or 'f' is located
    for r in range(row_nums):
        crossover_points = [i for i in range(len(parent1[r])) if parent1[r][i] in ['c', 'f']]
        if crossover_points:
            crossover_point = random.choice(crossover_points)
            child1[r][crossover_point:] = parent2[r][crossover_point:]
            child2[r][crossover_point:] = parent1[r][crossover_point:]
            train_key1[0][r]=list(range(crossover_point))
            train_key1[1][r]=list(range(crossover_point,len(parent2[r])))
            train_key2[0][r]=list(range(crossover_point,len(parent1[r])))
            train_key2[1][r]=list(range(crossover_point))
#     stride_set=(range(1,7))#ending不可达
# padding_set=(range(0,4))#ending不可达
# C_out_set=(range(1,33))#ending不可达
# kernel_size_set=(range(1,6,2))
# out_features_set=range(100,1800)
    # Perform mutation on numeric elements with the specified probability
        del_key[0].append([])
        del_key[1].append([])
        for i in range(len(child1[r])):
            if(child1[r][i]=='c'):
                k=0
            elif(child1[r][i]=='f'):
                k=5
            # 01注释
            if isinstance(child1[r][i], int) and random.random() < mutation_rate:
                
                if(k==1):
                    child1[r][i] = random.choice(C_out_set)  
                elif(k==2):
                    child1[r][i] = random.choice(kernel_size_set)
                elif(k==3):
                    stride_set=(range(1,max(2,min(child1[r][i-1],7))))
                    child1[r][i] = random.choice(stride_set)
                elif(k==4):
                    child1[r][i] = random.choice(padding_set)
                elif(k==6):
                    child1[r][i] = random.choice(out_features_set)
            if isinstance(child2[r][i], int) and random.random() < mutation_rate:
                if(k==1):
                    child2[r][i] = random.choice(C_out_set)  
                elif(k==2):
                    child2[r][i] = random.choice(kernel_size_set)
                elif(k==3):
                    stride_set=(range(1,max(2,min(child1[r][i-1],7))))
                    child2[r][i] = random.choice(stride_set)
                elif(k==4):
                    child2[r][i] = random.choice(padding_set)
                elif(k==6):
                    child2[r][i] = random.choice(out_features_set)
            k+=1

    return child1, child2,train_key1,train_key2,del_key

def filter_and_delete_weights(folder_path, key_value):
    # 列举文件夹下以"model"开头的.pth文件
    pattern = re.compile(r'^model_(\d+)_(\d+)_(\d+)\.pth$')
    weight_files = [f for f in os.listdir(folder_path) if pattern.match(f) and os.path.isfile(os.path.join(folder_path, f))]

    # 删除不在key_value中的文件
    for file_name in weight_files:
        match = pattern.match(file_name)
        if match:
            g, i, j = map(int, match.groups())
            flag=True
            for k in range(len(key_value)):
                if  [g, i, j] in key_value[k]:
                    flag=False
                    break
                
            if  flag:
                file_path = os.path.join(folder_path, file_name)
                os.remove(file_path)
                print(f"Deleted: {file_path}")


# 从编码转为nn.ModuleList
def code2model(x):
    res=nn.ModuleList()
    temo_cin=2
    return res


# 首先初始化第一级编码
X1=[]
for i in range(N1):
    X1.append([])
    for u in range(row_nums):
        # 01注释
        X1[i].append(list(np.random.choice([0, 1], size=random.randint(r_min, r_max)))) 
#接着继续初始化

X2=[]
key_value=[]
first_key=list(range(N1))
for i in range(N1):
    X2.append([])
    key_value.append([])
    for k in range(N2):
        X2[i].append([])
        key_value[i].append([])
        for u in range(row_nums):
            X2[i][k].append([])
            key_value[i][k]=[0,i,k]
            for j in X1[i][u]:
                
                if(j==0): #卷积，加入，输出通道数，卷积核，步长、padding
                    X2[i][k][u].append('c')
                    X2[i][k][u].append(random.choice(C_out_set))
                    X2[i][k][u].append(random.choice(kernel_size_set))
                    stride_set=range(1,max(min(X2[i][k][u][-1],7),2))
                    X2[i][k][u].append(random.choice(stride_set))
                    X2[i][k][u].append(random.choice(padding_set))
                elif(j==1):
                    X2[i][k][u].append('f')
                    X2[i][k][u].append(random.choice(out_features_set))
                else:
                    raise ValueError("无效的第一级编码")
filter_and_delete_weights("/home/shx/code/test4_conv/save_weights", key_value)
# 开始迭代交叉变异等操作
##开始迭代
Y=[ [0 for _ in range(N2)]for _ in range(N1)]
sum_081=0
sum_mei081=0
param=[]
for i in range(N1):
    for j in range(N2):
         Y[i][j]=cal_f(X2[i][j],0,i,j,param)
text_save('test4_conv/results2/Y2_best.txt',[-1,Y])
temp_result=[]
for i in range(N1):
    count_081=0
    max_081=0
    for j in range(N2):
        if Y[i][j][0]>0.81:
            count_081+=1
            sum_081+=1
        else:
            sum_mei081+=1
        if max_081<Y[i][j][0]:
            max_081=Y[i][j][0]
    temp_result.append([first_key[i],max_081,count_081])
             
text_save('test4_conv/results2/Y_X1.txt',[-1,temp_result])
first_key_record=N1
for g in range(1,G):
    X1
    # 随机选择两个父代一级编码
    if(len(X1)<=inc_line_k_min_k):
    # if True:
        parent_index=random.sample(range(len(X1)), 2)
        child,train_key0,train_key1,del_key = crossover_first(X1[parent_index[0]], X1[parent_index[1]], crossover_rate_frist, mutation_rate_frist)
        X1.append(child[0])
        X1.append(child[1])
        first_key.append(first_key_record)
        first_key_record+=1
        first_key.append(first_key_record)
        first_key_record+=1
        key_value.append([])
        # key_value[-1].append([g,len(key_value)]-1,0)
        key_value.append([])
        # key_value[-1].append([g,len(key_value)]-1,0)
        # 选取前kge
        temp=[]
        sorted_pairs0 = sorted(zip(X2[parent_index[0]], Y[parent_index[0]],key_value[parent_index[0]]), key=lambda pair: pair[1], reverse=True)
        sorted_pairs1 = sorted(zip(X2[parent_index[1]], Y[parent_index[1]],key_value[parent_index[1]]), key=lambda pair: pair[1], reverse=True)
        X2.append([])
        Y.append([])
        X2.append([])
        Y.append([])
        for i in range(0,N2+1,2):
            temp.append([])
            temp.append([])
            rand0=random.randint(0,k_best_first)
            rand1=random.randint(0,k_best_first)
            for r in range(row_nums):
                temp[i].append([])
                temp[i+1].append([])
                k=0
                k_sum=0
                temp2=[]
                while k<len(X1[parent_index[0]][r]):
                    if isinstance(temp[i][r], int) or  isinstance(temp[i+1][r], int):
                         isinstance(temp[i][r], int)
                    if k in train_key0[0][r] :#给第一个子代

                            if X1[parent_index[0]][r][k]==1:
                                temp[i][r]+=sorted_pairs0[rand0][0][r][k_sum:k_sum+2]
                                k_sum+=2
                            else:
                                temp[i][r]+=sorted_pairs0[rand0][0][r][k_sum:k_sum+5]
                                k_sum+=5

                                
                    else:#给第二个子代

                            if X1[parent_index[0]][r][k]==1:
                                temp2+=sorted_pairs0[rand1][0][r][k_sum:k_sum+2]
                                k_sum+=2
                            else:
                                temp2+=sorted_pairs0[rand1][0][r][k_sum:k_sum+5]
                                k_sum+=5
                        
                    k+=1
                k=0
                k_sum=0
                while k<len(X1[parent_index[1]][r]):
                    if k in train_key0[1][r]:
                            if X1[parent_index[1]][r][k]==1:
                                temp[i][r]+=sorted_pairs1[rand0][0][r][k_sum:k_sum+2]
                                k_sum+=2
                            else:
                                temp[i][r]+=sorted_pairs1[rand0][0][r][k_sum:k_sum+5]
                                k_sum+=5
                    else:
                            if X1[parent_index[1]][r][k]==1:
                                temp[i+1][r]+=sorted_pairs1[rand1][0][r][k_sum:k_sum+2]
                                k_sum+=2
                            else:
                                temp[i+1][r]+=sorted_pairs1[rand1][0][r][k_sum:k_sum+5]
                                k_sum+=5
                    k+=1
                temp[i+1][r]+=temp2
                #删减一下
                k=0
                key_k=0
                while k<len(temp[i][r]):
                    if temp[i][r][k]=='c' or temp[i][r][k]=='f':
                        key_k+=1
                    if key_k in del_key[0][r]:
                        if temp[i][r][k]=='c':
                            temp2=[]
                            temp2+='f'
                            temp2.append(random.choice(out_features_set))
                            temp[i][r]=temp[i][r][0:k]+temp2+temp[i][r][k+5:]
                        elif temp[i][r][k]=='f':
                            temp2=[]
                            temp2+='c'
                            temp2.append(random.choice(C_out_set))
                            temp2.append(random.choice(kernel_size_set))
                            stride_set=range(1,max(2,min(7,temp2[-1])))
                            temp2.append(random.choice(stride_set))
                            temp2.append(random.choice(padding_set))
                            temp[i][r]=temp[i][r][0:k]+temp2+temp[i][r][k+2:]
                        # else:
                        #     continue
                    k+=1
                k=0
                key_k=0
                while k<len(temp[i+1][r]):
                    if temp[i+1][r][k]=='c' or temp[i+1][r][k]=='f':
                        key_k+=1
                    if key_k in del_key[1][r]:
                        if temp[i+1][r][k]=='c':
                            temp2=[]
                            temp2+='f'
                            temp2.append(random.choice(out_features_set))
                            temp[i+1][r]=temp[i+1][r][0:k]+temp2+temp[i+1][r][k+5:]
                        elif temp[i+1][r][k]=='f':
                            temp2=[]
                            temp2+='c'
                            temp2.append(random.choice(C_out_set))
                            temp2.append(random.choice(kernel_size_set))
                            stride_set=range(1,max(2,min(7,temp2[-1])))
                            temp2.append(random.choice(stride_set))
                            temp2.append(random.choice(padding_set))
                            temp[i+1][r]=temp[i+1][r][0:k]+temp2+temp[i+1][r][k+2:]
                        # else:
                        #     continue
                    k+=1
            # key_value[parent_index[0]][rand0]
            param=[sorted_pairs0[rand0][2][0],sorted_pairs0[rand0][2][1],sorted_pairs0[rand0][2][2],\
                sorted_pairs1[rand0][2][0],sorted_pairs1[rand0][2][1],sorted_pairs1[rand0][2][2],train_key0,del_key[0],True]
            Y[-2].append(cal_f(temp[i],g,len(X1)-2,len(X2[-1]),param))
            if(Y[-2][-1][0]>0.81):
                sum_081+=1
            else:
                sum_mei081+=1
            param=[sorted_pairs0[rand0][2][0],sorted_pairs0[rand0][2][1],sorted_pairs0[rand0][2][2],\
                sorted_pairs1[rand0][2][0],sorted_pairs1[rand0][2][1],sorted_pairs1[rand0][2][2],train_key1,del_key[1],False]
            Y[-1].append(cal_f(temp[i+1],g,len(X1)-1,len(X2[-2]),param))
            if(Y[-1][-1][0]>0.81):
                sum_081+=1
            else:
                sum_mei081+=1
            key_value[-2].append([g,len(key_value)-2,len(key_value[-2])])
            X2[-2].append(temp[i])
            key_value[-1].append([g,len(key_value)-1,len(key_value[-1])])
            X2[-1].append(temp[i+1])

    # 以下是对二级编码的内部交叉变异
    for i in range(len(X2)):
        for j in range(len(X2[i])//2): 
            # for _ in range(N2//2):
                parent_index=random.sample(range(len(X2[i])), 2)
                child1,child2,train_key0,train_key1,del_key=crossover_second(X2[i][parent_index[0]], X2[i][parent_index[1]], crossover_rate_frist, mutation_rate_frist)
                X2[i].append(child1)
                param=[key_value[i][parent_index[0]][0],key_value[i][parent_index[0]][1],key_value[i][parent_index[0]][2],\
                    key_value[i][parent_index[1]][0],key_value[i][parent_index[1]][1],key_value[i][parent_index[1]][2],train_key0,del_key[0],False]
                # [g-1,i,parent_index[0],g-1,i,parent_index[1],train_key0,del_key[0],False]
                Y[i].append(cal_f(child1,g,i,N2+2*j,param))
                if(Y[i][-1][0]>0.81):
                    sum_081+=1
                else:
                    sum_mei081+=1
                key_value[i].append([g,i,len(Y[i])-1])
                X2[i].append(child2)
                param=[key_value[i][parent_index[0]][0],key_value[i][parent_index[0]][1],key_value[i][parent_index[0]][2],\
                    key_value[i][parent_index[1]][0],key_value[i][parent_index[1]][1],key_value[i][parent_index[1]][2],train_key1,del_key[1],False]
                # [g-1,i,parent_index[0],g-1,i,parent_index[1],train_key1,del_key[1],False]
                Y[i].append(cal_f(child2,g,i,N2+2*j+1,param))
                if(Y[i][-1][0]>0.81):
                    sum_081+=1
                else:
                    sum_mei081+=1
                key_value[i].append([g,i,len(Y[i])-1])
        sorted_pairs = sorted(zip(X2[i], Y[i], key_value[i]), key=lambda pair: pair[1], reverse=True)
        # 选择前 N2 个元素
        X2[i] = [pair[0] for pair in sorted_pairs[:N2]]
        Y[i] = [pair[1] for pair in sorted_pairs[:N2]]
        key_value[i] = [pair[2] for pair in sorted_pairs[:N2]]
        filter_and_delete_weights("/home/shx/code/test4_conv/save_weights", key_value)
        
    #对一级编码进行筛选
    if g%5==0 and len(X1)>inc_line_k_min_k:
        best_y=[]
        temp_y=0
        for y in Y:
            temp_y=0
            for v,tex in y:
                if v>temp_y:
                    temp_y=v
            best_y.append(temp_y)
        sorted_pairs = sorted(zip(best_y,X1,X2, Y, key_value,first_key), key=lambda pair: pair[0], reverse=True)
        X1 = [pair[1] for pair in sorted_pairs[:len(best_y)-1]]
        X2 = [pair[2] for pair in sorted_pairs[:len(best_y)-1]]
        Y = [pair[3] for pair in sorted_pairs[:len(best_y)-1]]
        key_value = [pair[4] for pair in sorted_pairs[:len(best_y)-1]]
        first_key = [pair[5] for pair in sorted_pairs[:len(best_y)-1]]
        filter_and_delete_weights("/home/shx/code/test4_conv/save_weights", key_value)
    print(g)
    text_save('test4_conv/results2/Y2_best.txt',[g,Y])
    temp_result=[]
    for i in range(len(Y)):
        count_081=0
        max_081=0
        for j in range(len(Y[i])):
            if Y[i][j][0]>0.81:
                count_081+=1
            if max_081<Y[i][j][0]:
                max_081=Y[i][j][0]
        temp_result.append([first_key[i],max_081,count_081])
                
    text_save('test4_conv/results2/Y_X1.txt',[g,temp_result])
text_save('test4_conv/results2/Y2_best.txt',[sum_081,sum_mei081])
        
