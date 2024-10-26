import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#import ipdb
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
# from attack import arc_attack, apgd_attack
from utils import seed_torch, arr_to_str, proj_onto_simplex
from datasets import get_loaders


def add_path(path):
    if path not in sys.path:
        print('Adding {}'.format(path))
        sys.path.append(path)

add_path("../lib")       

def add_normal_noise(inputs, delta_range_c = 5):
    noise = torch.rand_like(inputs)# 生成 [0, 1] 范围内的随机噪声
    #noise = noise * delta_range_c + delta_range_c# 将噪声缩放到 [c, 2c] 范围内
    noisy_inputs = torch.clamp(inputs + noise, 0, 255)# 加噪声并限制在 [0, 255] 范围内
    return noisy_inputs

def train(model, lr_scheduler, optimizer, trainloader, args):
    #pbar = tqdm(trainloader)
    pbar = trainloader
    #pbar.set_description("Train:{:3d} epoch lr {:.1e}".format(epoch, curr_lr))
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.cuda(), targets.cuda()
        adv_inp = add_normal_noise(inputs)
        optimizer.zero_grad()
        model.train()
        pred = model(adv_inp)
        loss = F.cross_entropy(pred, targets, reduction="none").mean()
        loss.backward()
        optimizer.step()


def osp_iter(epoch, model_ls, prob, osp_lr_init,osploader):

    M = len(prob)
    err = np.zeros(M)
    n = 0
    for batch_idx, (inputs, targets) in enumerate(osploader):
        inputs, targets = inputs.cuda(), targets.cuda()
        adv_inp = add_normal_noise(inputs)
        for m in range(M):
            model_ls[m].eval()  # 确保每个模型都在评估模式
            t_m = model_ls[m](adv_inp)
            err[m] += (t_m.max(1)[1] != targets).sum().item()

        n += targets.size(0)
        pbar_dic = OrderedDict()
        pbar_dic['Adv Acc'] = '{:2.2f}'.format(100. * (1 - sum(err * prob) / n))
    grad = err/n
    return grad

def weighted_average_model(model_ls, prob, Net):#得搞清楚这个net是啥，起到什么作用
    # 初始化一个空白模型，该模型结构与模型列表中的模型相同
    # 初始化最终模型的参数为零
    for param in Net.parameters():
        param.data.zero_()

    # 遍历每个模型，按照采样概率 prob 对其参数进行加权
    for i, model in enumerate(model_ls):
        model_params = model.state_dict()  # 获取模型的参数
        for key in model_params:
            # 按照概率 prob 加权累加到 final_model 的参数中
            Net.state_dict()[key].data += model_params[key].data * prob[i]

    # 返回最终模型的参数
    return Net.state_dict()

criterion = nn.CrossEntropyLoss()#这是干啥的


def localUpdateBARRE(train_ds, Net, global_parameters, args):

    outdir = args['outdir']
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    seed_torch(args['seed'])
    trainloader, osploader = get_loaders(args, train_ds)
    #normalize = get_normalize(args)

    model_ls = []
    prob=[]
    model = copy.deepcopy(Net)
    model.load_state_dict(global_parameters)  # 将 global_parameters 中的模型参数加载到模型中
    for iteration in range(args['M']):

        if iteration <= args['resume_iter']:
            print('需要恢复模型状态')

        else:
            start_epoch = -1  # start from epoch 0 or last checkpoint epoch

            if args['optimizer'] == "sgd":
                optimizer = optim.SGD(model.parameters(), lr=args['learning_rate'], momentum=0.9, weight_decay=5e-4)
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                    milestones=[int(0.5 * args['total_epochs']), int(0.75 * args['total_epochs'])], gamma=0.1)
            elif args['optimizer'] == "adam":
                optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=5e-4)

            for epoch in range(start_epoch + 1, args['total_epochs']):

                train(model, lr_scheduler,optimizer,trainloader,args)
                if args['optimizer'] == "sgd":
                    lr_scheduler.step()
                
            model_ls.append(model)
            prob = np.ones(len(model_ls))/len(model_ls)
    
    eta_best = 1
    osp_lr_init = args['osp_lr_max']*lr_scheduler.get_lr()[0]
    print('==> Begin OSP routine, starting alpha=' + arr_to_str(prob))
    for t in range(args['osp_epochs']):
        osp_lr = 0.3
        g_t = osp_iter(t,model_ls, prob,osp_lr_init,osploader) #sub-gradient of eta(alpha_t)
        eta_t = sum(g_t * prob) #eta(alpha_t)
        if eta_t <= eta_best:
            t_best = t
            prob_best = np.copy(prob)
            eta_best = eta_t
        prob = proj_onto_simplex(prob - osp_lr * g_t)
    print('==> End OSP routine, final alpha=' + arr_to_str(prob_best))
    prob = np.copy(prob_best)                    
    print('alpha = ', prob)

    return weighted_average_model(model_ls, prob, Net)
