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
    noise = noise * delta_range_c + delta_range_c# 将噪声缩放到 [c, 2c] 范围内
    noisy_inputs = torch.clamp(inputs + noise, 0, 255)# 加噪声并限制在 [0, 255] 范围内
    return noisy_inputs

def train(epoch, model_ls, lr_scheduler, optimizer, trainloader,args):
    train_adv_loss = 0.
    train_other_adv_loss = 0.
    adv_correct = 0
    total = 0
    #pbar = tqdm(trainloader)
    pbar = trainloader
    curr_lr = lr_scheduler.get_lr()[0]
    #pbar.set_description("Train:{:3d} epoch lr {:.1e}".format(epoch, curr_lr))
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.cuda(), targets.cuda()
        #adv_inp = apgd_attack(model_ls, inputs, targets, prob, 8 / 255.0, 2 / 255.0, 10, other_weight=args.other_weight, num_classes=num_classes, normalize=normalize)
        adv_inp = add_normal_noise(inputs)
        optimizer.zero_grad()
        model_ls[-1].train()
        pred = model_ls[-1](adv_inp)
        adv_loss = F.cross_entropy(pred, targets, reduction="none").mean()
        _, adv_predicted = pred.detach().max(1)

        other_pred = -model_ls[-1](adv_inp)
        other_advloss = - F.log_softmax(other_pred, dim=1) * (1 - F.one_hot(targets, num_classes=10))
        other_advloss = other_advloss.sum() / ((10 - 1) * len(targets))

        total_advloss = adv_loss + args['other_weight'] * other_advloss

        total_advloss.backward()
        total += targets.size(0)
        adv_correct += adv_predicted.eq(targets).sum().item()

        optimizer.step()

        train_adv_loss += adv_loss.item()
        train_other_adv_loss += other_advloss.item()

        # pbar_dic = OrderedDict()
        # pbar_dic['Adv Acc'] = '{:2.2f}'.format(100. * adv_correct / total)
        # pbar_dic['adv loss'] = '{:.3f}'.format(train_adv_loss / (batch_idx + 1))
        # pbar_dic['otheradv loss'] = '{:.3f}'.format(train_other_adv_loss / (batch_idx + 1))
        # pbar.set_postfix(pbar_dic)


def osp_iter(epoch, model_ls, prob, osp_lr_init,osploader):

    M = len(prob)
    err = np.zeros(M)
    n = 0
    pbar = tqdm(osploader)
    curr_lr = osp_lr_init/(1+epoch)#可能要删
    model_ls[-1].eval()
    pbar.set_description("OSP:{:3d} epoch lr {:.4f}".format(epoch, curr_lr))
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.cuda(), targets.cuda()
        #adv_inp = arc_attack(model_ls, inputs, targets, prob, 8 / 255.0, 8 / 255.0, 10,  num_classes=num_classes, normalize=normalize, g=2)
        adv_inp = add_normal_noise(inputs)
        for m in range(M):
            t_m = model_ls[m](adv_inp)
            err[m]+= (t_m.max(1)[1] != targets).sum().item()

        n += targets.size(0)
        pbar_dic = OrderedDict()
        pbar_dic['Adv Acc'] = '{:2.2f}'.format(100. * (1-sum(err*prob)/n))
        pbar.set_postfix(pbar_dic)
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


def localUpdateBARRE(client, epoch, Net, global_parameters, args):

    outdir = args['outdir']
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    seed_torch(args['seed'])
    trainloader, osploader = get_loaders(args,client.train_ds)
    #normalize = get_normalize(args)

    model_ls = []
    model = Net
    model.load_state_dict(global_parameters)  # 将 global_parameters 中的模型参数加载到模型中
    for iteration in range(args['M']):
        model_ls.append(model)
        prob = np.ones(len(model_ls))/len(model_ls)
        print('alpha = ',prob)
        if iteration >= 1:
            model_ls[-1].load_state_dict(model_ls[-2].state_dict())

        if iteration <= args['resume_iter']:
            print('需要恢复模型状态')

        else:
            start_epoch = -1  # start from epoch 0 or last checkpoint epoch

            if args['optimizer'] == "sgd":
                optimizer = optim.SGD(model_ls[-1].parameters(), lr=args['learning_rate'], momentum=0.9, weight_decay=5e-4)
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                    milestones=[int(0.5 * args['total_epochs']), int(0.75 * args['total_epochs'])], gamma=0.1)
            elif args['optimizer'] == "adam":
                optimizer = optim.Adam(model_ls[-1].parameters(), lr=args['learning_rate'], weight_decay=5e-4)

            for epoch in range(start_epoch + 1, args['total_epochs']):

                train(epoch,model_ls, lr_scheduler,optimizer,trainloader,args)
                if args['optimizer'] == "sgd":
                    lr_scheduler.step()
                if epoch >= args['total_epochs']//2:
                    if ((epoch - args['total_epochs']//2 + 1)% args['osp_freq'] == 0 or epoch == args['total_epochs']-1) and iteration > 2:
                        eta_best = 1
                        osp_lr_init = args['osp_lr_max']*lr_scheduler.get_lr()[0]
                        print('==> Begin OSP routine, starting alpha=' + arr_to_str(prob))
                        for t in range(args['osp_epochs']):
                            osp_lr = 0.3
                            g_t = osp_iter(t,model_ls, prob,osp_lr_init,osploader) #sub-gradient of eta(alpha_t)
                            eta_t = sum(g_t*prob) #eta(alpha_t)
                            if eta_t <= eta_best:
                                t_best = t
                                prob_best = np.copy(prob)
                                eta_best = eta_t
                            print("best acc = {:2.2f} @ alpha_best = ".format(100.*(1-eta_best)) + arr_to_str(prob_best))
                            prob = proj_onto_simplex(prob - osp_lr * g_t)
                        print('==> End OSP routine, final alpha=' + arr_to_str(prob_best))
                        prob = np.copy(prob_best)
    
    return weighted_average_model(model_ls, prob, Net)
