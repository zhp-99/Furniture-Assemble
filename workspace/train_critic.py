import os
import time
import sys
from time import strftime
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from datasets.dataset import AssembleDataset
from models.model_assemble import Network as CriticNetwork

from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter


from tensorboardX import SummaryWriter

def log_writer(epoch, writer, content, is_val):
    prefix = "Val/" if is_val else "Train/"
    for key, value in content.items():
        print("Epoch: %d, %s%s: %f" % (epoch, prefix, key, value))
        writer.add_scalar(prefix + key, value, epoch)

def train():
    writer = SummaryWriter(log_dir='logs')
    feat_dim = 128
    cp_feat_dim = 32
    dir_feat_dim = 32
    lr = 0.001
    weight_decay = 1e-5
    lr_decay_every = 500
    lr_decay_by = 0.9
    batch_size = 128
    num_epochs = 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Creating network ...... ')
    network = CriticNetwork(feat_dim, cp_feat_dim, dir_feat_dim)
    network_opt = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
    network_lr_scheduler = torch.optim.lr_scheduler.StepLR(network_opt, step_size=lr_decay_every, gamma=lr_decay_by)

    # Continue_to_play
    #     network.load_state_dict(torch.load(os.path.join(conf.exp_dir, 'ckpts', '%d-network.pth' % conf.saved_epoch)))
    #     network_opt.load_state_dict(torch.load(os.path.join(conf.exp_dir, 'ckpts', '%d-optimizer.pth' % conf.saved_epoch)))
    #     network_lr_scheduler.load_state_dict(torch.load(os.path.join(conf.exp_dir, 'ckpts', '%d-lr_scheduler.pth' % conf.saved_epoch)))

    # send parameters to device
    network.to(device)
    for state in network_opt.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
    # load dataset
    print('Loading dataset ...... ')
    train_dataset = AssembleDataset("train")
    val_dataset = AssembleDataset("val")

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   pin_memory=True, num_workers=0, drop_last=True)
    train_num_batch = len(train_dataloader)

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                                 pin_memory=True, num_workers=0, drop_last=True)

    # start training
    start_epoch = 0
    print('Start training ...... ')
    for epoch in range(start_epoch, num_epochs):
        train_batches = enumerate(train_dataloader, 0)
        val_batches = enumerate(val_dataloader, 0)

        train_ep_loss, train_cnt = 0, 0

        ### train for every batch
        total_all_acc, positive_acc, negative_acc, fail_acc = 0, 0, 0, 0
        percent_positive, percent_negative, percent_fail = 0, 0, 0
        for train_batch_ind, batch in tqdm(train_batches, total=train_num_batch):
            # set models to training mode
            network.train()
            total_loss, all_acc, positive, negative, fail = critic_forward(batch=batch, network=network, device=device)
            total_all_acc += all_acc
            positive_acc += positive[0]
            percent_positive += positive[1]
            negative_acc += negative[0]
            percent_negative += negative[1]
            fail_acc += fail[0]
            percent_fail += fail[1]

            network_opt.zero_grad()
            total_loss.backward()
            network_opt.step()
            network_lr_scheduler.step()

            train_ep_loss += total_loss
            train_cnt += 1

        content = {
            "total_loss": train_ep_loss / train_cnt,
            "all_acc": total_all_acc / train_cnt,
            "positive_acc": positive_acc / train_cnt,
            "negative_acc": negative_acc / train_cnt,
        }
        log_writer(epoch, writer, content, is_val=False)
        if epoch % 1 == 0:
            # validate
            total_all_acc, positive_acc, negative_acc, fail_acc = 0, 0, 0, 0
            percent_positive, percent_negative, percent_fail = 0, 0, 0
            val_cnt = 0
            for val_batch_ind, batch in tqdm(val_batches, total=len(val_dataloader)):
                # set models to evaluation mode
                network.eval()

                with torch.no_grad():
                    all_acc, positive, negative, fail = critic_forward(batch=batch, network=network, device=device, is_val=True)
                    total_all_acc += all_acc
                    positive_acc += positive[0]
                    percent_positive += positive[1]
                    negative_acc += negative[0]
                    percent_negative += negative[1]
                    fail_acc += fail[0]
                    percent_fail += fail[1]
                    val_cnt += 1
                
            content = {
                "all_acc": total_all_acc / val_cnt,
                "positive_acc": positive_acc / val_cnt,
                "negative_acc": negative_acc / val_cnt,
            }
            log_writer(epoch, writer, content, is_val=True)

        # save checkpoint
        if epoch % 10 == 0:
            with torch.no_grad():
                print('Saving checkpoint ...... ')
                torch.save(network.state_dict(), os.path.join("logs", 'ckpts', '%d-network.pth' % epoch))
                torch.save(network_opt.state_dict(), os.path.join("logs", 'ckpts', '%d-optimizer.pth' % epoch))
                torch.save(network_lr_scheduler.state_dict(), os.path.join("logs", 'ckpts', '%d-lr_scheduler.pth' % epoch))
                print('DONE')


def get_acc(reward, pred_score):
    reward_succ = reward < 0.04
    pred_score_succ = pred_score < 0.04
    all_acc = (pred_score_succ == reward_succ).float().mean()

    positive_mask = (reward < 0.04)
    positive_acc = pred_score_succ[positive_mask].float().mean()
    percent_positive = positive_mask.sum()/len(reward)

    negative_mask = ((reward >= 0.04) & (reward < 0.3))
    pred_score_neg = ((pred_score >= 0.04) & (pred_score < 0.3))
    negative_acc = pred_score_neg[negative_mask].float().mean()
    percent_negative = negative_mask.sum()/len(reward)

    fail_mask = (reward >= 0.3)
    pred_score_fail = (pred_score >= 0.3)
    fail_acc = pred_score_fail[fail_mask].float().mean()
    percent_fail = fail_mask.sum()/len(reward)

    return all_acc, (positive_acc,percent_positive), (negative_acc,percent_negative),(fail_acc, percent_fail)


def critic_forward(batch, network, device=None, is_val=False):
    # zhp: need to change: add normal to point feature
    table_points, leg_points, sampled_point, sampled_normal, action_direct, reward = batch
    
    pcs = table_points.to(device)
    sampled_point = sampled_point.to(device)
    action_direct = action_direct.to(device)
    reward = reward.to(device)  
    pred_score = network.forward(pcs, sampled_point, action_direct)   # after sigmoid
    if not is_val:
        criterion = torch.nn.MSELoss()
        loss = criterion(pred_score, reward) *1000
        total_loss = loss.mean()
        all_acc, positive, negative, fail = get_acc(reward, pred_score)
        return total_loss, all_acc, positive, negative, fail
    else: 
        return get_acc(reward, pred_score)


if __name__ == '__main__':
    train()
