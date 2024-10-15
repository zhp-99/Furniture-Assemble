import os
import time
import sys
from time import strftime
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from datasets.dataset import AssembleDataset
from models.model_critic_fir import Network as CriticNetwork
from models.pointnet2_ops.pointnet2_utils import furthest_point_sample

from tqdm import tqdm, trange


from tensorboardX import SummaryWriter


def train():
    feat_dim = 128
    cp_feat_dim = 32
    dir_feat_dim = 32
    lr = 0.001
    weight_decay = 1e-5
    lr_decay_every = 500
    lr_decay_by = 0.9
    batch_size = 32
    num_epochs = 10000
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

    # affordance, actor, critic = None, None, None
    # # load aff2 + actor2 + critic2
    # if conf.model_version == 'model_critic_fir':
    #     aff_def = utils.get_model_module(conf.aff_version)
    #     affordance = aff_def.Network(conf.feat_dim, conf.task_feat_dim, conf.cp_feat_dim, conf.dir_feat_dim, task_input_dim=task_input_dim)
    #     affordance.load_state_dict(torch.load(os.path.join(conf.aff_path, 'ckpts', '%s-network.pth' % conf.aff_eval_epoch)))
    #     affordance.to(conf.device)

    #     actor_def = utils.get_model_module(conf.actor_version)
    #     actor = actor_def.Network(conf.feat_dim, conf.task_feat_dim, conf.cp_feat_dim, conf.dir_feat_dim, z_dim=conf.z_dim, task_input_dim=task_input_dim)
    #     actor.load_state_dict(torch.load(os.path.join(conf.actor_path, 'ckpts', '%s.pth' % conf.actor_eval_epoch)))
    #     actor.to(conf.device)

    #     critic_def = utils.get_model_module(conf.critic_version)
    #     critic = critic_def.Network(conf.feat_dim, conf.task_feat_dim, conf.cp_feat_dim, conf.dir_feat_dim, task_input_dim=task_input_dim)
    #     critic.load_state_dict(torch.load(os.path.join(conf.critic_path, 'ckpts', '%s-network.pth' % conf.critic_eval_epoch)))
    #     critic.to(conf.device)


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


    # train for every epoch
    print('Start training ...... ')
    for epoch in trange(start_epoch, num_epochs):   # 每个epoch重新获得一次train dataset


        train_batches = enumerate(train_dataloader, 0)
        val_batches = enumerate(val_dataloader, 0)

        val_fraction_done = 0.0
        val_batch_ind = -1

        ep_loss, ep_cnt = 0, 0
        train_ep_loss, train_cnt = 0, 0

        ### train for every batch
        for train_batch_ind, batch in train_batches:
            train_fraction_done = (train_batch_ind + 1) / train_num_batch
            train_step = epoch * train_num_batch + train_batch_ind



            # save checkpoint
            # if epoch % 2 == 0 and train_batch_ind == 0:
            #     with torch.no_grad():
            #         print('Saving checkpoint ...... ')
            #         torch.save(network.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-network.pth' % epoch))
            #         torch.save(network_opt.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-optimizer.pth' % epoch))
            #         torch.save(network_lr_scheduler.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-lr_scheduler.pth' % epoch))
            #         print('DONE')

            # set models to training mode
            network.train()

            # forward pass (including logging)
            total_loss = critic_forward(batch=batch, network=network,
                                        step=train_step,
                                        lr=network_opt.param_groups[0]['lr'],
                                        )

            # optimize one step
            network_opt.zero_grad()
            total_loss.backward()
            network_opt.step()
            network_lr_scheduler.step()

            train_ep_loss += total_loss
            train_cnt += 1

        print("epoch: %d, total_train_loss: %f" % (epoch, train_ep_loss / train_cnt))

        #     # validate one batch
        #     while val_fraction_done <= train_fraction_done and val_batch_ind + 1 < val_num_batch:
        #         val_batch_ind, val_batch = next(val_batches)

        #         val_fraction_done = (val_batch_ind + 1) / val_num_batch
        #         val_step = (epoch + val_fraction_done) * train_num_batch - 1

        #         log_console = not conf.no_console_log and (last_val_console_log_step is None or \
        #                 val_step - last_val_console_log_step >= conf.console_log_interval)
        #         if log_console:
        #             last_val_console_log_step = val_step

        #         # set models to evaluation mode
        #         network.eval()

        #         with torch.no_grad():
        #             # forward pass (including logging)
        #             loss = critic_forward(batch=val_batch, data_features=data_features, network=network, conf=conf, is_val=True, \
        #                                    step=val_step, epoch=epoch, batch_ind=val_batch_ind, num_batch=val_num_batch, start_time=start_time, \
        #                                    log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=val_writer, lr=network_opt.param_groups[0]['lr'],
        #                                    affordance=affordance, actor=actor, critic=critic)
        #             ep_loss += loss
        #             ep_cnt += 1

        # utils.printout(flog, "epoch: %d, total_train_loss: %f, total_val_loss: %f" % (epoch, train_ep_loss / train_cnt, ep_loss / ep_cnt))


def critic_forward(batch, network, lr=None, device=None, step=None, is_val=False, log_tb=False, tb_writer=None):
    # zhp: need to change: add normal to point feature
    table_points, leg_points, sampled_point, sampled_normal, action_direct, reward = batch

    pcs = table_points.to(device)
    sampled_point = sampled_point.to(device)
    action_direct = action_direct.to(device)
    reward = reward.to(device)  

    pred_score = network.forward(pcs, sampled_point, action_direct)   # after sigmoid
    loss = network.get_ce_loss_total(pred_score, reward)
    total_loss = loss.mean()


    # if is_val:
    #     pred = pred_score.detach().cpu().numpy() > conf.critic_score_threshold
    #     Fscore, precision, recall, accu = utils.cal_Fscore(np.array(pred, dtype=np.int32), gt_result.detach().cpu().numpy())

    # display information
    data_split = 'val' if is_val else 'train'
    # with torch.no_grad():

    #     # log to tensorboard
    #     if log_tb and tb_writer is not None:
    #         tb_writer.add_scalar('critic_loss', total_loss.item(), step)
    #         tb_writer.add_scalar('critic_lr', lr, step)
    #     if is_val and log_tb and tb_writer is not None:
    #         tb_writer.add_scalar('Fscore', Fscore, step)
    #         tb_writer.add_scalar('precision', precision, step)
    #         tb_writer.add_scalar('recall', recall, step)
    #         tb_writer.add_scalar('accu', accu, step)


    return total_loss


if __name__ == '__main__':
    train()
