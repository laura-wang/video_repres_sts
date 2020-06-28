import os
import time
import torch
import torchvision
import argparse
from torch import nn, optim
from torch.utils.data import DataLoader

from networks import c3d_large_BN, r3d, r21d
from datasets import ucf101_dataset
from utils.video_transforms import RandomCrop, RandomHorizontalFlip, ToTensor

from tensorboardX import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--bs', type=int, default=4, help='30, batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='10, num of workers to load data')
    parser.add_argument('--motion_flag', type=tuple, default=(1,1,1,1), help='motion flag')
    parser.add_argument('--app_flag', type=tuple, default=(1,1,1,1), help='app flag')
    parser.add_argument('--motion_dims', type=int, default=14, help='motion labels dimensions')
    parser.add_argument('--app_dims', type=int, default=13, help='appearance labels dimensions')
    parser.add_argument('--motion_w', type=float, default=1, help='weight of motion statistics')
    parser.add_argument('--app_w', type=float, default=0.1, help='weight of appearance statistics')
    parser.add_argument('--epoch_num', type=int, default=18, help='training epochs')
    parser.add_argument('--max_save_num', type=int, default=3, help='max save epoach num')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--wd', type=float, default=0.005, help='weight decay')
    parser.add_argument('--step_s', type=int, default=6, help='priod of learning rate decay.')
    parser.add_argument('--gamma', type=float, default=0.1, help='multiplicative factor of learning rate decay')
    parser.add_argument('--plt', type=str, default='windows', help='platform, controls data list and data dir')
    parser.add_argument('--dataset', type=str, default='ucf101', help='ucf101')
    parser.add_argument('--model', type=str, default='r3d', help='c3d/r3d/r21d, model type')
    parser.add_argument('--model_save_dir', type=str, default='pretrain_models', help='pretrain models save dir')
    parser.add_argument('--log_dir', type=str, default='visual_logs', help='visual logs save dir' )
    parser.add_argument('--data_list', type=str, default='list/ucf101_train.list', help='data list')
    parser.add_argument('--rgb_prefix', type=str, default='/data1/dataset/ucf101_jpegs_256/', help='rgb dir')
    parser.add_argument('--flow_x_prefix', type=str, default='/data1/dataset/ucf101_tvl1_flow/tvl1_flow/u/', help='flow x dir')
    parser.add_argument('--flow_y_prefix', type=str, default='/data1/dataset/ucf101_tvl1_flow/tvl1_flow/v/', help='flow y dir')

    args = parser.parse_args()

    return args


def train(args):

    torch.backends.cudnn.benchmark = True
    exp_name = 'pretrain_{}_{}_lr_{}'.format(args.dataset, args.model, args.lr)

    model_save_dir = os.path.join(args.model_save_dir, exp_name)
    log_dir = os.path.join(args.log_dir, exp_name)

    transforms = torchvision.transforms.Compose([RandomCrop(112),
                                                 RandomHorizontalFlip(0.5),
                                                 ToTensor()])
    train_dataset = ucf101_dataset.ucf101(args.data_list, args.rgb_prefix, args.flow_x_prefix, args.flow_y_prefix,
                                          motion_flag=args.motion_flag, app_flag=args.app_flag, transforms=transforms)
    dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=args.num_workers)

    ## init model
    if args.model == 'c3d':
        model = c3d_large_BN.c3d_BN(motion_dims=args.motion_dims, app_dims=args.app_dims)
    elif args.model == 'r3d':
        model = r3d.R3DNet(layer_sizes=(1,1,1,1), motion_dims=args.motion_dims, app_dims=args.app_dims)
    elif args.model=='r21d':
        model = r21d.R2Plus1DNet(layer_sizes=(1,1,1,1), motion_dims=args.motion_dims, app_dims=args.app_dims)

    train_params = [{'params': r21d.get_1x_lr_params(model), 'lr': args.lr},
                    {'params': r21d.get_2x_lr_params(model), 'lr': args.lr * 2}]

    ## define loss and learning schedule
    criterion = nn.MSELoss()
    optimizer = optim.SGD(train_params, lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_s, gamma=args.gamma)
    model.to(device)
    criterion.to(device)

    writer = SummaryWriter(log_dir=log_dir)
    iterations = 0
    model.train()

    for epoch in range(args.epoch_num):

        start_time = time.time()

        for i, sample in enumerate(dataloader):
            video_clips = sample['rgb_clip']
            motion_label = sample['motion_label']
            app_label = sample['app_label']

            video_clips = video_clips.to(device, dtype=torch.float)
            motion_label = motion_label.to(device, dtype=torch.float)
            app_label = app_label.to(device, dtype=torch.float)


            optimizer.zero_grad()
            motion_out, app_out = model(video_clips)
            motion_loss = criterion(motion_out, motion_label) * args.motion_w
            app_loss = criterion(app_out, app_label) * args.app_w
            loss = motion_loss  + app_loss
            loss.backward()
            optimizer.step()

            iterations += 1

            writer.add_scalar('motion_loss', motion_loss, iterations)
            writer.add_scalar('app_loss', app_loss, iterations)
            writer.add_scalar('mse_loss', loss, iterations)


            print("[Epoch{}/{}] Motion Loss: {:.4f} App Loss： {:.4f} Loss: {:4f} Time:{:.4f}".format(
                epoch + 1, i, motion_loss, app_loss, loss, (time.time() - start_time)))

            start_time = time.time()

        scheduler.step()
        model_saver(model, optimizer, epoch, model_save_dir, max_to_keep=args.max_save_num)

    writer.close()


def model_saver(net, optimizer, epoch, model_save_dir, max_to_keep=3):
    tmp_dir = os.listdir(model_save_dir)
    print(tmp_dir)
    tmp_dir.sort()
    if len(tmp_dir) >= max_to_keep:
        os.remove(os.path.join(model_save_dir, tmp_dir[0]))

    torch.save({
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'opt_dict': optimizer.state_dict(),
    }, os.path.join(model_save_dir, 'C3D_epoch-' + '{:02}'.format(epoch + 1) + '.pth.tar'))


if __name__ == '__main__':


    args = parse_args()
    print(args.motion_flag)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train(args)




