import os
import math
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from visdom import Visdom
import numpy as np
import time

root = r"./"
from model import efficientnet_b0 as create_model
from dataloader import train_loader,val_loader
from utils import read_split_data, train_one_epoch, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = "B0"

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    # 如果存在预训练权重则载入
    model = create_model(num_classes=args.num_classes).to(device)
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后一个卷积层和全连接层外，其他权重全部冻结
            if ("features.top" not in name) and ("classifier" not in name):
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 用于比较成功率
    best_acc = 0.0
    global_step = 0
    for epoch in range(args.epochs):
        # train
        train_loss,train_acc = train_one_epoch(model=model,
                                               optimizer=optimizer,
                                               data_loader=train_loader,
                                               device=device,
                                               epoch=epoch)

        scheduler.step()

        # validate
        val_loss,val_acc = evaluate(model=model,
                                    data_loader=val_loader,
                                    device=device,
                                    epoch=epoch)

        tags = ["train_loss","train_acc","val_loss","val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2],val_loss,epoch)
        tb_writer.add_scalar(tags[3],val_acc,epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)


        viz.line([[train_loss, train_acc]],[global_step],  win='train', update='append')
        viz.line([[val_loss, val_acc]],[global_step],  win='val', update='append')
            #  delay time 0. 5s
        time.sleep(0.5)

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))

        if best_acc < val_acc:
            best_acc = val_acc

            torch.save(model.state_dict(), "./weights/model-best.pth")
        global_step += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    # http://download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="./")

    # download model weights
    # 链接: https://pan.baidu.com/s/1ouX0UmjCsmSx3ZrqXbowjw  密码: 090i
    parser.add_argument('--weights', type=str, default='./weights/model-19.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    viz = Visdom()
    #  create a window and initialize it （创建监听窗口）
    viz.line([[0.0,0.0]], [0.], win='train',
             opts=dict(title='train_loss&train_acc', legend=['train_loss', 'train_acc']))
    viz.line([[0.0,0.0]], [0.], win='val', opts=dict(title='val_loss&val_acc', legend=['val_loss', 'val_acc']))


    opt = parser.parse_args()

    main(opt)
