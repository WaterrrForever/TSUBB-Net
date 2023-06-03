import argparse
import datetime
import os

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from first_train_eval import train_one_epoch, evaluate
from model import Waterrr_pre
from my_dataset import MyDataSet
from utils import read_split_data, create_lr_scheduler, get_params_groups


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    train_images_path, val_images_path = read_split_data(args.data_path)

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     # transforms.RandomHorizontalFlip(), # 以给定的概率水平翻转
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5], [0.5])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   # transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5], [0.5])])}
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_dataset = MyDataSet(images_path=train_images_path,
                              transform=data_transform["train"])

    val_dataset = MyDataSet(images_path=val_images_path,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    nw = 0
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = Waterrr_pre().to(device)

    pg = get_params_groups(model, weight_decay=args.wd)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)

    current_mae, current_ssim = 1.0, -1.0
    for epoch in range(args.epochs):
        # train
        train_loss, lr = train_one_epoch(model=model,
                                         optimizer=optimizer,
                                         data_loader=train_loader,
                                         device=device,
                                         epoch=epoch,
                                         lr_scheduler=lr_scheduler)

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}

        if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:
            mae_metric, ssim_obj = evaluate(model, val_loader, device=device, epoch=epoch)
            mae_info, ssim_info = mae_metric.compute(), ssim_obj.compute()
            print(f"[epoch: {epoch}] val_MAE: {mae_info:.3f} ssim_info: {ssim_info:.3f}")
            # write into txt
            with open(results_file, "a") as f:
                write_info = f"[epoch: {epoch}] train_loss: {train_loss:.4f} lr: {lr:.6f} " \
                             f"MAE: {mae_info:.3f} ssim_info: {ssim_info:.3f} \n"
                f.write(write_info)

            # save_best
            if current_mae >= mae_info and current_ssim <= ssim_info:
                torch.save(save_file, "weights/model_best.pth")

        # torch.save(save_file, f"weights/model_{epoch}.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=5e-2)
    parser.add_argument('--data-path', type=str,
                        default="/home/omnisky/Desktop/papercode/data/fabric/train/D0")
    parser.add_argument('--weights', type=str, default='./convnext_tiny_1k_224_ema.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument("--eval-interval", default=10, type=int, help="validation interval default 10 Epochs")
    opt = parser.parse_args()

    main(opt)
