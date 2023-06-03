import sys
import torch
from tqdm import tqdm
import Evaluation_Index as EI
from utils import loss_function


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler):
    model.train()

    lambda_reg = 0.0001

    accu_loss = torch.zeros(1).to(device)

    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)

        sample_num += images[0].shape[0]
        re_images, shape_images = model(images)
        loss = loss_function(re_images, labels) + loss_function(shape_images, labels) + lambda_reg * torch.norm(
            torch.cat([param.view(-1) for param in model.parameters()]), p=1)

        optimizer.zero_grad()
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            optimizer.param_groups[0]["lr"]
        )
        lr = optimizer.param_groups[0]["lr"]

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()
    return accu_loss.item() / (step + 1), lr


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    model.eval()

    mae_metric = EI.MeanAbsoluteError()
    ssim_obj = EI.StructuralSimilarityIndex(window_size=11, channel=3, size_average=True)

    accu_loss = torch.zeros(1).to(device)  # 累计损失
    data_loader = tqdm(data_loader, file=sys.stdout)
    with torch.no_grad():
        for step, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)

            re_images, shape_images = model(images.to(device))
            mae_metric.update(re_images[-1], labels)
            ssim_obj.update(re_images[-1], labels)
            mae_metric.update(shape_images[-1], labels)
            ssim_obj.update(shape_images[-1], labels)

            data_loader.desc = "[valid epoch {}] loss: {:.3f}".format(
                epoch,
                accu_loss.item() / (step + 1)
            )

    return mae_metric, ssim_obj
