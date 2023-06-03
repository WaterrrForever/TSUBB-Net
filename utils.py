import json
import math
import os
import pickle
import random
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.autograd import Variable
from tqdm import tqdm
import Evaluation_Index as EI


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    train_images_path = []
    val_images_path = []
    every_class_num = []
    juhe_images_path = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]

    images = [os.path.join(root, i) for i in os.listdir(root)
              if os.path.splitext(i)[-1] in supported]

    images.sort()

    every_class_num.append(len(images))

    val_path = random.sample(images, k=int(len(images) * val_rate))

    for img_path in images:
        if img_path in val_path:
            val_images_path.append(img_path)
            juhe_images_path.append(img_path)
        else:
            train_images_path.append(img_path)
            juhe_images_path.append(img_path)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    return train_images_path, val_images_path, juhe_images_path


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)

            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i + 1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def loss_function(inputs, target):
    loss_function1 = nn.MSELoss()
    losses = [loss_function1(inputs[i], target) for i in range(len(inputs))]
    total_loss = sum(losses)
    return total_loss


def initialization(data_loader, device, model, juhe_dataset, num_cluster):
    # datas1 = np.zeros([len(juhe_dataset) * N, 256])
    datas1 = []
    ii = 0
    u_mean1 = []
    model = model.cpu()
    for step, (images, labels) in enumerate(data_loader):
        # images, labels = images.to(device), labels.to(device)

        _, u, _1 = model(images)
        u1 = u
        u1 = [u1[i].cpu() for i in range(len(u1))]
        for i in range(len(u1)):
            u2 = u1[i]
            batch_size, c, w, h = u2.shape
            N = w * h
            if step == 0:
                datas1.append(np.zeros([len(juhe_dataset) * N, c]))
            a = torch.reshape(u2, (batch_size, c, -1))
            a = a.permute(0, 2, 1)
            a = torch.reshape(a, (-1, c))
            datas1[i][ii * batch_size * N:(ii + 1) * batch_size * N] = a.data.numpy()
        ii = ii + 1
    for i in range(4):
        kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(datas1[i])
        u_mean = kmeans.cluster_centers_
        u_mean = torch.from_numpy(u_mean)
        u_mean = Variable(u_mean).cuda()
        u_mean1.append(u_mean)
    return u_mean1


def cmp(u, u_mean, m=2):
    p = []
    u2 = []
    # u_mean = [u_mean[i].cpu() for i in range(len(u_mean))]
    for i in range(len(u)):
        p_ik = 0
        u1 = u[i]
        batch_size, c, w, h = u1.shape
        u1 = torch.reshape(u1, (batch_size, c, -1))
        u1 = u1.permute(0, 2, 1)
        # p = torch.zeros([batch_size, n, num_cluster]).cuda()
        norms = torch.norm(u1[:, :, None] - u_mean[i][None, None, :], dim=-1)  # compute norms
        p_ik_num = 1 / (norms ** (2 / (m - 1)))  # numerator of p_ik
        p_ik_den = p_ik_num.sum(dim=-1)[:, :, None]  # denominator of p_ik
        p_ik = p_ik_num / p_ik_den  # compute p_ik
        p_ik = torch.softmax(p_ik, dim=2)
        p.append(torch.pow(p_ik, 1.5))
        u2.append(u1)
    # print(p[1,:])
    return p, u2


def update_cluster_centers(model, data_loader, device, u_mean, m=2):
    model.eval()
    model.cpu()
    for param in model.parameters():
        param.requires_grad = False
    u_mean1 = []
    u_mean = [u_mean[i].cpu() for i in range(len(u_mean))]
    for i, u_mean2 in enumerate(u_mean):
        num_cluster, c = u_mean2.shape
        denominator = torch.zeros([num_cluster])
        numerator = torch.zeros([num_cluster, c])
        for step, (images, labels) in enumerate(data_loader):
            _, u, _1 = model(images)
            u1 = u[0:4]
            # u_mean = [u_mean[i].cpu() for i in range(len(u_mean))]
            p, u2 = cmp(u1, u_mean)
            # Compute the numerator and denominator
            numerator = numerator + torch.sum(torch.sum((p[i] ** m).unsqueeze(-1) * u2[i].unsqueeze(-2), dim=1), dim=0)
            denominator = denominator + torch.sum(torch.sum(p[i] ** m, dim=1), dim=0)
        u_mean2 = numerator / denominator.unsqueeze(-1)
        u_mean1.append(u_mean2)
    model.cuda()
    model.train()
    for param in model.parameters():
        param.requires_grad = True

    return u_mean1


def train_one_epoch(model, batch_size, optimizer, data_loader, device, epoch, lr_scheduler, u_mean, num_cluster,
                    m=1.5, T=2):
    model.train()
    lambda_reg = 0.0001
    accu_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader1 = data_loader
    data_loader = tqdm(data_loader, file=sys.stdout)
    # u_mean = [u_mean[i].to(device) for i in range(len(u_mean))]
    if epoch % T == 1:
        u_mean = update_cluster_centers(model, data_loader1, device, u_mean)
    # u_mean = update_cluster_centers(model, data_loader1, device, u_mean)
    for step, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)

        sample_num += images[0].shape[0]
        # labels = torch.randn(1, 3, 56, 56).to(device)
        torch.autograd.set_detect_anomaly(True)
        _, u, _1 = model(images)

        ###
        u1 = u[0:4]
        u2 = [u1[i].detach() for i in range(len(u1))]
        u_mean = [u_mean[i].to(device) for i in range(len(u_mean))]
        p, _ = cmp(u2, u_mean, batch_size)
        ###
        # u_mean = [u_mean[i].to(device) for i in range(len(u_mean))]
        p = [p[i].to(device) for i in range(len(p))]
        for i in range(num_cluster):
            re_images, u, sh_image = model(images)
            u1 = u[0:4]
            Lr_k = loss_function(re_images, labels) + loss_function(sh_image, labels) + lambda_reg * torch.norm(
                torch.cat([param.view(-1) for param in model.parameters()]), p=1)
            Lc_k = 0
            for j in range(len(u1)):
                u3 = u1[j]
                batch_size, c, w, h = u3.shape
                u3 = torch.reshape(u3, (batch_size, c, -1))
                u3 = u3.permute(0, 2, 1)
                p_k = p[j][:, :, i] ** m
                u_k = u_mean[j][i, :].unsqueeze(0).unsqueeze(1).expand_as(u3)
                Lc_k = Lc_k + (0.1 ** (3 - j)) * (p_k * (u3 - u_k).pow(2).sum(dim=2)).sum()
            loss = Lr_k + 0.01 * Lc_k
            optimizer.zero_grad()
            loss.backward()
            accu_loss += loss.detach()

            data_loader.desc = "[train epoch {}] loss: {:.3f}, lr: {:.5f}".format(
                epoch,
                accu_loss.item() / (step + 1),
                # accu_num.item() / sample_num,
                optimizer.param_groups[0]["lr"]
            )
            lr = optimizer.param_groups[0]["lr"]

            optimizer.step()
            lr_scheduler.step()

    return accu_loss.item() / (step + 1), lr


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    model.eval()

    mae_metric = EI.MeanAbsoluteError()
    ssim_obj = EI.StructuralSimilarityIndex(window_size=11, channel=3, size_average=True)

    accu_loss = torch.zeros(1).to(device)

    data_loader = tqdm(data_loader, file=sys.stdout)
    with torch.no_grad():
        for step, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)

            re_images, _, sh_images = model(images)
            mae_metric.update(re_images[-1], labels)
            ssim_obj.update(re_images[-1], labels)
            mae_metric.update(sh_images[-1], labels)
            ssim_obj.update(sh_images[-1], labels)

            data_loader.desc = "[valid epoch {}] loss: {:.3f}".format(
                epoch,
                accu_loss.item() / (step + 1)
            )

    return mae_metric, ssim_obj


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):

        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)

            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step

            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())
