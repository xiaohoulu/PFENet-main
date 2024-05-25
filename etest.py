import os
import time
from operator import add

import cv2
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from lib.PFENet import PFENet
from tools.utils import calculate_metrics, create_dir, seeding

from tools.dataloader import test_dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_data(path: str, name: str):
    name_x = os.path.join(path, name, 'images')
    name_y = os.path.join(path, name, 'masks')
    # x = sorted([os.path.join(name_x, f) for f in os.listdir(name_x) if f.endswith('.jpg') or f.endswith('.png')])
    # y = sorted(os.path.join(name_y, f) for f in os.listdir(name_y) if f.endswith('.jpg') or f.endswith('.png'))
    return name_x, name_y


def process_mask(y_pred):
    y_pred = y_pred[0].cpu().numpy()
    y_pred = np.squeeze(y_pred, axis=0)
    # y_pred = y_pred > 0.5
    # y_pred = y_pred.astype(np.int32)
    y_pred = y_pred * 255
    y_pred = np.array(y_pred, dtype=np.uint8)
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)
    return y_pred


def get_score(metrics_score, test_len):
    f1 = metrics_score[0] / test_len
    iou = metrics_score[1] / test_len
    f2 = metrics_score[2] / test_len
    recall = metrics_score[3] / test_len
    acc = metrics_score[4] / test_len
    mae = metrics_score[5] / test_len

    return f1, iou, f2, recall, acc, mae


def print_score(score, net_name, Dataset, std=None):
    first_row = "{:^20s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}".format(net_name, "Dice", "Iou", "F2",
                                                                                  "Recall", "Acc", "MAE", "M_FPS")
    second_row = "{:^20s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}".format("----------", "----------",
                                                                            "----------", "----------",
                                                                            "----------", "----------",
                                                                            "----------")
    with open("./log/score.txt", 'a') as f:
        f.write('\n' + first_row + '\n')
        f.write(second_row + '\n')
        print(first_row)
        print(second_row)

        for i in range(len(Dataset)):
            if std is None:
                string = ''.join("{:^15.4f}".format(j) for j in score[i])
            else:
                string = ''.join("{:^15s}".format("{:.4f}+{:.4f}").format(j, k) for j, k in zip(score[i], std[i]))
            f.write("{:^20s}".format(Dataset[i]) + string + '\n')
            print("{:^20s}".format(Dataset[i]) + string)


def evaluate(model, save_path, test_loader, size, dataset_name):
    metrics_score_1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []
    test_len = len(test_loader)
    for i, (ori, image, gt, name) in tqdm(enumerate(test_loader), total=len(test_loader)):
        """ Image """
        image_ori = cv2.imread(ori, cv2.IMREAD_COLOR)
        h, w, _ = image_ori.shape
        image_ori = cv2.resize(image_ori, size)
        save_img = image_ori
        image = image.to(device)

        """ Mask """
        mask = cv2.imread(gt, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, size)
        save_mask = mask
        save_mask = np.expand_dims(save_mask, axis=-1)
        save_mask = np.concatenate([save_mask, save_mask, save_mask], axis=2)
        mask = np.expand_dims(mask, axis=0)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.to(device)

        with torch.no_grad():
            """ FPS calculation """
            start_time = time.time()
            p1 = model(image)[-1]
            # p1 = p1[0]
            p1 = torch.sigmoid(p1)

            end_time = time.time() - start_time
            time_taken.append(end_time)

            """ Evaluation metrics """
            score_1 = calculate_metrics(mask, p1)
            metrics_score_1 = list(map(add, metrics_score_1, score_1))
            p = F.interpolate(p1, (h, w), mode='bilinear')
            p = process_mask(p)
            p1 = process_mask(p1)

        """ Save the image - mask - pred """
        line = np.ones((size[0], 10, 3)) * 255
        cat_images = np.concatenate([save_img, line, save_mask, line, p1], axis=1)

        cv2.imwrite(f"{save_path}/{dataset_name}/all/{name}.jpg", cat_images)
        cv2.imwrite(f"{save_path}/{dataset_name}/mask/{name}.jpg", p)

    score = list(get_score(metrics_score_1, test_len))

    mean_time_taken = np.mean(time_taken)
    mean_fps = 1 / mean_time_taken
    score.append(mean_fps)
    # print("Mean FPS: ", mean_fps)
    return score


def test(net_name, weight_name, i):
    """ Seeding """
    seeding(42)

    """ Load the checkpoint """

    model = eval(net_name + "()")
    # model = nn.DataParallel(model)
    model = model.to(device)
    checkpoint_path = "./snapshots/mynetwork-best.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    scores = []

    """ Test dataset """
    path = '/home/ubuntu/xhl_project/dataset/Polyp/TestDataset'
    save_path = f"./results/%s_%d" % (weight_name, i)
    create_dir(save_path)

    size = (352, 352)
    for dataset_name in dataset[0]:
        create_dir(f"{save_path}/{dataset_name}/all")
        create_dir(f"{save_path}/{dataset_name}/mask")
        test_x, test_y = load_data(path, dataset_name)
        test_loader = test_dataset(test_x, test_y, size[0], val=False)
        scores.append(evaluate(model, save_path, test_loader, size, dataset_name))
    print_score(scores, weight_name + '_' + str(i), dataset[0])
    return scores


# return scores


if __name__ == '__main__':
    net = ['PFENet']
    dataset = [['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB'],
               ['CHAMELEON', 'CAMO', 'COD10K']]
    score_arr = []
    for n_name in net:
        for num in range(1):
            score_arr.append(test(n_name, n_name, num))
        if len(score_arr) > 1:
            score_arr = np.array(score_arr)
            score_mean = np.mean(score_arr, axis=0)
            score_std = np.std(score_arr, axis=0)
            print('\n')
            print_score(score_mean, n_name + "_mean", dataset[0], score_std)
