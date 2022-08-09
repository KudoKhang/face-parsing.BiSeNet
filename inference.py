#!/usr/bin/python
# -*- encoding: utf-8 -*-

from model import BiSeNet
from modules import FaceDetection
import torch
import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm

facedetector = FaceDetection()

def crop(image, bbox):
    return image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

def transform_area_bbox(bbox):
    left, top, right, bottom = bbox[:4]
    old_size = (right - left + bottom - top) / 2
    center_x = right - (right - left) / 2.0
    center_y = bottom - (bottom - top) / 2.0 + old_size * 0.5
    size = int(old_size * 3)

    roi_box = [0] * 4
    roi_box[0] = int(center_x - size / 2)
    roi_box[1] = int(center_y - size / 2)
    roi_box[2] = int(roi_box[0] + size)
    roi_box[3] = int(roi_box[1] + size)
    return roi_box

def restore_image(image_ori, result, bbox):
    size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
    result = cv2.resize(result, size, interpolation = cv2.INTER_LINEAR)
    image_ori[bbox[1]:bbox[3], bbox[0]:bbox[2]] = result
    return image_ori

def vis_parsing_maps(im_ori, im, bbox, parsing_anno, stride, save_im=False, save_path='results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    final_image = restore_image(im_ori, vis_im, bbox)
    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path, final_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def evaluate(result_folder='./results/image_test/', image_f='./data', checkpoint='model_final_diss.pth'):
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cpu()
    net.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    path_folder = [name for name in os.listdir(image_f) if name.endswith(('jpg', 'png', 'jpeg'))]
    with torch.no_grad():
        for image_path in tqdm(path_folder):
            img = cv2.imread(osp.join(image_f, image_path))
            image_ori = img.copy()
            bbox = facedetector.run(img)[0]
            bbox = transform_area_bbox(bbox)
            img = crop(img, bbox)
            image = cv2.resize(img, (512, 512), interpolation = cv2.INTER_LINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cpu()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            vis_parsing_maps(image_ori, image, bbox, parsing, stride=1, save_im=True, save_path=osp.join(result_folder, image_path))

if __name__ == "__main__":
    evaluate(image_f='src/image_test', checkpoint='model/weights/79999_iter.pth')


