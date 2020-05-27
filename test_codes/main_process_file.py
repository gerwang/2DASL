import argparse
import csv
import os

import cv2
import face_alignment
import openmesh as om
import torch
import torch.jit
import torch.nn as nn
from torch.backends import cudnn

import params
from benchmark_0resnet50_4chls_FAN2d_18pts_1998 import obtain_18pts_map
from ddfa_utils import reconstruct_vertex
from ddfa_utils_inference import parse_roi_box_from_landmark, crop_img
from filter import OneEuroFilter
from params import *
from resnet_xgtu_4chls import resnet50
import sys
from time import time, sleep
from landmark_sensetime.landmark_detector import LandmarkDetector


def convert_param_to_ori(pose, roi_box, img_ori):
    sx, sy, ex, ey = roi_box
    scale_x = (ex - sx) / params.std_size
    scale_y = (ey - sy) / params.std_size
    pose = pose.reshape(3, 4)
    pose[0] *= scale_x
    pose[1] *= scale_y
    pose[0, 3] += sx
    pose[1, 3] += img_ori.shape[0] - ey
    pose = pose.flatten()
    return pose


# total_time = 0
# total_cnt = 0

use_sensetime = True

map_106_to_68_index = list(range(0, 33, 2)) + \
                      list(range(33, 38)) + \
                      list(range(38, 43)) + \
                      list(range(43, 47)) + \
                      list(range(47, 52)) + \
                      list(range(52, 58)) + \
                      list(range(58, 64)) + \
                      list(range(84, 96)) + \
                      list(range(96, 104))


def convert_106_to_68(x):
    return x[map_106_to_68_index]


def process_one_image(fa, model, img_ori):
    # img is RGB image
    if use_sensetime:
        img_bgr = cv2.cvtColor(img_ori, cv2.COLOR_RGB2BGR)
        res = fa.detect_landmarks(img_bgr)
        if not res['lmk_success']:
            print('Warning: No faces were detected.')
            preds = None
        else:
            preds = [convert_106_to_68(res['landmarks'])]
        sleep(0.02)
    else:
        detected_faces = fa.face_detector.detect_from_image(img_ori[::8, ::8, ::-1].copy())
        for face in detected_faces:
            for i in range(4):
                face[i] *= 8
        preds = fa.get_landmarks(img_ori, detected_faces=detected_faces)  # (2, 68)

    '''
    for i in range(68):
        cv2.circle(img_ori, (int(preds[0][i][0]), int(preds[0][i][1])), 1, (0, 255, 0))
    img_renderer = cv2.cvtColor(img_ori, cv2.COLOR_RGB2BGR)
    cv2.imshow('frame', img_renderer)
    cv2.waitKey(0)
    '''
    if preds is None:
        return np.array([]), np.array([])
    inputs = []
    roi_boxes = []
    for pred in preds:
        # img_render = cv2.cvtColor(img_ori, cv2.COLOR_RGB2BGR)
        # img_render = cv2.resize(img_render, (img_render.shape[1] * 2, img_render.shape[0] * 2))
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # fontScale = 0.6
        # fontColor = (0, 255, 0)
        # lineType = 2
        # for k in range(pred.shape[0]):
        #     cv2.circle(img_render, (int(pred[k, 0]) * 2, int(pred[k, 1]) * 2), 2, (0, 0, 255), -1)
        #     cv2.putText(img_render, str(k), (int(pred[k, 0]) * 2, int(pred[k, 1]) * 2), font,
        #                 fontScale,
        #                 fontColor,
        #                 lineType)
        # cv2.imshow('frame', img_render)
        # cv2.waitKey(0)

        pred = pred.T
        roi_box = parse_roi_box_from_landmark(pred)
        img = crop_img(img_ori, roi_box)

        mean = 127.5
        std = 128
        img_normalized = (img.astype(np.float32) - mean) / std

        cropped_pred = pred - np.expand_dims(np.array(roi_box[:2]), axis=1)
        cropped_pred[0, :] *= params.std_size / img_normalized.shape[1]
        cropped_pred[1, :] *= params.std_size / img_normalized.shape[0]
        cropped_pred[cropped_pred > 119] = 119
        lms_map = obtain_18pts_map(cropped_pred)

        '''
        cv2.imshow('frame', np.expand_dims((lms_map + 1) / 2 * 255, axis=2).repeat(1, 1, 3))
        cv2.waitKey(0)
        '''

        img_normalized = cv2.resize(img_normalized, dsize=(params.std_size, params.std_size),
                                    interpolation=cv2.INTER_LINEAR)

        comb_input = np.concatenate([img_normalized, np.expand_dims(lms_map.astype(np.float32), axis=2)],
                                    axis=2)
        comb_input = comb_input.transpose([2, 0, 1])
        inputs.append(torch.from_numpy(comb_input).unsqueeze(0))
        roi_boxes.append(roi_box)
        # cv2.imshow('frame', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
    if len(inputs) == 0:
        return np.zeros((0, 62))
    inputs = torch.cat(inputs, dim=0).cuda()
    with torch.no_grad():
        output = model(inputs)
    param_prediction = output.cpu().numpy()
    # param_prediction[i, :12] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
    # print('resnet: {}'.format(time() - start_time))
    return param_prediction, roi_boxes


class TupleWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return tuple(self.module(x))


class ListWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return list(self.module(x))


def init_model():
    with torch.jit.optimized_execution(True):
        if use_sensetime:
            fa = LandmarkDetector()
        else:
            fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, face_detector='sfd')
            fa.face_alignment_net = ListWrapper(torch.jit.trace_module(TupleWrapper(fa.face_alignment_net), {
                'forward': torch.randn(1, 3, 256, 256, dtype=torch.float32, device=torch.device('cuda:0'))
            }))
            fa.face_detector.face_detector = ListWrapper(
                torch.jit.trace_module(TupleWrapper(fa.face_detector.face_detector), {
                    'forward': torch.randn(1, 3, 60, 80, dtype=torch.float32, device=torch.device('cuda:0'))
                }))

        device_ids = [0]
        checkpoint_fp = '../models/2DASL_checkpoint_epoch_allParams_stage2.pth.tar'
        num_classes = 62
        map_location = {f'cuda:{i}': 'cuda:0' for i in range(8)}
        checkpoint = torch.load(checkpoint_fp, map_location=map_location)['res_state_dict']
        torch.cuda.set_device(device_ids[0])
        model = resnet50(pretrained=False, num_classes=num_classes)
        model = nn.DataParallel(model, device_ids=device_ids)
        model.load_state_dict(checkpoint)
        model = model.module.cuda()
        model = torch.jit.script(model)
        # model = torch.jit.trace_module(model, {
        #     'forward': torch.randn(1, 4, 120, 120, dtype=torch.float32, device=torch.device('cuda:0'))})

    cudnn.benchmark = True
    model.eval()
    return fa, model


def load_template_mesh():
    template_path = r'../3D_results_plot/image00074.obj'
    mesh = om.read_trimesh(template_path)
    return mesh


def render_img(img_ori, param_prediction):
    param_prediction = param_prediction.flatten()
    M = param_prediction[:12].reshape(3, 4)
    p = M[:, :3]
    offset = M[:, 3:]
    alpha_shp = param_prediction[12:52].reshape(-1, 1)
    alpha_exp = param_prediction[52:].reshape(-1, 1)

    vertex = (u + w_shp @ alpha_shp + w_exp @ alpha_exp)
    vertex = p @ vertex.reshape(-1, 3).T + offset
    vertex = vertex.T
    for i in range(vertex.shape[0]):
        # cv2.circle(img_ori, (int(vertex[i, 0]), img_ori.shape[0] + 1 - int(vertex[i, 1])), radius=1, color=(0, 255, 0))
        cv2.circle(img_ori, (int(vertex[i, 0]), img_ori.shape[0] - int(vertex[i, 1])), radius=1, color=(0, 255, 0))
    return img_ori


def main(args):
    fa, model = init_model()
    mesh = load_template_mesh()
    if args.write_csv:
        csv_writer = csv.writer(open(os.path.join(args.output, 'params.csv'), 'w'))
        csv_writer.writerow(
            ['name'] + ['pose_{}'.format(i) for i in range(12)] + ['identity_{}'.format(i) for i in range(40)] + [
                'exp_{}'.format(i) for i in range(10)])
    else:
        csv_writer = None
    if args.mode == 'camera':
        cap = cv2.VideoCapture(0)
        frame_idx = 0
        filters = []
        prev_track_success = False
        while cap.isOpened():

            # start_time = time()

            ret, img_ori = cap.read()

            # end_time = time()
            # print('read image: {}'.format(end_time - start_time))
            # start_time = end_time

            if not ret:
                break
            img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
            param_prediction, roi_boxes = process_one_image(fa, model, img_ori)  # only process one face currently
            if param_prediction.shape[0] < 1:
                prev_track_success = False
                if len(filters) != 0:
                    for filter in filters:
                        filter.reset()
                continue
            if len(filters) == 0:
                filters = [OneEuroFilter(mincutoff=10.0) for _ in range(param_prediction.shape[1])]
            if prev_track_success:
                for i in range(len(filters)):
                    param_prediction[0][i] = filters[i].process(param_prediction[0][i])
            prev_track_success = True
            param_prediction = param_prediction * param_std + param_mean
            for i in range(param_prediction.shape[0]):
                param_prediction[i, :12] = convert_param_to_ori(param_prediction[i, :12], roi_boxes[i], img_ori)
            '''
            cv2.imshow('{}'.format(frame_idx), img_rendered)
            img_crop = render_img(img_crop, param_prediction)
            cv2.imshow('{}'.format(frame_idx), img_crop)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            '''
            print(','.join(map(str, param_prediction.flatten())))
            sys.stdout.flush()
            if args.write_csv:
                for i in range(param_prediction.shape[0]):
                    output_name = '{}_{}'.format(frame_idx, i)
                    csv_writer.writerow([output_name] + param_prediction[i].tolist())
            frame_idx += 1
            # print('predict time: {}'.format(time() - start_time))
    elif args.mode == 'video':
        cap = cv2.VideoCapture(args.input)
        frame_idx = 0
        filters = []
        while cap.isOpened():
            ret, img_ori = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(args.output, '{}.png'.format(frame_idx)), img_ori)
            img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
            param_prediction = process_one_image(fa, model, img_ori)  # only process one face currently
            if param_prediction.shape[0] < 1:
                continue
            if len(filters) == 0:
                filters = [OneEuroFilter() for _ in range(param_prediction.shape[1])]
            for i in range(len(filters)):
                param_prediction[0][i] = filters[i].process(param_prediction[0][i])
            img_rendered = render_img(img_ori, param_prediction[0])
            img_rendered = cv2.cvtColor(img_rendered, cv2.COLOR_RGB2BGR)
            img_rendered = cv2.resize(img_rendered, (img_rendered.shape[1] // 2, img_rendered.shape[0] // 2))
            cv2.imwrite(os.path.join(args.output, 'render_{}.png'.format(frame_idx)), img_rendered)
            '''
            cv2.imshow('{}'.format(frame_idx), img_rendered)
            img_crop = render_img(img_crop, param_prediction)
            cv2.imshow('{}'.format(frame_idx), img_crop)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            '''
            for i in range(param_prediction.shape[0]):
                output_name = '{}_{}'.format(frame_idx, i)
                csv_writer.writerow([output_name] + param_prediction[i].tolist())
                vertex = reconstruct_vertex(param_prediction[i], dense=True)
                mesh.points()[:] = vertex.T
                om.write_mesh(os.path.join(args.output, output_name + '.obj'), mesh)
            frame_idx += 1
    elif args.mode == 'images':
        file_list = os.listdir(args.input)
        for file_name in file_list:
            img_ori = cv2.imread(os.path.join(args.input, file_name))
            img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
            param_prediction = process_one_image(fa, model, img_ori)
            img_rendered = render_img(img_ori, param_prediction)
            img_rendered = cv2.cvtColor(img_rendered, cv2.COLOR_RGB2BGR)
            cv2.imshow(file_name, img_rendered)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            for i in range(param_prediction.shape[0]):
                output_name = '{}_{}'.format(file_name, i)
                csv_writer.writerow([output_name] + param_prediction[i].tolist())
                vertex = reconstruct_vertex(param_prediction[i], dense=True)
                mesh.points()[:] = vertex.T
                om.write_mesh(os.path.join(args.output, output_name + '.obj'), mesh)
    else:
        raise ValueError('unknown mode: {}'.format(args.mode))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('process files')
    parser.add_argument('--input', help='input directory path')  # image no order, video has order
    parser.add_argument('--output', help='output directory')
    parser.add_argument('--mode', help='input mode', choices=['video', 'images', 'camera'], default='video')
    parser.add_argument('--write_csv', action='store_true')
    args = parser.parse_args()
    main(args)
