#-------------------------------------#
#       创建YOLO类
#-------------------------------------#
import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from nets.yolo4_tiny import YoloBody, YoloBodyHalf
from utils.utils import (DecodeBox, letterbox_image, non_max_suppression,
                         yolo_correct_boxes)


#--------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path、classes_path和phi都需要修改！
#   如果出现shape不匹配，一定要注意
#   训练时的model_path、classes_path和phi参数的修改
#--------------------------------------------#
class YOLO(object):
    _defaults = {
        "model_path"        : 'model_data/yolov4_tiny_weights_voc.pth',
        "model_path_prune"        : 'model_data/finetune_prune_model.pth',
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "classes_path"      : 'model_data/voc_classes.txt',
        #-------------------------------#
        #   所使用的注意力机制的类型
        #   phi = 0为不使用注意力机制
        #   phi = 1为SE
        #   phi = 2为CBAM
        #   phi = 3为ECA
        #-------------------------------#
        "phi"               : 0,
        # "model_image_size"  : (416, 416, 3),
        "model_image_size"  : (256, 256, 3),
        "confidence"        : 0.2,
        "iou"               : 0.3,
        "cuda"              : True,
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化YOLO
    #---------------------------------------------------#
    def __init__(self, args):
        self.args = args
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()


    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    #---------------------------------------------------#
    #   获得所有的先验框
    #---------------------------------------------------#
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape([-1, 3, 2])

    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self):
        #---------------------------------------------------#
        #   建立yolov4_tiny模型
        #---------------------------------------------------#
        self.net = YoloBody(len(self.anchors[0]), len(self.class_names), self.phi).eval()
        # print(self.net)

        #---------------------------------------------------#
        #   载入yolov4_tiny模型的权重
        #---------------------------------------------------#
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        self.net.load_state_dict(state_dict)
        print('Finished!')

        if self.cuda:
            # self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

        if self.args.prune:
            self.net = self.prune()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            state_dict = torch.load(self.model_path_prune, map_location=device)
            self.net.load_state_dict(state_dict)

            if self.cuda:
                # self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()
        else:
            print('ori model parameters:', np.sum(np.prod(v.size()) for name, v in self.net.named_parameters()) / 1e6, "M")

        #---------------------------------------------------#
        #   建立特征层解码用的工具
        #---------------------------------------------------#
        self.yolo_decodes = []
        self.anchors_mask = [[3,4,5],[1,2,3]]
        for i in range(2):
            self.yolo_decodes.append(DecodeBox(np.reshape(self.anchors,[-1,2])[self.anchors_mask[i]], len(self.class_names),  (self.model_image_size[1], self.model_image_size[0])))

        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #---------------------------------------------------------#
        image = image.convert('RGB')

        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        if self.letterbox_image:
            crop_img = np.array(letterbox_image(image, (self.model_image_size[1],self.model_image_size[0])))
        else:
            crop_img = image.resize((self.model_image_size[1],self.model_image_size[0]), Image.BICUBIC)
        photo = np.array(crop_img,dtype = np.float32) / 255.0
        photo = np.transpose(photo, (2, 0, 1))
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        #images = [photo]
        images = np.expand_dims(photo, axis=0)

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(images))
            if self.cuda:
                images = images.cuda()

            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            #t1 = time.time()
            outputs = self.net(images)
            #t2 = time.time()
            #print("Inference time with the TensorRT engine: {}".format(t2-t1))
            #print(outputs[0])
            #print(outputs[0].shape)
            #print(outputs[1].shape)
            output_list = []
            for i in range(2):
                output_list.append(self.yolo_decodes[i](outputs[i]))
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(output, len(self.class_names),
                                                    conf_thres=self.confidence,
                                                    nms_thres=self.iou)

            #---------------------------------------------------------#
            #   如果没有检测出物体，返回原图
            #---------------------------------------------------------#
            try:
                batch_detections = batch_detections[0].cpu().numpy()
            except:
                return image

            #---------------------------------------------------------#
            #   对预测框进行得分筛选
            #---------------------------------------------------------#
            top_index = batch_detections[:,4] * batch_detections[:,5] > self.confidence
            top_conf = batch_detections[top_index,4]*batch_detections[top_index,5]
            top_label = np.array(batch_detections[top_index,-1],np.int32)
            top_bboxes = np.array(batch_detections[top_index,:4])
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)

            #-----------------------------------------------------------------#
            #   在图像传入网络预测前会进行letterbox_image给图像周围添加灰条
            #   因此生成的top_bboxes是相对于有灰条的图像的
            #   我们需要对其进行修改，去除灰条的部分。
            #-----------------------------------------------------------------#
            if self.letterbox_image:
                boxes = yolo_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)
            else:
                top_xmin = top_xmin / self.model_image_size[1] * image_shape[1]
                top_ymin = top_ymin / self.model_image_size[0] * image_shape[0]
                top_xmax = top_xmax / self.model_image_size[1] * image_shape[1]
                top_ymax = top_ymax / self.model_image_size[0] * image_shape[0]
                boxes = np.concatenate([top_ymin,top_xmin,top_ymax,top_xmax], axis=-1)

        font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[0], 1)

        for i, c in enumerate(top_label):
            predicted_class = self.class_names[c]
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[self.class_names.index(predicted_class)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[self.class_names.index(predicted_class)])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image


    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        if self.letterbox_image:
            crop_img = np.array(letterbox_image(image, (self.model_image_size[1],self.model_image_size[0])))
        else:
            crop_img = image.resize((self.model_image_size[1],self.model_image_size[0]), Image.BICUBIC)
        photo = np.array(crop_img,dtype = np.float32) / 255.0
        photo = np.transpose(photo, (2, 0, 1))
        images = [photo]

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(images))
            if self.cuda:
                images = images.cuda()

            outputs = self.net(images)
            output_list = []
            for i in range(2):
                output_list.append(self.yolo_decodes[i](outputs[i]))

            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(output, len(self.class_names), conf_thres=self.confidence, nms_thres=self.iou)
            try:
                batch_detections = batch_detections[0].cpu().numpy()
                top_index = batch_detections[:,4] * batch_detections[:,5] > self.confidence
                top_conf = batch_detections[top_index,4]*batch_detections[top_index,5]
                top_label = np.array(batch_detections[top_index,-1],np.int32)
                top_bboxes = np.array(batch_detections[top_index,:4])
                top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)

                if self.letterbox_image:
                    boxes = yolo_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)
                else:
                    top_xmin = top_xmin / self.model_image_size[1] * image_shape[1]
                    top_ymin = top_ymin / self.model_image_size[0] * image_shape[0]
                    top_xmax = top_xmax / self.model_image_size[1] * image_shape[1]
                    top_ymax = top_ymax / self.model_image_size[0] * image_shape[0]
                    boxes = np.concatenate([top_ymin,top_xmin,top_ymax,top_xmax], axis=-1)
            except:
                pass

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                outputs = self.net(images)
                output_list = []
                for i in range(2):
                    output_list.append(self.yolo_decodes[i](outputs[i]))

                output = torch.cat(output_list, 1)
                batch_detections = non_max_suppression(output, len(self.class_names), conf_thres=self.confidence, nms_thres=self.iou)
                try:
                    batch_detections = batch_detections[0].cpu().numpy()
                    top_index = batch_detections[:,4] * batch_detections[:,5] > self.confidence
                    top_conf = batch_detections[top_index,4]*batch_detections[top_index,5]
                    top_label = np.array(batch_detections[top_index,-1],np.int32)
                    top_bboxes = np.array(batch_detections[top_index,:4])
                    top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)

                    if self.letterbox_image:
                        boxes = yolo_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)
                    else:
                        top_xmin = top_xmin / self.model_image_size[1] * image_shape[1]
                        top_ymin = top_ymin / self.model_image_size[0] * image_shape[0]
                        top_xmax = top_xmax / self.model_image_size[1] * image_shape[1]
                        top_ymax = top_ymax / self.model_image_size[0] * image_shape[0]
                        boxes = np.concatenate([top_ymin,top_xmin,top_ymax,top_xmax], axis=-1)
                except:
                    pass
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time


    def prune(self):
        model = YoloBody(len(self.anchors[0]), len(self.class_names), self.phi).eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        model.load_state_dict(state_dict)
        # print('Finished!')
        print('ori model parameters:', np.sum(np.prod(v.size()) for name, v in model.named_parameters()) / 1e6, "M")

        if self.cuda:
            # self.net = nn.DataParallel(self.net)
            model = model.cuda()

        total_list = []
        bn_list = []
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                total_list.append(m.weight.data.shape[0])
                bn_list.append(m.weight.data.clone())

        thre_list = []
        for idx, bn in enumerate(bn_list):
            y, i = torch.sort(bn)
            # thre_index = int(total_list[idx] * 0.75)
            thre_index = int(total_list[idx] * 0.5)
            thre = y[thre_index-1]
            thre_list.append(thre)


        pruned = 0
        cfg = []
        cfg_mask = []
        i = 0

        for k, m in enumerate(model.modules()):
            # print(k, m)
            if isinstance(m, nn.BatchNorm2d):
                weight_copy = m.weight.data.clone()
                mask = weight_copy.abs().gt(thre_list[i]).float().cuda()
                pruned = pruned + mask.shape[0] - torch.sum(mask)
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)
                cfg.append(int(torch.sum(mask)))
                cfg_mask.append(mask.clone())
                # print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                #       format(k, mask.shape[0], int(torch.sum(mask))))
                i += 1
            # elif isinstance(m, nn.MaxPool2d):
            #     cfg.append('M')


        # print('Pre-processing Successful!')

        # Make real prune
        # print(cfg)
        newmodel = YoloBodyHalf(len(self.anchors[0]), len(self.class_names), self.phi).eval()
        newmodel.cuda()

        layer_index = [i for i in range(len(cfg_mask))]
        lll = [i for i in range(3, 15)]
        layer_id_in_cfg = 0
        start_mask = torch.ones(3)
        end_mask = cfg_mask[layer_id_in_cfg]
        # print(model)
        # print(newmodel)
        for [m0, m1] in zip(model.modules(), newmodel.modules()):
            if isinstance(m0, nn.BatchNorm2d):
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                m1.weight.data = m0.weight.data[idx1].clone()
                m1.bias.data = m0.bias.data[idx1].clone()
                m1.running_mean = m0.running_mean[idx1].clone()
                m1.running_var = m0.running_var[idx1].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                    end_mask = cfg_mask[layer_id_in_cfg]
            elif isinstance(m0, nn.Conv2d):
                if layer_id_in_cfg >= 15:
                    continue
                if layer_id_in_cfg in lll:
                    idx0 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    # print('In shape: {:d} Out shape:{:d}'.format(idx0.shape[0], idx1.shape[0]))
                    w = m0.weight.data[:, idx0, :, :].clone()
                    w = w[idx1, :, :, :].clone()
                    m1.weight.data = w.clone()
                    # m1.bias.data = m0.bias.data[idx1].clone()
                else:
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    # print('In shape: {:d} Out shape:{:d}'.format(idx0.shape[0], idx1.shape[0]))
                    w = m0.weight.data[:, idx0, :, :].clone()
                    w = w[idx1, :, :, :].clone()
                    m1.weight.data = w.clone()
                    # m1.bias.data = m0.bias.data[idx1].clone()
            elif isinstance(m0, nn.Linear):
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                m1.weight.data = m0.weight.data[:, idx0].clone()

        torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, './newmodel.pt')
        newmodel.load_state_dict(torch.load('./newmodel.pt', map_location=device)['state_dict'])
        # print(newmodel)
        print('prune model parameters:', np.sum(np.prod(v.size()) for name, v in newmodel.named_parameters()) / 1e6, "M")
        return newmodel


if __name__ == "__main__":
    # yolo = YOLO()
    # print(yolo.net)
    pass
