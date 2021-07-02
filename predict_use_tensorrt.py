import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import time
from PIL import Image
import cv2
import torchvision

import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from nets.yolo4_tiny import YoloBody, YoloBodyHalf
from utils.utils import DecodeBox, letterbox_image, non_max_suppression, yolo_correct_boxes

import time
import argparse
import cv2
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser("test")
parser.add_argument('--test_mode', type=str, default='camera', help='location of the data corpus')
parser.add_argument('--prune', action='store_true', default=False, help='use auxiliary tower')
args = parser.parse_args()

filename = 'img/street.jpg'
max_batch_size = 1
onnx_model_path = 'yolo_v4_tiny_prune.onnx'

TRT_LOGGER = trt.Logger()


class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolov4_tiny_weights_voc.pth',
        "model_path_prune": 'model_data/finetune_prune_model.pth',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/voc_classes.txt',

        "phi": 0,

        "model_image_size": (256, 256, 3),
        "confidence": 0.2,
        "iou": 0.3,
        "cuda": True,

        "letterbox_image": False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, args):
        self.args = args
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape([-1, 3, 2])

    def generate(self):

        self.yolo_decodes = []
        self.anchors_mask = [[3, 4, 5], [1, 2, 3]]
        for i in range(2):
            self.yolo_decodes.append(
                DecodeBox(np.reshape(self.anchors, [-1, 2])[self.anchors_mask[i]], len(self.class_names),
                          (self.model_image_size[1], self.model_image_size[0])))

        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    def detect_image(self, image):

        image = image.convert('RGB')

        image_shape = np.array(np.shape(image)[0:2])

        if self.letterbox_image:
            crop_img = np.array(letterbox_image(image, (self.model_image_size[1], self.model_image_size[0])))
        else:
            crop_img = image.resize((self.model_image_size[1], self.model_image_size[0]), Image.BICUBIC)
        photo = np.array(crop_img, dtype=np.float32) / 255.0
        photo = np.transpose(photo, (2, 0, 1))

        images = np.ascontiguousarray(photo, dtype=np.float32)
        images = np.expand_dims(images, axis=0)

        with torch.no_grad():

            outputs = run(images)

            outputs_new = []
            for i in range(len(outputs)):
                if i == 0:
                    outputs_new.append(
                        torch.from_numpy(np.asarray(outputs[i].reshape((1, 75, 8, 8)), dtype=np.float32)))
                if i == 1:
                    outputs_new.append(
                        torch.from_numpy(np.asarray(outputs[i].reshape((1, 75, 16, 16)), dtype=np.float32)))
            output_list = []
            for i in range(2):
                output_list.append(self.yolo_decodes[i](outputs_new[i]))

            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(output, len(self.class_names),
                                                   conf_thres=self.confidence,
                                                   nms_thres=self.iou)

            try:
                batch_detections = batch_detections[0].cpu().numpy()
            except:
                return image

            top_index = batch_detections[:, 4] * batch_detections[:, 5] > self.confidence
            top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
            top_label = np.array(batch_detections[top_index, -1], np.int32)
            top_bboxes = np.array(batch_detections[top_index, :4])
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(
                top_bboxes[:, 1], -1), np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)

            if self.letterbox_image:
                boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                           np.array([self.model_image_size[0], self.model_image_size[1]]), image_shape)
            else:
                top_xmin = top_xmin / self.model_image_size[1] * image_shape[1]
                top_ymin = top_ymin / self.model_image_size[0] * image_shape[0]
                top_xmax = top_xmax / self.model_image_size[1] * image_shape[1]
                top_ymax = top_ymax / self.model_image_size[0] * image_shape[0]
                boxes = np.concatenate([top_ymin, top_xmin, top_ymax, top_xmax], axis=-1)

        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

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
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        if self.letterbox_image:
            crop_img = np.array(letterbox_image(image, (self.model_image_size[1], self.model_image_size[0])))
        else:
            crop_img = image.resize((self.model_image_size[1], self.model_image_size[0]), Image.BICUBIC)
        photo = np.array(crop_img, dtype=np.float32) / 255.0
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
            batch_detections = non_max_suppression(output, len(self.class_names), conf_thres=self.confidence,
                                                   nms_thres=self.iou)
            try:
                batch_detections = batch_detections[0].cpu().numpy()
                top_index = batch_detections[:, 4] * batch_detections[:, 5] > self.confidence
                top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
                top_label = np.array(batch_detections[top_index, -1], np.int32)
                top_bboxes = np.array(batch_detections[top_index, :4])
                top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(
                    top_bboxes[:, 1], -1), np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)

                if self.letterbox_image:
                    boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                               np.array([self.model_image_size[0], self.model_image_size[1]]),
                                               image_shape)
                else:
                    top_xmin = top_xmin / self.model_image_size[1] * image_shape[1]
                    top_ymin = top_ymin / self.model_image_size[0] * image_shape[0]
                    top_xmax = top_xmax / self.model_image_size[1] * image_shape[1]
                    top_ymax = top_ymax / self.model_image_size[0] * image_shape[0]
                    boxes = np.concatenate([top_ymin, top_xmin, top_ymax, top_xmax], axis=-1)
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
                batch_detections = non_max_suppression(output, len(self.class_names), conf_thres=self.confidence,
                                                       nms_thres=self.iou)
                try:
                    batch_detections = batch_detections[0].cpu().numpy()
                    top_index = batch_detections[:, 4] * batch_detections[:, 5] > self.confidence
                    top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
                    top_label = np.array(batch_detections[top_index, -1], np.int32)
                    top_bboxes = np.array(batch_detections[top_index, :4])
                    top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(
                        top_bboxes[:, 1], -1), np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3],
                                                                                                    -1)

                    if self.letterbox_image:
                        boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                                   np.array([self.model_image_size[0], self.model_image_size[1]]),
                                                   image_shape)
                    else:
                        top_xmin = top_xmin / self.model_image_size[1] * image_shape[1]
                        top_ymin = top_ymin / self.model_image_size[0] * image_shape[0]
                        top_xmax = top_xmax / self.model_image_size[1] * image_shape[1]
                        top_ymax = top_ymax / self.model_image_size[0] * image_shape[0]
                        boxes = np.concatenate([top_ymin, top_xmin, top_ymax, top_xmax], axis=-1)
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

        print('ori model parameters:', np.sum(np.prod(v.size()) for name, v in model.named_parameters()) / 1e6, "M")

        if self.cuda:
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

            thre_index = int(total_list[idx] * 0.5)
            thre = y[thre_index - 1]
            thre_list.append(thre)

        pruned = 0
        cfg = []
        cfg_mask = []
        i = 0

        for k, m in enumerate(model.modules()):

            if isinstance(m, nn.BatchNorm2d):
                weight_copy = m.weight.data.clone()
                mask = weight_copy.abs().gt(thre_list[i]).float().cuda()
                pruned = pruned + mask.shape[0] - torch.sum(mask)
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)
                cfg.append(int(torch.sum(mask)))
                cfg_mask.append(mask.clone())

                i += 1

        newmodel = YoloBodyHalf(len(self.anchors[0]), len(self.class_names), self.phi).eval()
        newmodel.cuda()

        layer_index = [i for i in range(len(cfg_mask))]
        lll = [i for i in range(3, 15)]
        layer_id_in_cfg = 0
        start_mask = torch.ones(3)
        end_mask = cfg_mask[layer_id_in_cfg]

        for [m0, m1] in zip(model.modules(), newmodel.modules()):
            if isinstance(m0, nn.BatchNorm2d):
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                m1.weight.data = m0.weight.data[idx1].clone()
                m1.bias.data = m0.bias.data[idx1].clone()
                m1.running_mean = m0.running_mean[idx1].clone()
                m1.running_var = m0.running_var[idx1].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]
            elif isinstance(m0, nn.Conv2d):
                if layer_id_in_cfg >= 15:
                    continue
                if layer_id_in_cfg in lll:
                    idx0 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))

                    w = m0.weight.data[:, idx0, :, :].clone()
                    w = w[idx1, :, :, :].clone()
                    m1.weight.data = w.clone()

                else:
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))

                    w = m0.weight.data[:, idx0, :, :].clone()
                    w = w[idx1, :, :, :].clone()
                    m1.weight.data = w.clone()

            elif isinstance(m0, nn.Linear):
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                m1.weight.data = m0.weight.data[:, idx0].clone()

        torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, './newmodel.pt')
        newmodel.load_state_dict(torch.load('./newmodel.pt', map_location=device)['state_dict'])

        print('prune model parameters:', np.sum(np.prod(v.size()) for name, v in newmodel.named_parameters()) / 1e6,
              "M")
        return newmodel


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(device_mem))

        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def get_engine(max_batch_size=1, onnx_file_path="", engine_file_path="", \
               fp16_mode=False, int8_mode=False, save_engine=False,
               ):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine(max_batch_size, save_engine):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(explicit_batch) as network, \
                trt.OnnxParser(network, TRT_LOGGER) as parser:

            builder.max_workspace_size = 1 << 30
            builder.max_batch_size = max_batch_size

            builder.fp16_mode = fp16_mode
            builder.int8_mode = int8_mode
            if int8_mode:
                raise NotImplementedError

            if not os.path.exists(onnx_file_path):
                quit('ONNX file {} not found'.format(onnx_file_path))

            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            network.get_input(0).shape = [max_batch_size, 3, 256, 256]

            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))

            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            save_engine = True
            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):

        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(max_batch_size, save_engine)


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return [out.host for out in outputs]


def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs


fp16_mode = True
int8_mode = False
trt_engine_path = './model_fp16_{}_int8_{}.trt'.format(fp16_mode, int8_mode)
engine = get_engine(max_batch_size, onnx_model_path, trt_engine_path, fp16_mode, int8_mode)
context = engine.create_execution_context()
inputs, outputs, bindings, stream = allocate_buffers(engine)


def run(frame):
    inputs[0].host = frame
    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    return trt_outputs


if __name__ == "__main__":
    yolo = YOLO(args)
    mode = args.test_mode
    video_path = 0
    video_path = 'img/1.mp4'
    video_save_path = ""
    video_fps = 25.0

    if mode == "img":
        img = 'img/street.jpg'
        image = Image.open(img)
        r_image = yolo.detect_image(image)
        r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        fps = 0.0
        while (True):
            t1 = time.time()
            ref, frame = capture.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(yolo.detect_image(frame))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break
        capture.release()
        out.release()
        cv2.destroyAllWindows()

    elif mode == "camera":
        capture = cv2.VideoCapture(0)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        fps = 0.0
        while (True):
            t1 = time.time()
            ref, frame = capture.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(yolo.detect_image(frame))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break
        capture.release()
        out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        test_interval = 100
        img = Image.open('img/street.jpg')
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video' or 'fps'.")
