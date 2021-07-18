"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import json
from json import JSONEncoder


"""class JsonEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


def toJSON(object):
    return json.dumps(object, default=lambda o: o.__dict__,
                      sort_keys=True, indent=4)
def obj_dict(obj):
    return obj.__dict__

class coordinate:
    def __init__(self, x=0, y=0):
        self.x = x,
        self.y = y
    def dump(self):
        return {"x": self.x, "y": self.y}

class sinirlayici_kutu:
    def __init__(self, ust_sol, alt_sag):
        self.ust_sol = ust_sol
        self.alt_sag = alt_sag

    def dump(self):
        return {"ust_sol": self.ust_sol.dump(), "alt_sag": self.alt_sag.dump()}

class DetectedObject:
    def __init__(self, sinif, inis_durumu, sinirlayici_kutu):
        self.sinif = sinif,
        self.inis_durumu = inis_durumu,
        self.sinirlayici_kutu = sinirlayici_kutu
    def dump(self):
        return {"sinif": self.sinif, "inis_durumu": str(self.inis_durumu), "sinirlayici_kutu": self.sinirlayici_kutu.dump()}

class DetectionGroup:
    def __init__(self, frame_id):
        self.frame_id = frame_id
        self.nesneler = []  # DetectedObject List

    def add_detected_object(self, d_object):
        self.nesneler.append(d_object)

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,indent=4)

    def dump(self):
        return {"frame_id": self.frame_id, "nesneler": [do.dump() for do in self.nesneler]}

"""

@torch.no_grad()
def run_detect(weights='yolov5s.pt',  # model.pt path(s)
               source='data/images',  # file/dir/URL/glob, 0 for webcam
               imgsz=640,  # inference size (pixels)
               conf_thres=0.25,  # confidence threshold
               iou_thres=0.45,  # NMS IOU threshold
               max_det=1000,  # maximum detections per image
               device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
               view_img=False,  # show results
               save_txt=False,  # save results to *.txt
               save_json=False,  # save results to *.json
               save_conf=False,  # save confidences in --save-txt labels
               save_crop=False,  # save cropped prediction boxes
               nosave=False,  # do not save images/videos
               classes=None,  # filter by class: --class 0, or --class 0 2 3
               agnostic_nms=False,  # class-agnostic NMS
               augment=False,  # augmented inference
               visualize=False,  # visualize features
               update=False,  # update all models
               project='runs/detect',  # save results to project/name
               name='exp',  # save results to project/name
               exist_ok=False,  # existing project/name ok, do not increment
               line_thickness=3,  # bounding box thickness (pixels)
               hide_labels=False,  # hide labels
               hide_conf=False,  # hide confidences
               half=False,  # use FP16 half-precision inference
               ):
    check_requirements(exclude=('tensorboard', 'thop'))
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt or save_json else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet50', n=2)  # initialize
        modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    if save_json:
        jdict = []
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img,
                     augment=augment,
                     visualize=increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if save_json:
                detection_group_list = {}
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                if save_json:
                    detection_group = []
                    #detection_group = DetectionGroup(p.stem if dataset.mode == "image" else frame)
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    if save_json:  # Write to file
                        height, width, *_ = im0.shape
                        inis_durumu = -1
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        """ust_coordinate = coordinate(round((xywh[0] - xywh[2] / 2) * width), round((xywh[1] - xywh[3] / 2) * height))
                        alt_coordinate = coordinate(round((xywh[0] + xywh[2] / 2) * width), round((xywh[1] + xywh[3] / 2) * height))
                        detected_object = DetectedObject(int(cls), inis_durumu, sinirlayici_kutu(ust_coordinate, alt_coordinate))"""
                        detection_group.append({"sinif": int(cls),
                                                "inis_durumu": inis_durumu,
                                                "sinirlayici_kutu": {
                                                    "ust_sol": {
                                                        "x": round((xywh[0] - xywh[2] / 2) * width),
                                                        "y": round((xywh[1] - xywh[3] / 2) * height)
                                                    },
                                                    "alt_sag": {
                                                        "x": round((xywh[0] + xywh[2] / 2) * width),
                                                        "y": round((xywh[1] + xywh[3] / 2) * height)
                                                    }
                                                }
                                                })
                        #detection_group.add_detected_object(detected_object)
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                if save_json:
                    detection_group_list = {
                        "frame_id": p.stem if dataset.mode == "image" else frame,
                        "nesneler": detection_group
                    }
                    jdict.append(detection_group_list)
                    #detection_group_list.append(detection_group)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    if save_json:
        with open(f"{save_dir}/labels/{p.stem}.json", "w") as f:
            """for a in detection_group_list:
                f.write(a.toJSON())"""
            #f.write(str([o.toJSON() for o in detection_group_list]))
            wlist = {"frame_list": jdict}
            json.dump(wlist, f, indent=4)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(f'Done. ({time.time() - t0:.3f}s)')
