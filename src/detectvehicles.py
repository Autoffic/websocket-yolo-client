# YOLOv5 by Ultralytics, GPL-3.0 license

"""

script tailored for vehicle detection

This scripts utilized dlib's correlation tracker to for object dection most of the times
initially, object is detected using yolo and then it will be tracked over certain period 
again giving detection to yolo.

default_classes is used in this script to filter out the classes, this is overidden by 
command line arguments, this can also be changed according to coco.names

There is also option to select roi in the first frame.

"""



"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detectvehicles.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detectvehicles.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s.xml                # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle

Usage - inference per second
    $ python detectvehicles.py --interence-per-second 5

Usage - ROI:
    $ python detectvehicles.py --number-of-rois 0           # whole image is the region of interest
                                                1           # only one region of interest
                                                {other}     # multiple ROIs
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import dlib
from termcolor import colored

import torch

import numpy
import csv

from yolov5.utils.augmentations import letterbox


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # project directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, smart_inference_mode

SRC = FILE.parents[0]  # src directory
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))  # add SRC to PATH
SRC = Path(os.path.relpath(SRC, Path.cwd()))  # relative

from emittowebsocket import connect, disconnect, emit

from filterrois import filter_roi
from getrois import get_rois
from Lanes import *

from centroidtracker import CentroidTracker

from emitifpassed import passedLane

FPS = 24  # assuming 24 fps from standard sources

# the default classes to detect
default_classes=[1,2,3,5,7]

def get_fps(source: str) -> int:
    '''
    Returns the fps of the video stream, 
    if it's not possible to get the fps, default fps is returned 

    @param source: the video source
    '''

    # Finding the fps of the video if it's not a webcam
    if source != "0":
        video = cv2.VideoCapture(source)
        
        # Find OpenCV version
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

        if int(major_ver)  < 3 :
            fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        else :
            fps = video.get(cv2.CAP_PROP_FPS)
        video.release()

        return fps
    return FPS


def display_and_wait(frame):

    cv2.imshow("im0s", frame)
    key = cv2.waitKey(0) & 0xff

    # if the `q` key was pressed exit
    if key == ord("q"):
        cv2.destroyAllWindows()


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=default_classes,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        inference_per_second=4, # inference per second
        number_of_rois=0,
        web_socket=False, # whether or not to transfer the data via websocket
        number_of_lanes=3, # total number of lanes to select
        disable_centroid_tracking=False, # to disable centroid tracking
        read_inputs_from_csv=False, # When user inputs have been saved in csv formats
        inference_only=False # to find bounding box based on inference only
):

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(str(weights.resolve()), device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    FPS = get_fps(source)
        
    # Running inference inference_per_second times a second 
    # here time is relative to video i.e. fps frames is compared to 1 second
    FRAMES_TO_SKIP = FPS - inference_per_second

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    # for profiling dlib
    dlib_profiler = Profile()
    seen_dlib = 0

    # for profiling image operations
    im_profilers = Profile(), Profile(), Profile()

    # keeping track of frame counts
    total_frames = 0

    # this will be a list of list containing tracker, class name and confidence
    # for simplicity the confidence is same as the object detection confidence
    trackers = []
    lanes = Lanes(numpy.zeros((600, 600, 3), numpy.uint8)) # black image (just to make lanes globally accessible)
    rois = []

    if not webcam:
        # finding out the source name
        seperator = os.sep
        video_name = source.rsplit(seperator, 1)[1]
        video_name_without_extension = video_name.rsplit(".", 1)[0]
        read_success_roi = False
        read_success_lane = False

    if read_inputs_from_csv and not webcam:

        rois_file_location = Path(str(ROOT) + "/resources/files/rois").resolve()
        rois_file = Path(str(rois_file_location) + f"/{video_name_without_extension}.csv").resolve()

        if rois_file.exists():
            read_success_roi = True            

            # rois
            with open(str(rois_file), mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file) # reader doesn't read the first line and treats it as key
                for row in csv_reader:
                    rois.append([float(row["start_point_x"]), float(row["start_point_y"]), float(row["width"]), float(row["height"])])
            number_of_rois = rois.__len__()
        else:
            print("\nCouldn't find roi csv file for given video.\n")

        lanes_file_location = Path(str(ROOT) + "/resources/files/lanes").resolve()
        lanes_file = Path(str(lanes_file_location) + f"/{video_name_without_extension}.csv").resolve()

        if lanes_file.exists():
            read_success_lane = True

            # lanes
            with open(str(lanes_file), mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file) # this reader doesn't read the first line and treats it as key
                for row in csv_reader:
                    lane = Lane()
                    lane.start_point = (int(row["start_point_x"]), int(row["start_point_y"]))
                    lane.end_point = (int(row["end_point_x"]), int(row["end_point_y"]))
                    lanes.lanes_dict[row["lane_id"]] = lane
        else:
            print("\nCouldn't find lane csv file for given video.\n")

    if not disable_centroid_tracking:
        # for proifiling centroid tracker
        ct_profiler = Profile()

        # for profiling the lane passing detector
        lp_profiler = Profile()

        ct=CentroidTracker(maxDisappeared=5,maxDistance=50)

        # keep track of object in previous step
        previous_objects = {}

    for path, im, im0s, vid_cap, s in dataset:
        
        if not disable_centroid_tracking:
            # bounding boxes for centroid tracking
            rects = []
            # objects dictionary for centroid tracking
            objects = {}

        # selecting roi in the very first frame
        if (total_frames == 0):

            if number_of_rois > 0:
                if webcam or not read_success_roi: # if reading from csv isn't successful
                    rois = get_rois("Press Esc after selecting all the rois", im0s[0] if webcam else im0s, number_of_rois)

                if webcam:
                    im0s[0] = filter_roi(rois, im0s[0], interpolation=False, fill_empty=True)
                else:
                    im0s = filter_roi(rois, im0s, interpolation=False, fill_empty=True)

            colored_text = colored("\nGetting the lanes in order\n", 'green')
            print(colored_text)

            if webcam or not read_success_lane: # if reading from csv isn't successful
                lanes = Lanes(im0s[0].copy() if webcam else im0s, number_of_lanes) # overriding the above lanes value 

                # asking for lanes in roi selected image
                lanes.getAllData()

        if number_of_rois > 0:
            with im_profilers[0]:
                # this is already done above so skipping in the very first frame
                if not (total_frames == 0):
                    img = filter_roi(rois, im0s[0] if webcam else im0s, interpolation=False, fill_empty=True)
                    if webcam:
                        im0s[0] = img
                    else:
                        im0s = img

                # drawing the lanes
                for lane_name, lane in lanes.lanes_dict.items():
                    cv2.putText(im0s[0] if webcam else im0s, lane_name, lane.start_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (88, 11, 22) , 2)
                    cv2.line(im0s[0] if webcam else im0s, lane.start_point, lane.end_point, (255, 0, 0), 2)

                '''
                a copy of code from utils.dataloaders to resize the im accordingly
                '''
                if webcam:
                    im0 = im0s.copy()
                    im = np.stack([letterbox(x, imgsz, stride=stride, auto=pt)[0] for x in im0])  # resize
                    im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
                    im = np.ascontiguousarray(im)  # contiguous
                else:
                    #**********************************************************************
                    im0 = im0s.copy()
                    # transformation isn't implemented here
                    im = letterbox(im0, imgsz, stride=stride, auto=pt)[0]  # padded resize
                    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                    im = numpy.ascontiguousarray(im)  # contiguous
                    #**********************************************************************
        else:
            # drawing the lanes
            for lane_name, lane in lanes.lanes_dict.items():
                cv2.putText(im0s[i] if webcam else im0s, lane_name, lane.start_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (88, 11, 22) , 1)
                cv2.line(im0s if webcam else im0s, lane.start_point, lane.end_point, (255, 0, 0), 2)

        # since open'c cv's defuault channel is bgr and that of dlib's is rgb
        rgb = cv2.cvtColor(im0s[0] if webcam else im0s, cv2.COLOR_BGR2RGB)

        # doing inference only on few frames
        inferencing = inference_only or not FRAMES_TO_SKIP or ((total_frames % int(FRAMES_TO_SKIP)) == 0)
        if inferencing:

            with dt[0]:
                im = torch.from_numpy(im).to(device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop


                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det): 
                    # Rescale boxes from img_size to im0 size
                    with im_profilers[2]:
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    trackers = []
                    print("\n******Using computationally intensive object detection**************")
                    for *xyxy, conf, cls in reversed(det):

                        if not disable_centroid_tracking:
                            # updating the bounding box list for centroid tracking
                            rects.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], cls])

                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                        if not inference_only:
                            # construct a dlib rectangle object from the bounding
                            # box coordinates and then start the dlib correlation
                            # tracker
                            tracker = dlib.correlation_tracker()
                            rect = dlib.rectangle(xyxy[0], xyxy[1], xyxy[2], xyxy[3])
                            tracker.start_track(rgb, rect)                                               

                            # add the tracker to our list of trackers so we can
                            # utilize it during skip frames
                            # class name name and confidence is also added to keep it similar to the predictions
                            trackers.append([tracker, conf, cls])                           
                        
        # determining the bounding box using correlation tracker
        else:

            if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
            
            # loop over the trackers and append the predicitons
            pred = []
            with dlib_profiler:  # for profiling dlib's tracker
                for tracker, conf, cls in trackers:
                    # update the tracker and grab the updated position
                    tracker.update(rgb)
                    pos = tracker.get_position()

                    # unpack the position object
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())

                    if not disable_centroid_tracking:
                        # updating bounding box list for centroid tracking
                        rects.append([startX, startY, endX, endY, cls])

                    pred.append([startX, startY, endX, endY, float(conf), float(cls)])
            
            pred = torch.tensor(pred)
            pred = [pred]

            for i, det in enumerate(pred):  # per image
                seen_dlib += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop

                annotator = Annotator(im0, line_width=line_thickness, example=str(names))

                if len(det): 

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):

                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))

                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

        if not disable_centroid_tracking:
            # updating the centroid tracker
            
            # profiling the centroid tracking process
            with ct_profiler:
                objects = ct.update(rects)

            '''
                The most cpu intensive process in this loop is determining whether some centroids
                have passed any of the lanes

                so profiling here is intended to profile lane passing check
            '''
            with lp_profiler:
                for (objectID, centroid) in objects.items():

                    className = names[int(ct.idToClass[objectID])]

                    # drawing the centroids and ids
                    if save_img or save_crop or view_img: 
                        # draw both the ID of the object and the centroid of the
                        # object on the output frame
                        text = "ID {}".format(objectID)
                        cv2.putText(im0, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.circle(im0, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

                    
                    object_centroid_in_this_step = centroid
                    object_centroid_in_previous_step = previous_objects.get(objectID)

                    if object_centroid_in_previous_step is not None:
                        passedlane = passedLane(lanes.lanes_dict, object_centroid_in_this_step, object_centroid_in_previous_step)

                        if passedlane is not None and web_socket:
                            emit([passedlane, className])
            
        # displaying if the bounding boxes are obtained through detection or tracking
        if save_img or save_crop or view_img: 
            annotator.box_label([10, 10, 10 + 20, 10 + 30], "Detecting" if inferencing else "Tracking", color=colors(9, True))

        # Stream results
        im0 = annotator.result()
        with im_profilers[1]:
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)

                cv2.waitKey(1) # wait 1ms

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
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)
        total_frames += 1

        if inferencing:
            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        else:
            # Print the time taken by dlib tracking
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '} Dlib Tracking: {dlib_profiler.dt* 1E3:.1f}ms ")
        
        if number_of_rois > 0:
            LOGGER.info(f"Image operation (filtering): {im_profilers[0].dt * 1E3:.1f}ms")
            if len(det):
                LOGGER.info(f"Image operation (scaling): {im_profilers[2].dt * 1E3:.1f}ms")
            if view_img:
                LOGGER.info(f"Image operation (displaying)): {im_profilers[1].dt * 1E3:.1f}ms")

        if not disable_centroid_tracking:

            # Print the time taken by centroid tracking and lane pass checking
            LOGGER.info(f"Centroid Tracking : {ct_profiler.dt * 1E3:.1f}ms, Lane Pass Checking: {lp_profiler.dt * 1E3:.1f}ms ")

            previous_objects = objects.copy()

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    t1 = tuple(x.t / seen_dlib * 1E3 for x in (dlib_profiler, ct_profiler, lp_profiler))
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    LOGGER.info(f'Speed: %.1fms dlib-tracking, %.1fms centroid-tracking, %.1fms lane passing checking per image' % t1)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, default=default_classes, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--inference-per-second', type=int, default=4, help='run inference this many times a second')
    parser.add_argument('--number-of-rois', type=int, default=0, help='number of region of interests')
    parser.add_argument('--web-socket', action='store_true', default=False, help='whether or not to transfer data through websocket')
    parser.add_argument('--number-of-lanes', type=int, default=3, help='total number of lanes to select in the given source')
    parser.add_argument('--disable-centroid-tracking', action='store_true', default=False, help="Either to disable centroid tracking")
    parser.add_argument('--read-inputs-from-csv', action="store_true", default=False, help="When user inputs have been saved in csv formats")
    parser.add_argument('--inference-only', action='store_true', default=False, help="Either to run only on inference")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))

    return opt


def main(opt: argparse.Namespace):
    check_requirements(exclude=('tensorboard', 'thop'))

    if opt.web_socket:
        connect()

    run(**vars(opt))

    if opt.web_socket:
        disconnect()

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
