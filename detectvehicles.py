# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

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
import optparse
import os
import platform
import sys
from pathlib import Path
import dlib

import torch

import numpy

from yolov5.utils.augmentations import letterbox

import socketio

# websocket stuffs
socketio_client = socketio.Client()
yolo_event = 'YOLO_EVENT'

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, smart_inference_mode


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

def get_rois(title: str, frame, number: int):
    '''
    Opens up the roi selection window and return multiple region of interest coordinates.
    The coordinates is in the form [ [xStart, yStart, width height], ...

    @param title: the title of the opened roi selection window
    @param frame: the frame for roi selection
    @param number: number of rois

    if the number is 0 then whole image is roi
    if the number is 1 then only one roi is allowed
     '''
    if number == 0:
        rois = [0, 0, frame.shape[0], frame.shape[1]]
        return rois
    elif number == 1:
        coordinates = cv2.selectROI(title, frame, showCrosshari=False)
        rois = coordinates
    else:
        coordinates = cv2.selectROIs(title, frame, showCrosshair=False)

        rois = numpy.zeros((number, 4)) # [ [xStart, yStart, width, height], ...
        for i, roi in enumerate(coordinates):
            rois[i]=roi

    cv2.destroyWindow(title)

    return rois

def filter_roi(rois, parent_image, interpolation: bool=False, fill_empty: bool=False):
    '''
    Filters out the roi from the image.
    
    if @param: interpolation is True then
    image is interpolated to make height and width of each rois same
    
    if @parem: interpolation is False and
       @param: fill_empty is True then for resizing, the empty pixels is made black.
       Here, the empty pixels means those pixels of sub-images which is smaller than
       the image of maximum height and width

    if @parem: interpolation is False and
       @param: fill_empty is also false then the images aren't resized,
       the original dimension is used but the region except roi is made black

    @param roi: array of region of interests in the format [xStart, yStart, width, height]
    @param parent_image: the image on which roi was selected
    '''

    if(interpolation or fill_empty):
        # cropping the image
        image_parts = []
        max_height = 0
        max_width = 0
        total_width = 0
  
        for roi in rois:  # different parts of the image ( region of interest array )

            image_parts.append(parent_image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]): int(roi[0] + roi[2])])

            # this is done for resizing all the images to the same size
            height, width =  image_parts[-1].shape[:2]
            max_height = max(max_height, height)
            max_width = max(max_width, width)
            total_width += width

        
        # using interpolation for resizing, produces distorted image if the image sizes differs a lot
        if interpolation:
            for i, image in enumerate(image_parts):
                image_parts[i] = cv2.resize(image, [max_height, max_width])

        elif fill_empty:
            resized_image_array = []
            for image_part in image_parts:

                # the dimension and channel of image part
                img_height = int(image_part.shape[0])
                img_width = int(image_part.shape[1])
                img_channel = image_part.shape[2]

                # appending a fully black image of max_height and max_width
                resized_image_array.append(numpy.zeros((max_height, max_width, img_channel), image_part.dtype))

                # only copying the filled parts and leaving the rest as the dark image

                resized_image_array[-1][0:img_height, 0:img_width] = image_part[0:img_height, 0:img_width]

            image_parts = resized_image_array

        return_image = numpy.concatenate(image_parts, axis=1)

    else:
        # creating a bigger empty image to place the image parts
        whole_image = numpy.zeros(parent_image.shape, parent_image.dtype)

        for roi in rois: 
            # stitching the part image into the whole image
            whole_image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]): int(roi[0] + roi[2])] = \
                parent_image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]): int(roi[0] + roi[2])]

        return_image = whole_image

    return return_image

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
        web_socket=False # whether or not to transfer the data via websocket
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
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
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
    FRAMES_TO_SKIP = FPS / inference_per_second

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    # keeping track of frame counts
    total_frames = 0

    # this will be a list of list containing tracker, class name and confidence
    # for simplicity the confidence is same as the object detection confidence
    trackers = []
    rois = []  # Just to make this global, ensure it is initialized before filtering the roi

    for path, im, im0s, vid_cap, s in dataset:

        # selecting roi in the very first frame
        if (total_frames == 0) and number_of_rois > 0:
            rois = get_rois("Press Esc after selecting all the rois", im0s, number_of_rois)

        if number_of_rois > 0:
            im0s = filter_roi(rois, im0s, interpolation=False, fill_empty=True)

            '''
            a copy of code from utils.dataloaders to resize the im accordingly
            only valid for video stream, not for webcam and images
            '''
            #**********************************************************************
            im0 = im0s.copy()
            # transformation isn't implemented here
            im = letterbox(im0, imgsz, stride=stride, auto=pt)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = numpy.ascontiguousarray(im)  # contiguous
            #**********************************************************************

        # since open'c cv's defuault channel is bgr and that of dlib's is rgb
        rgb = cv2.cvtColor(im0s, cv2.COLOR_BGR2RGB)

        # doing inference only on few frames
        if not FRAMES_TO_SKIP or ((total_frames % int(FRAMES_TO_SKIP)) == 0):

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
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    trackers = []
                    print("\n******Using computationally intensive object detection**************")
                    for *xyxy, conf, cls in reversed(det):
                        if web_socket:
                            socketio_client.emit(yolo_event, str([xyxy, conf, cls, "Inferenced"]))

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
            for tracker, conf, cls in trackers:
                # update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                pred.append([startX, startY, endX, endY, float(conf), float(cls)])
            
            pred = torch.tensor(pred)
            pred = [pred]

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

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                         
                        if web_socket:
                            socketio_client.emit(yolo_event, str([xyxy, conf, cls, "Tracking"]))
                        
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
                       

        
        # displaying if the bounding boxes are obtained through detection or tracking
        annotator.box_label([10, 10, 10 + 20, 10 + 30], "Detecting" if (not FRAMES_TO_SKIP or total_frames % FRAMES_TO_SKIP == 0) else "Tracking", color=colors(9, True))

        # Stream results
        im0 = annotator.result()
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

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
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
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))

    return opt


def main(opt: argparse.Namespace):
    check_requirements(exclude=('tensorboard', 'thop'))

    if opt.web_socket:
        socketio_client.connect('http://localhost:5000')

    run(**vars(opt))

    if opt.web_socket:
        socketio_client.disconnect()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)