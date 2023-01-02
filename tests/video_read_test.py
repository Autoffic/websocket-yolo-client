
import argparse
import os
import sys
from pathlib import Path

import time

import logging
# changing logging level won't work unless this is done due to some unknown reason
# if other problem is found, this hacky fix should be removed
#
# using basicConfig to override the setting of log level from other parts (if any)
# logging level isn't changed
logging.basicConfig(level=logging.getLogger().getEffectiveLevel())


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # project directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_file, check_imshow, check_requirements, cv2,
                           print_args)

SRC = FILE.parents[0]  # src directory
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))  # add SRC to PATH
SRC = Path(os.path.relpath(SRC, Path.cwd()))  # relative

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


def run(
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        view_img=False,  # show results
        open_cv_only=False
):

    FPS = get_fps(source)
    LOGGER.info(f"Read FPS: {FPS}")

    # for calculating display fps
    display_previous_time = 0

    if open_cv_only:
        vid_cap = cv2.VideoCapture(str(source))

        fourcc = int(vid_cap.get(cv2.CAP_PROP_FOURCC))
        fourccs = ''.join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
         # for improving fps
        print(f"The format is: {fourccs}")
        
        # vid_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MPEG"))

        print(f"Using backend: {vid_cap.getBackendName()}")

        while(vid_cap.isOpened()):
            ret, im0s = vid_cap.read()

            if ret:
                current_time = time.time()
                time_difference = current_time - display_previous_time

                cv2.putText(im0s,F"FPS: {int(1/time_difference)}", (50, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(50, 50, 250)) 

                cv2.imshow('Testing Video Read Speed', im0s)
                display_previous_time = current_time

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        vid_cap.release()
        cv2.destroyAllWindows()

    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Dataloader
    if webcam:
        view_img = check_imshow()
        dataset = LoadStreams(source, vid_stride=5)
    else:
        dataset = LoadImages(source)
   
    if view_img:
        for path, im, im0s, vid_cap, s in dataset:
            # for improving fps
            # vid_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

            current_time = time.time()
            time_difference = current_time - display_previous_time

            cv2.putText(im0s,F"FPS: {int(1/time_difference)}", (50, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(50, 50, 250))  
            cv2.imshow("Testing Video Read Speed", im0s)
            cv2.waitKey(1)
            display_previous_time = current_time
        
    

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--open-cv-only', action='store_true', help='use opencvs video capture only')
    opt = parser.parse_args()
    print_args(vars(opt))

    return opt


def main(opt: argparse.Namespace):
    check_requirements(exclude=('tensorboard', 'thop'))

    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
