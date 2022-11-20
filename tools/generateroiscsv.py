import argparse
import os
import sys
from pathlib import Path
import cv2

import csv

# to make source folder available

module_path = Path(__file__).resolve()
project_path = module_path.parents[1]

if str(project_path) not in sys.path:
    sys.path.append(str(project_path))  # adding project folder to path

default_video = Path(str(project_path) + "/resources/videos/production ID 4979730.mp4").resolve()
default_output_file_folder = Path(str(project_path) + "/resources/files/rois/")

from src.getrois import get_rois
from src.filterrois import filter_roi

from generatelanescsv import generatelanescsv

CV2_KEY_ENTER = 13

def generateroiscsv(
    source: str = str(default_video),
    number_of_rois:int = 1,
    output_file: str = str(default_output_file_folder),
    generate_lanes_csv: bool = False,
    number_of_lanes: int = 3
) -> None:

    # some conversions
    source = str(source)
    output_file = str(output_file)
    number_of_rois = int(number_of_rois)

    seperator = os.sep

    video_name = source.rsplit(seperator, 1)[1]
    video_name_without_extension = video_name.rsplit(".", 1)[0] 

    if output_file == str(default_output_file_folder):
        output_file  = Path(str(default_output_file_folder) + "/" + video_name_without_extension + ".csv") #getting the video name

    video =  cv2.VideoCapture(source)

    if (video.isOpened() == False):
        print("Error opening video file")
        return
    
    ret, frame = video.read()
    
    if ret==True:
        rois = get_rois("Press Esc after selection all the ROIs", frame=frame, number=number_of_rois)
        
        #saving to csv
        with open(output_file, "w") as fileToWrite:
            roi_writer = csv.writer(fileToWrite, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            # column names
            roi_writer.writerow(['row_id','start_point_x','start_point_y','width','height'])

            print("\n Writing rois coordinates to file: ", str(output_file))

            for i, roi in enumerate(rois):
                roi_writer.writerow([i,roi[0],roi[1],roi[2],roi[3]])

        frame = filter_roi(rois=rois, parent_image=frame, interpolation=False, fill_empty=True)
          
        while (True):
            
            cv2.imshow("After getting rois", frame)
            
            k = cv2.waitKey(1) & 0xFF

            if k == ord('q') or k == CV2_KEY_ENTER:
                if generate_lanes_csv:
                    generatelanescsv(frame, number_of_lanes,
                        output_file=Path(str(project_path) + f"/resources/files/lanes/{video_name_without_extension}.csv").resolve(),
                        generate_from_image=True)
                break
            
        cv2.destroyAllWindows()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=default_video, help='the video file from which lane is to be extracted')
    parser.add_argument('--number-of-rois', type=int, default= 1, help='the number of rois')
    parser.add_argument('--generate-lanes-csv', action="store_true", default=False, help="whether to generate lanes csv after generating rois csv")
    parser.add_argument('--number-of-lanes', type=int, default=3, help="number of lanes to add after selecting rois")

    opt = parser.parse_args()
    return opt

   
def main(opt: argparse.Namespace):
    generateroiscsv(**vars(opt))

if __name__=="__main__":
    opt = parse_opt()
    main(opt)