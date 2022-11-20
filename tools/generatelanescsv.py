import argparse
import os
import sys
from pathlib import Path
import cv2
from numpy import float64, ndarray
import csv

# to make source folder available

module_path = Path(__file__).resolve()
project_path = module_path.parents[1]

if str(project_path) not in sys.path:
    sys.path.append(str(project_path))  # adding project folder to path

default_video = Path(str(project_path) + "/resources/videos/production ID 4979730.mp4").resolve()
default_output_file_folder = Path(str(project_path) + "/resources/files/lanes/")

from src.Lanes import Lanes

CV2_KEY_ENTER = 13

def generatelanescsv(
    source: str | ndarray | ndarray[float64]  = str(default_video),
    number_of_lanes:int = 3,
    output_file = str(default_output_file_folder),
    generate_from_image = False
) -> None:

    if not generate_from_image:
        source = str(source)

    output_file = str(output_file)
    number_of_lanes = int(number_of_lanes)

    seperator = os.sep

    if not generate_from_image:
        video_name = source.rsplit(seperator, 1)[1]
        video_name_without_extension = video_name.rsplit(".", 1)[0] 

    if output_file == str(default_output_file_folder) and not generate_from_image:
        output_file  = Path(str(default_output_file_folder) + "/" + video_name_without_extension + ".csv") #getting the video name
    elif output_file == str(default_output_file_folder):
        output_file = Path(str(default_output_file_folder) + "/" + "filtered-img-csv.csv") # default name
    
    if generate_from_image:
        ret, frame = True, source # loading the image directly
    else:
        video =  cv2.VideoCapture(source)

        if (video.isOpened() == False):
            print("Error opening video file")
            return
        
        ret, frame = video.read()
    
    
    if ret==True:
        lanes= Lanes(frame, number_of_lanes=number_of_lanes)

        lanes_dictionary = lanes.getAllData()
        
        #saving to csv
        with open(output_file, "w") as fileToWrite:
            lanes_writer = csv.writer(fileToWrite, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            # column names
            lanes_writer.writerow(['lane_id','start_point_x','start_point_y','end_point_x','end_point_y'])

            print("\n Writing lanes coordinates to file: ", str(output_file))

            # for determining color
            c = 110
            for key, value in lanes_dictionary.items():
                lane_name = key
                lane_start = value.start_point
                lane_end = value.end_point

                color = (c * 1 % 255, c * 2 % 255, c * 3 % 255)
                
                cv2.putText(frame, lane_name, lane_start, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
                cv2.line(frame, lane_start, lane_end, color, 2)

                c = c + 100
                
                lanes_writer.writerow([lane_name, lane_start[0], lane_start[1], lane_end[0], lane_end[1]])
            
        while (True):
            cv2.imshow("After getting lanes", frame)
            
            k = cv2.waitKey(1) & 0xFF

            if k == ord('q') or k == CV2_KEY_ENTER:
                break
            
        cv2.destroyAllWindows()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default= default_video, help='the video file from which lane is to be extracted')
    parser.add_argument('--number-of-lanes', type=int, default= 3, help='the number of lanes')

    opt = parser.parse_args()
    return opt

   
def main(opt: argparse.Namespace):
    generatelanescsv(**vars(opt))

if __name__=="__main__":
    opt = parse_opt()
    main(opt)