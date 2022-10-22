import sys
from typing import NoReturn
from pathlib import Path
import cv2
import numpy

# to make source folder available

module_path = Path(__file__).resolve()
project_path = module_path.parents[1]

if str(project_path) not in sys.path:
    sys.path.append(str(project_path))  # adding project folder to path

from src.Lanes import Lanes

CV2_KEY_ENTER = 13

def test() -> NoReturn:
    test_img = numpy.zeros((600, 600, 3), numpy.uint8)
    test_lanes = Lanes(test_img)

    lanes_dictionary = test_lanes.getAllData()
    
    # for determining color
    c = 110
    for key, value in lanes_dictionary.items():
        lane_name = key
        lane_start = value.start_point
        lane_end = value.end_point

        color = (c * 1 % 255, c * 2 % 255, c * 3 % 255)
        
        cv2.putText(test_img, lane_name, lane_start, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
        cv2.line(test_img, lane_start, lane_end, color, 2)

        c = c + 100

    while (True):
        cv2.imshow("After getting lanes", test_img)
        
        k = cv2.waitKey(1) & 0xFF

        if k == ord('q') or k == CV2_KEY_ENTER:
            break
        
    cv2.destroyAllWindows()

if __name__=="__main__":

    test()