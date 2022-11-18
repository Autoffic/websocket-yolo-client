from typing import List, Tuple
import numpy as np
import cv2
from termcolor import colored

CV2_KEY_ENTER = 13 # carriage return = 13


class Lane:
    '''
    Contains start point and end point
    '''
    def __init__(self, start_point = (-1, -1), end_point = (-1, -1)) -> None:
        self.start_point = start_point
        self.end_point = end_point

    def getLaneFromImage(self, img, window_name: str)-> List[List[Tuple[int, int]]]:
        colored_text = colored("\n Press r to reset and retry (getting counting lines) \n \t \
            press enter to confirm the selection\n \t \
            please select only one lane\n", 'cyan')
        print(colored_text)

        img_copy = img.copy()

        line: List[List[Tuple[int, int]]] = []

        preview = None
            
        initialPoint: Tuple[int, int] = (-1, -1)

        # either a line is being drawn
        drawing = False

        # mouse callback
        def drawLine(event,x,y,flags,param) -> None:
            '''
                Mouse callback
            '''

            # accessing variables from enclosing scope
            nonlocal initialPoint, drawing, preview, line

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                initialPoint = (x,y)
                preview = img.copy()
                # it will be a point (line starting and ending at same point)
                cv2.line(preview, initialPoint, (x,y), (0, 255, 0), 2)

            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    preview = img.copy()
                    cv2.line(preview, initialPoint, (x,y), (0, 255, 0), 2)

            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                cv2.line(img_copy, initialPoint, (x,y), (255, 0, 0), 2)
                line.append([initialPoint, (x,y)])

        # set the named window and callback          
        cv2.namedWindow(window_name)

        cv2.setMouseCallback(window_name, drawLine)

        while (True):
            # if we are drawing show preview, otherwise the image
            if drawing:
                cv2.imshow(window_name, preview)
            else :
                cv2.imshow(window_name, img_copy)

            k = cv2.waitKey(1) & 0xFF

            # retrying on mistake
            if k == ord('r'):
                colored_text = colored("\n Retrying on new image \n", 'green')
                print(colored_text)

                line =  self.getLaneFromImage(img, window_name)

                # assigning to object
                self.start_point, self.end_point = line[0]
                return line

            # exiting after selecting all the lines
            if k == CV2_KEY_ENTER:
                # if line doesn't contain two valid points
                if (not line.__len__() == 1) or (not line[0].__len__()==2) or \
                     (not line[0][0].__len__()==2) or (not line[0][1].__len__()==2):

                    colored_text = colored(f"\n Line: {line} \
                        Invalid lane selection please try again\n", 'red')
                    
                    print(colored_text)

                    cv2.destroyAllWindows()

                    line =  self.getLaneFromImage(img, window_name)

                    # assigning to object
                    self.start_point, self.end_point = line[0]
                    return line  

                else: # for confirming the selection uncomment the following
                    '''
                    # adding the lines to display
                    for line in line:
                        cv2.line(img, line[0], line[1], (255, 0, 0), 2)

                    # confirming the selection
                    while (True):
            
                        cv2.imshow("Press Enter to confirm and r to retry: ", img)
                    
                        k = cv2.waitKey(1) & 0xFF

                        if k == ord('q') or k == CV2_KEY_ENTER:
                            break
                        
                        # starting from beginning if retry
                        if k == ord('r'):
                            line =  self.getLaneFromImage(img, window_name)

                            # assigning to object
                            self.start_point, self.end_point = line
                            return line
                    '''
                    cv2.destroyAllWindows()
                    
                    # assigning to object
                    self.start_point, self.end_point = line[0]
                    return line

class Lanes:

    def __init__(self, img: np.ndarray, number_of_lanes: int = 3):
        '''
        img: the source image
        '''

        self.img = img

        #Dictionary of lanes
        self.lanes_dict = {}

        self.number_of_lanes = number_of_lanes

    
    def getCountingLine(self, img: np.ndarray, window_name: str) -> List[List[Tuple[int, int]]]:
        '''
        img is the source image,
        window_name is the name of window on top of which image is drawn

        returns the array of lines represented by starting and ending coordinates.

        eg: [[(187, 207), (393, 413)], [(358, 111), (504, 318)]], where
                    line1                    line2
         '''

        colored_text = colored("\n Press r to reset and retry (getting counting lines) \n \t \
            press enter to confirm the selection\n", 'cyan')
        print(colored_text)

        img_copy = img.copy()

        lines = []

        preview = None
            
        initialPoint = (-1, -1)

        # either a line is being drawn
        drawing = False

        # mouse callback
        def drawLine(event,x: int,y: int, flags, param) -> None:
            # accessing variables from enclosing scope
            nonlocal initialPoint, drawing, preview, lines

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                initialPoint = (x,y)
                preview = img.copy()
                # it will be a point (line starting and ending at same point)
                cv2.line(preview, initialPoint, (x,y), (0, 255, 0), 2)

            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    preview = img.copy()
                    cv2.line(preview, initialPoint, (x,y), (0, 255, 0), 2)

            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                cv2.line(img_copy, initialPoint, (x,y), (255, 0, 0), 2)
                lines.append([initialPoint, (x,y)])


        # set the named window and callback          
        cv2.namedWindow(window_name)

        cv2.setMouseCallback(window_name, drawLine)

        while (True):
            # if we are drawing show preview, otherwise the image
            if drawing:
                cv2.imshow(window_name, preview)
            else :
                cv2.imshow(window_name, img_copy)

            k = cv2.waitKey(1) & 0xFF

            # retrying on mistake
            if k == ord('r'):
                colored_text = colored("\n Retrying on new image \n", 'green')
                print(colored_text)

                return self.getCountingLine(img, window_name=window_name)

            # exiting after selecting all the lines
            if k == CV2_KEY_ENTER: 
                break
        
        cv2.destroyAllWindows()

        return lines

    def getLaneEndingPoints(self, img: np.ndarray, window_name: str) -> List[Tuple[int, int]]:
        '''
        Returns the list of points
        '''
        colored_text = colored("\n Press r to reset and retry (getting lane end points) \n \t \
            press enter to confirm the selection\n", 'cyan')
        print(colored_text)

        img_copy = img.copy()

        points = []
        point_available = False
        preview = None

        # mouse callback
        def drawPoint(event,x,y,flags,param):
            # accessing variables from enclosing scope
            nonlocal points, point_available, preview

            if event == cv2.EVENT_LBUTTONDOWN:
                if (x,y) in points:
                    points.remove((x,y))
                    if points.__len__() == 0:
                        point_available = False
                else:
                    point_available = True
                    points.append((x,y))

                if point_available:
                    preview = img.copy()
                    for point in points:
                        cv2.circle(preview, point, 2, (0, 0, 255), -1)

        # set the named window and callback          
        cv2.namedWindow(window_name)

        cv2.setMouseCallback(window_name, drawPoint)

        while (True):
            if point_available:
                cv2.imshow(window_name, preview)
            else:
                cv2.imshow(window_name, img_copy)

            k = cv2.waitKey(1) & 0xFF

            # retrying on mistake
            if k == ord('r'):
                colored_text = colored("\n Retrying on new image \n", 'green')
                print(colored_text)

                return self.getLaneEndingPoints(self.img, window_name=window_name)

            # exiting after selecting all the points
            if k == CV2_KEY_ENTER: 
                break
        
        cv2.destroyAllWindows()

        return points

    def getAllData(self) -> dict[str, Lane]:
        '''
        returns a dictionary of lane id as key and lane object as value
        '''
        for i in range(self.number_of_lanes):
            new_lane = Lane()
            new_lane.getLaneFromImage(self.img, f"Lane {i}")
            self.lanes_dict[f"lane{i}"] = new_lane
        
        return self.lanes_dict