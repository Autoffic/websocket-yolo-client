import numpy as np
import collections

from typing import Dict
from typing import Tuple

from Lanes import *

class Line:
    '''
    A equation of line (ax + by + c = 0)
    '''

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

class Point:
    '''
    A point (x, y)
    '''

    def __init__(self, x, y):
        self.x = x
        self.y = y

def calcLine(point1: Point, point2: Point) -> Line: #line's equation calculation
    if point2.y - point1.y == 0:
         a = 0
         b = -1.0
    elif point2.x - point1.x == 0:
        a = -1.0
        b = 0
    else:
        a = (point2.y - point1.y) / (point2.x - point1.x) # slope of the line
        b = -1.0

    c = (-a * point1.x) - b * point1.y
    return Line(a, b, c)

def areLineSegmentsIntersecting(line1: Line, line2: Line,
                        Line1EndPoint1: Point, Line1EndPoint2: Point,
                        Line2EndPoint1: Point, Line2EndPoint2: Point) -> bool:

    det = line1.a * line2.b - line2.a * line1.b
    if det == 0:
        return False #lines are parallel
    else:
        x = (line2.b * -line1.c - line1.b * -line2.c)/det # x coordinate of intersecting point
        y = (line1.a * -line2.c - line2.a * -line1.c)/det # y coordinate of intersecting point

        if x <= max(Line1EndPoint1.x, Line1EndPoint2.x) and x >= min(Line1EndPoint1.x, Line1EndPoint2.x) and \
            y <= max(Line1EndPoint1.y, Line1EndPoint2.y) and y >= min(Line1EndPoint1.y, Line1EndPoint2.y) and \
            x <= max(Line2EndPoint1.x, Line2EndPoint2.x) and x >= min(Line2EndPoint1.x, Line2EndPoint2.x) and \
            y <= max(Line2EndPoint1.y, Line2EndPoint2.y) and y >= min(Line2EndPoint1.y, Line2EndPoint2.y):
            return True #line segments are intersecting inside the line segments
        else:
            return False #lines segments are intersecting but outside of the line segments

def hasPassedTheLineSegment(line: Line, lineStartPoint: Point, lineEndPoint: Point, pointA: Point, pointB: Point) -> bool:
    '''
        Returns true if the points are on different sides of the line segment
    '''
    lineFromPoints = calcLine(pointA, pointB) # another line segment
    return areLineSegmentsIntersecting(line, lineFromPoints, lineStartPoint, lineEndPoint, pointA, pointB)

def passedLane(lanes:Dict[str, Lane], pointA: Tuple[int, int], pointB: Tuple[int, int]) -> str | None:
    '''
        lanes : dictionary laneid and Lane
        pointA : tuple of one point
        pointB : tuple of another point

        Returns the lane id if the points are different sides of a lane
        else returns None
    '''

    pointA: Point = Point(*pointA)
    pointB: Point = Point(*pointB)

    for (laneid, lane) in lanes.items():

        lineStartPoint: Point = Point(*lane.start_point)
        lineEndPoint: Point = Point(*lane.end_point)

        line = calcLine(lineStartPoint, lineEndPoint)

        if hasPassedTheLineSegment(line, lineStartPoint, lineEndPoint, pointA, pointB):
            return laneid
    
    # if the points aren't in the opposite side of any lanes
    return None
    

# test code
if __name__ == "__main__":
    import cv2

    cv2.namedWindow('frame')
    frame = np.zeros((800,800,3), np.uint8)

    while(1):
        frame = np.zeros((800,800,3), np.uint8)

        lanes = Lanes(frame, 3)
        lanesDict = lanes.getAllData()

        points = lanes.getLaneEndingPoints(frame, "Enter Any Two Points")
        
        # drawing lanes
        for laneId, lane in lanesDict.items():
            cv2.putText(frame, laneId, lane.start_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (88, 11, 22) , 2)
            cv2.line(frame, lane.start_point, lane.end_point,(255,0,0),2)

        # drawing points
        for point in points:
            cv2.circle(frame, point, 2, (222, 111, 92), -1)
    
        passingLane = passedLane(lanesDict, points[0], points[1])

        if passingLane is not None:
            print(f"\n The points are in the opposite sides of {passingLane}\n")

            cv2.line(frame, points[0], points[1], (0, 0, 255), 3)
            cv2.circle(frame, points[0], 2, (0, 255, 0), -1)
            cv2.circle(frame, points[1], 2, (0, 255, 0), -1)
        else:
            print(f"\n The points aren't in the opposite of any lanes\n")

        cv2.imshow('frame',frame)

        k = cv2.waitKey(0) & 0xFF

        if k == ord('q'):
            break
        if k == ord('n'):
            continue

    cv2.destroyAllWindows()