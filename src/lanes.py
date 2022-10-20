import numpy as np
import cv2

'''
    img is the source image,
    window_name is the name of window on top of which image is drawn

    returns the array of lines represented by starting and ending coordinates.

    eg: [[(187, 207), (393, 413)], [(358, 111), (504, 318)]], where
                   line1                    line2
'''
def getCountingLine(img, window_name: str):
    lines = []

    preview = None
        
    initialPoint = (-1, -1)

    # either a line is being drawn
    drawing = False

    # mouse callback
    def drawLine(event,x,y,flags,param):
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
            cv2.line(img, initialPoint, (x,y), (255, 0, 0), 2)
            lines.append([initialPoint, (x,y)])


    # set the named window and callback          
    cv2.namedWindow(window_name)

    cv2.setMouseCallback(window_name, drawLine)

    while (True):
        # if we are drawing show preview, otherwise the image
        if drawing:
            cv2.imshow(window_name, preview)
        else :
            cv2.imshow(window_name, img)

        k = cv2.waitKey(1) & 0xFF

        if k == ord('q'):
            break
    
    cv2.destroyAllWindows()

    return lines

def getLaneEndingPoints(img, window_name: str):
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
                if points.__len__ == 0:
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
            cv2.imshow(window_name, img)

        k = cv2.waitKey(1) & 0xFF

        if k == ord('q'):
            break
    
    cv2.destroyAllWindows()

    return points

if __name__=="__main__":
    img =  np.zeros((600, 600, 3), dtype=np.uint8)
    
    lines = getCountingLine(img, "Get Lines")

    for line in lines:
        cv2.line(img, line[0], line[1], (255, 0, 0), 2)
    
    points = getLaneEndingPoints(img, "Get Points")

    for point in points:
        cv2.circle(img, point, 2, (0, 0, 255), -1)

    while (True):
        
        cv2.imshow("After getting lines and points", img)
       
        k = cv2.waitKey(1) & 0xFF

        if k == ord('q'):
            break
    
    cv2.destroyAllWindows()