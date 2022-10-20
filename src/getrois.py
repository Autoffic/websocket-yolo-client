import cv2
import numpy

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

if __name__=="__main__":
    get_rois("Get Region of interest", 1)