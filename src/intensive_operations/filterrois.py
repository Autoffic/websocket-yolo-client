#!python3
#cython: language_level=3
#cython: infer_types=True

import numpy
import cv2
import cython
from cython import CythonDotParallel


def display_and_wait(frame):

    cv2.imshow("im0s", frame)
    key = cv2.waitKey(0) & 0xff

    # if the `q` key was pressed exit
    if key == ord("q"):
        cv2.destroyAllWindows()

def filter_roi(rois: cython.int[:, ::1], parent_image: cython.uchar[:, :, ::1], interpolation: bool =False, fill_empty: bool=False):
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

    #converting to c types
    number_of_rois: cython.int = rois.shape[0]
    c_interpolation: cython.bint = interpolation
    c_fill_empty: cython.bint = fill_empty

    if(c_interpolation or c_fill_empty):

        if number_of_rois == 1: # if there is only one roi, there is no point in combining different images
            roi = rois[0]
            return numpy.asarray(parent_image[roi[1]:roi[1] + roi[3], roi[0]: roi[0] + roi[2]])

        numpy_array_of_rois_memory_view = numpy.asarray(rois)

        # placeholders for the images (black image)
        if fill_empty: # constructing images of max_height and max_width among the rois 
            image_parts = numpy.zeros((number_of_rois, int(max(numpy_array_of_rois_memory_view[:, 3])), int(max(numpy_array_of_rois_memory_view[:, 2])), parent_image.shape[2]), dtype=numpy.ubyte)

            image_parts_memview: cython.uchar[:,:,:,::1] = image_parts

            i: cython.int = 0
            with cython.nogil:
                for i in range(number_of_rois):  # different parts of the image ( region of interest array )

                    startRoiY: cython.int = rois[i, 1]
                    imgHeight: cython.int = rois[i,3]
                    startRoiX: cython.int = rois[i,0]
                    imgWidth: cython.int = rois[i,2]
                    imgChannel: cython.int = parent_image.shape[2]

                    for j in range(imgHeight):
                        for k in range(imgWidth):
                            for l in range(imgChannel):
                                image_parts_memview[i,j,k,l] = parent_image[startRoiY + j, startRoiX + k, l]
                            
        # using interpolation for resizing, produces distorted image if the image sizes differs a lot
        elif interpolation:
             # very much inefficient
             # using it just to make it work as it's not an ndarray
            image_parts = []
            i: cython.int = 0
            for i in range(number_of_rois):
                image_parts.append(numpy.asarray(parent_image[int(rois[i][1]):int(rois[i][1] + rois[i][3]), int(rois[i][0]): int(rois[i][0] + rois[i][2])]))

            max_height = max(numpy_array_of_rois_memory_view[:, 3])
            max_width = max(numpy_array_of_rois_memory_view[:, 2])

            for i, image in enumerate(image_parts):
                image_parts[i] = cv2.resize(image, [max_height, max_width])

        return_image = numpy.concatenate(image_parts, axis=1)

    else:
        # creating a bigger empty image to place the image parts
        whole_image = numpy.zeros(parent_image.shape, parent_image.dtype)

        for roi in rois: 
            # stitching the part image into the whole image
            whole_image[roi[1]:roi[1] + roi[3], roi[0]: roi[0] + roi[2]] = \
                parent_image[roi[1]:roi[1] + roi[3], roi[0]: roi[0] + roi[2]]

        return_image = whole_image

    return return_image
