import  numpy
import cv2

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
