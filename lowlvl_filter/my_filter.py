"""
Krystian Lalik
07.11.2020
"""

import cv2 # image tensor object, and way of showing images
import numpy # for uint8


class MyFilter:
    def __init__(self, input_image, kernel_size=5, std_dev=1.0):
        self.input_image = input_image
        image_dimensions = input_image.shape
        self.image_rows = (image_dimensions[0] - 1)
        self.image_cols = (image_dimensions[1] - 1)
        if kernel_size < 3:
            self.kernel_size = 3
        else:
            self.kernel_size = kernel_size
        if (self.kernel_size % 2) == 0:
            self.kernel_size -= 1
        self.distance_from_center = (self.kernel_size - 1) // 2
        self.mean = 0  # did not test with any other mean than [0, 0] and right now it won't even work with any other
        self.std_dev = std_dev

        self.normalization_initialized = False
        self.__calculate_kernel_values()

    def perform_filtration(self):
        # i do not know how to create a nwe image without first taking old input_image as a base
        output_image = self.input_image
        return self.__perform_kernel_calculations(output_image)

    def __perform_kernel_calculations(self, output_image):
        for pixel_x in range(self.image_rows):
            for pixel_y in range(self.image_cols):
                output_image[pixel_x][pixel_y] = self.__perform_kernel_convolution(pixel_x, pixel_y)
        return output_image

    def __perform_kernel_convolution(self, pixel_x, pixel_y):
        convolution_row_start = pixel_x - self.distance_from_center
        convolution_row_end = pixel_x + self.distance_from_center
        convolution_col_start = pixel_y - self.distance_from_center
        convolution_col_end = pixel_y + self.distance_from_center

        convolution_sum = 0
        kernel_x_position = -1
        for image_x_pixel in range(convolution_row_start, convolution_row_end):
            kernel_x_position += 1
            kernel_y_position = -1
            for image_y_pixel in range(convolution_col_start, convolution_col_end):
                kernel_y_position += 1
                # edge handling method is cropping
                if convolution_row_start < 0 or convolution_col_start < 0 or convolution_row_end > self.image_rows or convolution_col_end > self.image_cols:
                    continue
                convolution_sum += (self.input_image[image_x_pixel][image_y_pixel] * int(self.kernel[kernel_x_position][kernel_y_position]))  # not having this cast to int at first, cost me way too much debugging time D:

        output_pixel = convolution_sum / self.kernel_sum
        output_pixel = numpy.uint8(output_pixel)
        return output_pixel

    def __calculate_kernel_values(self):
        self.kernel = [[0 for x in range(self.kernel_size)] for y in range(self.kernel_size)]
        xy_start = self.mean - self.distance_from_center
        xy_end = self.mean + self.distance_from_center
        self.kernel_sum = 0
        for x in range(xy_start, xy_end + 1):
            for y in range(xy_start, xy_end + 1):
                self.kernel[x + self.distance_from_center][y + self.distance_from_center] = self.__calculate_gaussian(x, y)
                self.kernel_sum += self.kernel[x + self.distance_from_center][y + self.distance_from_center]

    def __calculate_gaussian(self, x, y):
        math_e = 2.71828
        math_pi = 3.14156
        gauss_exponent = math_e ** (- ((x**2 + y**2) / (2 * self.std_dev**2)))
        gaussian_value = (1 / (2 * math_pi * self.std_dev**2)) * gauss_exponent
        return self.__normalize_gaussian(gaussian_value)

    # unfortunately this normalization function will not work properly for standard deviations above 1 :(
    # it probably requires some more mathematically complex algorithm to normalize kernel properly
    # also values in proper gaussian kernels are not rising linearly as they are here
    def __normalize_gaussian(self, gaussian_value):
        # normalization variables initialize on first function call
        # first normalized value will be for the furthest value and has to be equal to 1
        if self.normalization_initialized:
            return numpy.uint8(gaussian_value * self.normalization_ratio)
        else:
            self.normalization_initialized = True
            self.normalization_ratio = 1 / gaussian_value
            return numpy.uint8(gaussian_value * self.normalization_ratio)


def prepare_image(image_path):
    image_tensor = cv2.imread(image_path)
    prepared_image = cv2.cvtColor(image_tensor, cv2.COLOR_RGB2GRAY)
    cv2.imshow("grayscale of original image", prepared_image)
    return prepared_image


if __name__ == "__main__":
    filename = "dog.jpg"
    grayscale_image = prepare_image(filename)

    my_filter = MyFilter(grayscale_image, kernel_size=3, std_dev=0.8) # std_dev of 0.3 or smaller will not work
    filtered_image_tensor = my_filter.perform_filtration()

    cv2.imshow("grayscale image after filtration", filtered_image_tensor)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


