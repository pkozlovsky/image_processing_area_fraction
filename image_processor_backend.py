
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

class ImageProcessor:
    def __init__(self, image, threshold=200, dilatation_element_size=4, dilate=False, area_correction_factor=0.7):
        self.image = image
        self.threshold = threshold
        self.dilatation_element_size = dilatation_element_size
        self.dilate = dilate
        self.image_gray = self.image.convert('L')
        self.image_thresholded = None
        self.image_dilated = None
        self.black_area_percentage = None
        self.area_correction_factor = area_correction_factor

    def threshold_image(self):
        self.image_thresholded = self.image_gray.point(lambda x: 0 if x < self.threshold else 255, '1')

    def dilate_image(self):
        if self.dilate:
            structure_element = np.ones((self.dilatation_element_size, self.dilatation_element_size))
            image_array = np.array(self.image_thresholded)
            dilated = binary_dilation(image_array == 0, structure_element).astype(np.uint8) * 255
            self.image_dilated = np.invert(dilated)
        else:
            self.image_dilated = self.image_thresholded  # Use thresholded image directly if no dilation

    def calculate_black_area_percentage_obsolete(self):
        image_to_use = self.image_dilated if self.image_dilated is not None else self.image_thresholded
        black_pixels = np.sum(image_to_use == 0)
        total_pixels = image_to_use.size
        self.black_area_percentage = (black_pixels / total_pixels) * 100

           
    def calculate_black_area_percentage(self):
        # Ensure image_to_use is a numpy array for analysis
        image_to_use = np.array(self.image_dilated) if self.dilate else np.array(self.image_thresholded)
    
        # Debugging: Print the shape and data type of the image array
        print(f"Image shape: {image_to_use.shape}, Type: {image_to_use.dtype}")

        # Count black pixels (assuming black is 0)
        black_pixels = np.sum(image_to_use == 0)
    
        # Calculate the total number of pixels
        total_pixels = image_to_use.size
    
        # Calculate the percentage of black area
        self.black_area_percentage = (black_pixels / total_pixels) * 100 * self.area_correction_factor
    
        # Debugging: Print the count of black pixels and the total pixels
        print(f"Black Pixels: {black_pixels}, Total Pixels: {total_pixels}")


    def process_image(self):
        self.threshold_image()
        self.dilate_image()
        self.calculate_black_area_percentage()

    def display_images(self):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(self.image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(self.image_thresholded, cmap='gray')
        plt.title('Thresholded Image')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        if self.dilate:
            plt.imshow(self.image_dilated, cmap='gray')
            plt.title('Dilated Segment')
        else:
            plt.imshow(self.image_thresholded, cmap='gray')
            plt.title('Segment After Processing')
        plt.axis('off')

        plt.tight_layout()
        plt.show()



class ImageSplitter:
    def __init__(self, file_path, rows, cols, image_width_um, image_height_um):
        self.file_path = file_path
        self.rows = rows
        self.cols = cols
        self.image_width_um = image_width_um
        self.image_height_um = image_height_um
        self.image = Image.open(file_path)
        self.resolution_x = self.image.width / self.image_width_um
        self.resolution_y = self.image.height / self.image_height_um
        self.segments = []
    
    def split_image(self):
        # Calculate the width and height of each segment in pixels
        cell_width_pixels = self.image.width // self.cols
        # Calculate the height of each segment in pixels
        cell_height_pixels = self.image.height // self.rows
        # Iterate over the rows and columns to crop the image into segments
        for i in range(self.rows):
            for j in range(self.cols):
                left = j * cell_width_pixels
                upper = i * cell_height_pixels
                right = (j + 1) * cell_width_pixels
                lower = (i + 1) * cell_height_pixels
                cropped_image = self.image.crop((left, upper, right, lower))
                # Append the cropped image to the segments list
                self.segments.append(cropped_image)

    def save_segments(self):
        for i, segment in enumerate(self.segments):
            segment.save(f'cropped_image_{i}.png')

    def calculate_segment_areas(self):
        cell_width_pixels = self.image.width // self.cols
        cell_height_pixels = self.image.height // self.rows
        cell_area_um2 = (cell_width_pixels / self.resolution_x) * (cell_height_pixels / self.resolution_y)
        return cell_area_um2


import csv

class ParameterReader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.parameters = {}
    
    def read_parameters(self):
        with open(self.filepath, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            self.parameters = next(reader)  # Assumes only one row of parameters
        return self.parameters

import os
import csv
from PIL import Image, ImageOps
from image_processor_backend import ImageSplitter, ImageProcessor

class ImageProcessorManager:
    def __init__(self, params):
        self.input_dir = params['input_dir']
        self.output_dir = params['output_dir']
        self.rows = int(params['rows'])
        self.cols = int(params['cols'])
        self.image_width_um = float(params['image_width_um'])
        self.image_height_um = float(params['image_height_um'])
        self.black_area_percentages = []

    def process_images(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for filename in os.listdir(self.input_dir):
            if filename.lower().endswith(('.bmp', '.jpg', '.png')):
                splitter = ImageSplitter(os.path.join(self.input_dir, filename),
                                         self.rows, self.cols,
                                         self.image_width_um, self.image_height_um)
                splitter.split_image()

                for i, segment in enumerate(splitter.segments):
                    processor = ImageProcessor(segment, dilate=False)
                    processor.process_image()
                    #self.black_area_percentages.append(processor.black_area_percentage)
                    self.black_area_percentages.append(((filename, i+1), processor.black_area_percentage))


                    output_filename = f'processed_segment_{i+1}_{filename}'
                    processor.image_dilated.save(os.path.join(self.output_dir, output_filename))
    # Save the black area percentages to a CSV file
    def save_percentages_to_csv(self):
        csv_file_path = os.path.join(self.output_dir, 'black_area_percentages.csv')
        with open(csv_file_path, 'w', newline='') as csvfile:
            fieldnames = ['segment_name', 'percentage']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for segment_info, percentage in self.black_area_percentages:
                # Assuming segment_info is a tuple or similar structure containing the original input file name and the segment name
                input_file_name, segment_name = segment_info
                writer.writerow({'segment_name': f"{input_file_name}_{segment_name}", 'percentage': percentage})




