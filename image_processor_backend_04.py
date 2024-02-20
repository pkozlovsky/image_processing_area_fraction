
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import pandas as pd

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
from PIL import Image, ImageOps
from image_processor_backend_02 import ImageSplitter, ImageProcessor

class ImageProcessorManager:
    def __init__(self, params , layer_thickness_file):
        self.input_dir = params['input_dir']
        self.output_dir = params['output_dir']
        self.rows = int(params['rows'])
        self.cols = int(params['cols'])
        self.image_width_um = float(params['image_width_um'])
        self.image_height_um = float(params['image_height_um'])
        self.black_area_percentages = []
        # Add new attributes for resolution in um/pixel if needed
        self.segment_area_um2 = (params['image_width_um'] / params['cols']) * (params['image_height_um'] / params['rows'])
        # Calculate the total area of each segment in um^2
        self.segment_area_um2 = (self.image_width_um / self.cols) * (self.image_height_um / self.rows)
        # Initialize thickness data
        self.layer_thickness_data = ThicknessReader(layer_thickness_file).read_thickness()



    def process_images(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for filename in os.listdir(self.input_dir):
            if filename.lower().endswith(('.bmp', '.jpg', '.png')):
                # Directly remove the last 4 characters (.ext) from the filename
                layer_name = filename[:-4]  # Removes the last 4 characters, including the dot
                print(f"Directly extracted layer name without extension: '{layer_name}'")

                # Create an ImageSplitter instance
                splitter = ImageSplitter(os.path.join(self.input_dir, filename),
                                 self.rows, self.cols,
                                 self.image_width_um, self.image_height_um)
                splitter.split_image()

                for i, segment in enumerate(splitter.segments):
                    processor = ImageProcessor(segment, dilate=False)
                    processor.process_image()
                    # Calculate black area in um^2
                    black_area_um2 = self.segment_area_um2 * (processor.black_area_percentage / 100)

                    # Calculate black area volume in um^3
                    # Assume layer_name is extracted from filename, you may need to adjust this
                    layer_name = filename.split('_')[0]  # Adjust based on your naming convention
                    
                    # Now use layer_name without the extension
                    layer_thickness = self.layer_thickness_data.get(layer_name, -1)  # Default to -1 if not found 
                    print(f"Layer thickness for '{layer_name}': {layer_thickness}")
                    black_area_volume_um3 = black_area_um2 * layer_thickness
                    print(f"Extracted layer name: '{layer_name}', Layer thickness: {layer_thickness}")
                    # calculating black_area_volume_um3
                    total_volume_um3 = self.segment_area_um2 * layer_thickness
                    print(f"Total volume of segment '{i+1}' in {filename}: {total_volume_um3} um^3")

                    # Append details including the total area of the segment
                    #self.black_area_percentages.append(((filename, i+1), processor.black_area_percentage, black_area_um2, self.segment_area_um2))
                    self.black_area_percentages.append(((filename, i+1), processor.black_area_percentage, black_area_um2, self.segment_area_um2, black_area_volume_um3, total_volume_um3))


                    output_filename = f'processed_segment_{i+1}_{filename}'
                    processor.image_dilated.save(os.path.join(self.output_dir, output_filename))

    def save_percentages_to_csv(self):
        csv_file_path = os.path.join(self.output_dir, 'asegment_details.csv')  # Renamed for clarity
        with open(csv_file_path, 'w', newline='') as csvfile:
            fieldnames = ['segment_name', 'segment_number', 'percentage', 'black_area_um2', 'total_area_um2', 'black_area_volume_um3', 'total_volume_um3']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for segment_info, percentage, black_area_um2, total_area_um2, black_area_volume_um3, total_volume_um3 in self.black_area_percentages:
                input_file_name, segment_number = segment_info
                writer.writerow({
                    'segment_name': f"{input_file_name}_{segment_number}",
                    'segment_number': segment_number,
                    'percentage': percentage,
                    'black_area_um2': black_area_um2,
                    'total_area_um2': total_area_um2,
                    'black_area_volume_um3': black_area_volume_um3,
                    'total_volume_um3': total_volume_um3
                })

    def save_percentages_to_dataframe(self):
        # Define the path for the output CSV file
        csv_file_path = os.path.join(self.output_dir, 'asegment_details.csv')
        
        # Prepare the data to be stored in the DataFrame
        data = {
            'segment_name': [],
            'segment_number': [],
            'percentage': [],
            'black_area_um2': [],
            'total_area_um2': [],
            'black_area_volume_um3': [],
            'total_volume_um3': []
        }
        
        # Populate the data dictionary
        for segment_info, percentage, black_area_um2, total_area_um2, black_area_volume_um3, total_volume_um3 in self.black_area_percentages:
            input_file_name, segment_number = segment_info
            data['segment_name'].append(f"{input_file_name}_{segment_number}")
            data['segment_number'].append(segment_number)
            data['percentage'].append(percentage)
            data['black_area_um2'].append(black_area_um2)
            data['total_area_um2'].append(total_area_um2)
            data['black_area_volume_um3'].append(black_area_volume_um3)
            data['total_volume_um3'].append(total_volume_um3)
        
        # Create a DataFrame
        df = pd.DataFrame(data)
        
        # Save the DataFrame to a CSV file
        df.to_csv(csv_file_path, index=False)
        print("Data saved to CSV file successfully!")

    

class ThicknessReader:
    def __init__(self, filepath):
        self.filepath = filepath

    def read_thickness(self):
        thickness_data = {}
        with open(self.filepath, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Assuming the first column is 'layer_name' and the second is 'layer_thk_um'
                thickness_data[row['layer_name']] = float(row['layer_thk_um'])
        return thickness_data



