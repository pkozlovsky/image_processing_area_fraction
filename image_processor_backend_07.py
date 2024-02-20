
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import pandas as pd
import os
from PIL import Image, ImageOps
import csv


class ImageProcessor:
    def __init__(self, image, threshold=200, dilatation_element_size=4, dilate=False, area_correction_factor=1):
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

class ParameterReader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.parameters = {}
    
    def read_parameters(self):
        with open(self.filepath, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.parameters = row
                break  # Assuming only one row of parameters is needed
            
            # Debugging: print the keys and values read from the file
            print("Read parameters:", self.parameters)
            
            # Convert string numerical values to appropriate types
            try:
                self.parameters['rows'] = int(self.parameters.get('rows', '0'))
                self.parameters['cols'] = int(self.parameters.get('cols', '0'))
                self.parameters['image_width_um'] = float(self.parameters.get('image_width_um', '0.0'))
                self.parameters['image_height_um'] = float(self.parameters.get('image_height_um', '0.0'))
                self.parameters['K_copper'] = float(self.parameters.get('K_copper', '0.0'))
                self.parameters['K_dielectric'] = float(self.parameters.get('K_dielectric', '0.0'))
            except ValueError as e:
                print(f"Error converting parameter: {e}")
            
        return self.parameters


class ImageProcessorManager:
    def __init__(self, params, layer_thickness_file):
        self.parameters = params
        self.input_dir = params['input_dir']
        self.output_dir = params['output_dir']
        self.rows = int(params['rows'])
        self.cols = int(params['cols'])
        self.image_width_um = float(params['image_width_um'])
        self.image_height_um = float(params['image_height_um'])
        self.K_copper = params['K_copper']
        self.K_dielectric = params['K_dielectric']
        self.segment_area_um2 = (self.image_width_um / self.cols) * (self.image_height_um / self.rows)
        self.layer_thickness_data = ThicknessReader(layer_thickness_file).read_thickness()
        self.dataframe_data = {
            'segment_name': [],
            'segment_number': [],
            'layer_name': [],
            'percentage': [],
            'black_area_um2': [],
            'total_area_um2': [],
            'black_area_volume_um3': [],
            'total_volume_um3': []
        }
        self.df = pd.DataFrame()  # Initialize an empty DataFrame

    def process_images(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for filename in os.listdir(self.input_dir):
            if filename.lower().endswith(('.bmp', '.jpg', '.png')):
                layer_name = filename[:-4]  # Assumes layer name is part of the filename

                splitter = ImageSplitter(os.path.join(self.input_dir, filename), self.rows, self.cols, self.image_width_um, self.image_height_um)
                splitter.split_image()

                for i, segment in enumerate(splitter.segments):
                    processor = ImageProcessor(segment, dilate=False)
                    processor.process_image()

                    black_area_um2 = self.segment_area_um2 * (processor.black_area_percentage / 100)
                    layer_thickness = self.layer_thickness_data.get(layer_name, -1)  # Default to -1 if not found
                    black_area_volume_um3 = black_area_um2 * layer_thickness
                    total_volume_um3 = self.segment_area_um2 * layer_thickness

                    self.dataframe_data['segment_name'].append(f"{layer_name}_{i+1}")
                    self.dataframe_data['segment_number'].append(i+1)
                    self.dataframe_data['layer_name'].append(layer_name)
                    self.dataframe_data['percentage'].append(processor.black_area_percentage)
                    self.dataframe_data['black_area_um2'].append(black_area_um2)
                    self.dataframe_data['total_area_um2'].append(self.segment_area_um2)
                    self.dataframe_data['black_area_volume_um3'].append(black_area_volume_um3)
                    self.dataframe_data['total_volume_um3'].append(total_volume_um3)

                    output_filename = f'processed_segment_{i+1}_{filename}'
                    processor.image_dilated.save(os.path.join(self.output_dir, output_filename))

        self.df = pd.DataFrame(self.dataframe_data)
        self.aggregate_volumes_by_segment()

    def aggregate_volumes_by_segment(self):
        # Aggregate total volumes by segment_number across all layers
        total_volume_aggregation = self.df.groupby('segment_number')['total_volume_um3'].sum().reset_index()
        total_volume_aggregation.rename(columns={'total_volume_um3': 'total_segment_volume_um3'}, inplace=True)
        
        # Aggregate black area volumes by segment_number across all layers
        black_volume_aggregation = self.df.groupby('segment_number')['black_area_volume_um3'].sum().reset_index()
        black_volume_aggregation.rename(columns={'black_area_volume_um3': 'total_black_area_volume_um3'}, inplace=True)
        
        # Merge aggregated volume data back into the main DataFrame
        self.df = pd.merge(self.df, total_volume_aggregation, on='segment_number', how='left')
        self.df = pd.merge(self.df, black_volume_aggregation, on='segment_number', how='left')
        # Calculate the metric for each segment
        self.df['effective_Kxy'] = (self.df['total_black_area_volume_um3'] / self.df['total_segment_volume_um3']) * self.parameters['K_copper'] + \
                             (1 - self.df['total_black_area_volume_um3'] / self.df['total_segment_volume_um3']) * self.parameters['K_dielectric']
        print("Volume aggregation completed.")

    def save_percentages_to_dataframe(self):
        # Define the path for the output CSV file
        csv_file_path = os.path.join(self.output_dir, 'segment_details.csv')
        
        # Check if the DataFrame is not empty
        if not self.df.empty:
            # Save the DataFrame to a CSV file
            self.df.to_csv(csv_file_path, index=False)
            print("Data saved to CSV file successfully.")
        else:
            print("DataFrame is empty. No data to save.")


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



