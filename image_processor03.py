import image_processor_backend_03 as ipb

# Assume the parameters CSV file is named 'parameters.csv' and is located in the same directory as this script
parameters_csv_file = 'parameters.csv'

# Step 2: Use ParameterReader to read parameters from the CSV
parameter_reader = ipb.ParameterReader(parameters_csv_file)
parameters = parameter_reader.read_parameters()

# Convert numerical parameters to appropriate types
parameters['rows'] = int(parameters['rows'])
parameters['cols'] = int(parameters['cols'])
parameters['image_width_um'] = float(parameters['image_width_um'])
parameters['image_height_um'] = float(parameters['image_height_um'])

# Step 3: Initialize the ImageProcessorManager with parameters from CSV
layer_thickness_file = 'layers_thk.csv'
processor_manager = ipb.ImageProcessorManager(params=parameters, layer_thickness_file=layer_thickness_file)
# DEBUGG # After loading thickness data
#print(self.layer_thickness_data)
# Process images and save processed segments
processor_manager.process_images()

# Save the black area percentages to a CSV file
processor_manager.save_percentages_to_csv()