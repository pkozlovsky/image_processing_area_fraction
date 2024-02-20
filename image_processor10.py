import image_processor_backend_10 as ipb

# Assume the parameters CSV file is named 'parameters.csv' and is located in the same directory as this script
parameters_csv_file = 'parameters.csv'

# Initialize ParameterReader and read parameters
parameter_reader = ipb.ParameterReader(parameters_csv_file)
parameters = parameter_reader.read_parameters()

# Initialize ImageProcessorManager with parameters and layer thickness file
layer_thickness_file = 'layers_thk.csv'
processor_manager = ipb.ImageProcessorManager(params=parameters, layer_thickness_file=layer_thickness_file)

# Process images and save processed segments
processor_manager.process_images()
processor_manager.save_percentages_to_dataframe()
processor_manager.print_aggregated_segment_summary()
#processor_manager.print_direct_segment_summary()



