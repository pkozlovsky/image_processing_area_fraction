
import image_processor_backend as ipb

# Initialize the splitter with the image path and the desired number of rows and columns
splitter = ipb.ImageSplitter('cropped_LiliumFP1_A1.bmp', rows=3, cols=3, image_width_um=4000, image_height_um=4000)
splitter.split_image()

# Calculate and print the total area of the base image in micrometers squared
print(f"Total area of the base image: {splitter.calculate_segment_areas() * 4} µm²")

# Process each segment with the ImageProcessor
for i, segment in enumerate(splitter.segments):
    # Initialize the processor for the current segment, with optional dilation
    processor = ipb.ImageProcessor(segment, dilate=False)  # Set dilate to True to apply dilation
    processor.process_image()  # Apply threshold and optionally dilate

    # Calculate and display the black area percentage for the current segment
    print(f"Segment {i+1} - Black area percentage: {processor.black_area_percentage:.2f}%")

    # Save the processed segments
    processor.image_dilated.save(f'processed_segment_{i+1}.png')

    # If you want to display the processed segments, uncomment the following line
    processor.display_images()
dis