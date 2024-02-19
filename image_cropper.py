import os
from PIL import Image, ImageOps

input_dir = 'c:\\Users\\pik1\\001_PROJECTS\\029_TMA76_Heat_transfer_nRF54H20_Lilium\\004_analyses\\area_fraction\\LiliumFP1_layer_snaps\\'  # input directory path
output_dir = 'c:\\Users\\pik1\\001_PROJECTS\\029_TMA76_Heat_transfer_nRF54H20_Lilium\\004_analyses\\area_fraction\\LiliumFP1_layer_snaps\\cropped'  #  output directory path

# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each .bmp file in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith('.bmp'):  # This makes the check case-insensitive
        file_path = os.path.join(input_dir, filename)
        image = Image.open(file_path)

        # Convert the image to grayscale and then to a binary image
        image_gray = image.convert('L')
        image_binary = image_gray.point(lambda x: 0 if x < 128 else 255, '1')

        # Invert the image (if necessary) so that the shapes are black and the background is white
        image_inverted = ImageOps.invert(image_binary)

        # Find the bounding box of the black shapes
        bbox = image_inverted.getbbox()

        # Crop the image to the bounding box
        cropped_image = image.crop(bbox)

        # Save the cropped image to the output directory
        output_file_path = os.path.join(output_dir, f'cropped_{filename}')
        cropped_image.save(output_file_path)
        print(f"Cropped image saved to {output_file_path}")

