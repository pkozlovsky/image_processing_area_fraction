from PIL import Image

# Load the image
file_path = 'M2_E2.bmp'  # Replace with the actual path to your image
image = Image.open(file_path)

# Define the grid size
rows, cols = 2, 2

# Physical dimensions of the image in micrometers
image_width_um = 4000  # Replace with your image's width in micrometers
image_height_um = 4000  # Replace with your image's height in micrometers

# Calculate the resolution of the image (pixels per micrometer)
resolution_x = image.width / image_width_um
resolution_y = image.height / image_height_um

# Calculate the total area of the base image in micrometers squared
total_area_um2 = image_width_um * image_height_um

# Print the total area of the base image
print(f"Total area of the base image: {total_area_um2} µm²")

# Calculate the size of each grid cell in pixels
cell_width_pixels = image.width // cols
cell_height_pixels = image.height // rows

# Calculate the area of each cropped image in micrometers squared
cell_area_um2 = (cell_width_pixels / resolution_x) * (cell_height_pixels / resolution_y)

# Print the area of each cropped image
print(f"Area of each cropped image: {cell_area_um2} µm²")

# Split the image into a grid and save each grid cell
for i in range(rows):
    for j in range(cols):
        # Calculate the coordinates of the current cell
        left = j * cell_width_pixels
        upper = i * cell_height_pixels
        right = (j + 1) * cell_width_pixels
        lower = (i + 1) * cell_height_pixels

        # Crop the current cell
        cropped_image = image.crop((left, upper, right, lower))

        # Save the cropped image
        cropped_image.save(f'cropped_image_{i}_{j}.png')