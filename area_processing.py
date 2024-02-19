from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

# Load the image
file_path = 'picture.bmp'  # Replace with your image file path
image = Image.open(file_path)

# Convert the image to grayscale
image_gray = image.convert('L')

# Define the threshold value
threshold = 200  # This value might need adjustment for your specific image

# Apply threshold transformation
# Pixels with a value below the threshold turn to black (0), above the threshold to white (255)
image_thresholded = image_gray.point(lambda x: 0 if x < threshold else 255, '1')

# Convert the thresholded image to a numpy array for further analysis if needed
image_array = np.array(image_thresholded)

# Perform image dilation to make the edges thicker
# Create a structure element for dilation, you can adjust its size to increase or decrease dilation
dilatation_element_size = 4 # This value might need adjustment for your specific image
structure_element = np.ones((dilatation_element_size, dilatation_element_size))  # This is a 3x3 matrix of ones

# Apply dilation
image_dilated = binary_dilation(image_array == 0, structure_element).astype(np.uint8) * 255
image_dilated = np.invert(image_dilated)

# Count the number of black pixels in the dilated image
black_pixels = np.sum(image_dilated == 0)

# Calculate the total number of pixels in the image
total_pixels = image_dilated.size

# Calculate the fraction of the area that is black pixels
black_area_fraction = black_pixels / total_pixels

# Convert the fraction to a percentage
black_area_percentage = black_area_fraction * 100

print(black_area_percentage)

# Display the original, thresholded, and dilated images

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(image_thresholded, cmap='gray')
plt.title('Thresholded Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(image_dilated, cmap='gray')
plt.title('Dilated Image')
plt.axis('off')

plt.tight_layout()
plt.show()
