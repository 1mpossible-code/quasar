import os
from PIL import Image

# Define input and output directories
input_dir = "original"
output_dir = "rescaled"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through all files in input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"):
        # Open image and resize to 32x32
        img = Image.open(os.path.join(input_dir, filename))
        img = img.resize((32, 32))

        # Save resized image to output directory
        output_path = os.path.join(output_dir, filename)
        img.save(output_path)
