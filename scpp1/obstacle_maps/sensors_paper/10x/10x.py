import os
from PIL import Image

# Get current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Supported image formats
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff')

for filename in os.listdir(script_dir):
    if filename.lower().endswith(image_extensions):
        image_path = os.path.join(script_dir, filename)
        try:
            with Image.open(image_path) as img:
                # Ensure image is in binary mode (1-bit pixels, black and white)
                img = img.convert('1')
                new_size = (img.width * 10, img.height * 10)
                resized_img = img.resize(new_size, Image.NEAREST)
                resized_img.save(image_path)
                print(f"Resized without blur: {filename}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
