from PIL import Image
import os

# Input folder where your .tiff images are
input_folder = r"C:\Users\sujan\OneDrive\Desktop\Unsorted"

# Loop through files in the folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.tiff', '.tif')):
        tiff_path = os.path.join(input_folder, filename)
        jpg_filename = os.path.splitext(filename)[0] + '.jpg'
        jpg_path = os.path.join(input_folder, jpg_filename)

        try:
            with Image.open(tiff_path) as img:
                # Convert to RGB (TIFFs can be CMYK, grayscale, etc.)
                rgb_img = img.convert('RGB')
                rgb_img.save(jpg_path, 'JPEG')
                print(f"✓ Converted: {filename} → {jpg_filename}")
        except Exception as e:
            print(f"✗ Failed to convert {filename}: {e}")
