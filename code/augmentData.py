import os
import cv2
import numpy as np
from imgaug import augmenters as iaa

# Set the directory paths for the images and masks
image_dir = "./images/"
mask_dir = "./Masks/"
output_dir = "./AugmentedData/"

# Create a list of all the image files in the directory
image_files = os.listdir(image_dir)

# Define the augmentation pipeline
aug_pipeline = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Rotate((-45, 45)),
    iaa.GaussianBlur(sigma=(0, 3.0)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.1*255)),
    iaa.Dropout(p=(0, 0.1)),
    iaa.Crop(percent=(0, 0.2))
])

# Loop through all the image files in the directory
for image_file in image_files:
    # Load the image and corresponding mask
    image_path = os.path.join(image_dir, image_file)
    mask_path = os.path.join(mask_dir, image_file.split('.')[0] + '_mask.png')
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    mask = mask[:,:,np.newaxis]
    
    # Apply the augmentation pipeline to the image and mask
    augmented_image, augmented_mask = aug_pipeline(image=image, segmentation_maps=[mask])
    augmented_mask = augmented_mask[0].astype(np.uint8)
    
    # Save the augmented image and mask
    cv2.imwrite(os.path.join(output_dir, 'aug45' + image_file.split('.')[0] + '.jpg'), augmented_image)
    cv2.imwrite(os.path.join(output_dir, 'aug45' + image_file.split('.')[0] + '_mask.png'), augmented_mask)
