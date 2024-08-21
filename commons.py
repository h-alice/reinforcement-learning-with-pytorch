"""
Common definitions and helper functions for the project.
"""
import numpy as np
import torch
from typing import List, NamedTuple
from torchvision import transforms
from PIL import Image

def stack_images(images: List[np.ndarray], /, border_thickness=1, border_color = [255, 255, 255]) -> np.ndarray:
    """
    # Stack Images
    Stack images horizontally with border.

    Parameters
    ----------- 
    images: List of images to stack, expected to be numpy integer arrays within 0~255.   
    border_thickness: Thickness of the border, default is 1.  
    border_color: Color of the border, default is white.  
    """
    tensor_list = []

    # Cast to tensor
    for index, img in enumerate(images):
        tensor_list.append(torch.tensor(img).permute(2, 0, 1))

    # Type checking, if float tensor (0.0~1.0), convert to integer tensor (0~255)
    if any([img.dtype == torch.float32 for img in tensor_list]):
        for index, img in enumerate(tensor_list):
            # Min-max normalization
            img = (img - img.min()) / (img.max() - img.min()) * 255
            img = img.to(torch.uint8)
            tensor_list[index] = img
    


    # Making border
    padded_list = []
    for index, img in enumerate(images):
        padded_list.append(np.array(transforms.Pad(border_thickness, fill=border_color)(Image.fromarray(img))))

    # Stack
    stacked = np.concatenate(padded_list, axis=1)

    # Cast back to numpy
    return stacked
