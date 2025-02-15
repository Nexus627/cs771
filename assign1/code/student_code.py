from ctypes import sizeof
import math
import random
import glob
import os
import numpy as np

import cv2
import numbers
import collections

import torch
from torch.utils import data

from utils import resize_image, load_image

# default list of interpolations
_DEFAULT_INTERPOLATIONS = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC]

#################################################################################
# These are helper functions or functions for demonstration
# You won't need to modify them
#################################################################################

class Compose(object):
  """Composes several transforms together.

  Args:
      transforms (list of ``Transform`` objects): list of transforms to compose.

  Example:
      >>> Compose([
      >>>     Scale(320),
      >>>     RandomSizedCrop(224),
      >>> ])
  """
  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, img):
    for t in self.transforms:
      img = t(img)
    return img

  def __repr__(self):
    repr_str = ""
    for t in self.transforms:
      repr_str += t.__repr__() + '\n'
    return repr_str


class RandomHorizontalFlip(object):
  """Horizontally flip the given numpy array randomly
     (with a probability of 0.5).
  """
  def __call__(self, img):
    """
    Args:
        img (numpy array): Image to be flipped.

    Returns:
        numpy array: Randomly flipped image
    """
    if random.random() < 0.5:
      img_flipped = cv2.flip(img, 1)
      return img_flipped
    return img

  def __repr__(self):
    return "Random Horizontal Flip"

#################################################################################
# You will need to fill in the missing code in these classes
#################################################################################
class Scale(object):
  """Rescale the input numpy array to the given size.

  Args:
      size (sequence or int): Desired output size. If size is a sequence like
          (w, h), output size will be matched to this. If size is an int,
          smaller edge of the image will be matched to this number.
          i.e, if height > width, then image will be rescaled to
          (size, size * height / width)

      interpolations (list of int, optional): Desired interpolation.
      Default is ``CV2.INTER_NEAREST|CV2.INTER_LINEAR|CV2.INTER_CUBIC``
      Pass None during testing: always use CV2.INTER_LINEAR
  """
  def __init__(self, size, interpolations=_DEFAULT_INTERPOLATIONS):
    assert (isinstance(size, int)
            or (isinstance(size, collections.abc.Iterable)
                and len(size) == 2)
           )
    self.size = size
    # use bilinear if interpolation is not specified
    if interpolations is None:
      interpolations = [cv2.INTER_LINEAR]
    assert isinstance(interpolations, collections.abc.Iterable)
    self.interpolations = interpolations

  def __call__(self, img):
    """
    Args:
        img (numpy array): Image to be scaled.

    Returns:
        numpy array: Rescaled image
    """
    # sample interpolation method
    interpolation = random.sample(self.interpolations, 1)[0]

    # scale the image
    if isinstance(self.size, int):
      # size is an int
      height, width, _ = img.shape
      if height > width:
        # width is the smaller edge
        new_width = self.size
        new_height = self.size * height / width
      else:
        # height is the smaller edge
        new_height = self.size
        new_width = self.size * width / height

      new_size = (int(new_width), int(new_height))
      img_resized = resize_image(img, new_size, interpolation)
    else:
      img_resized = resize_image(img, self.size, interpolation)
    
    return img_resized

  def __repr__(self):
    if isinstance(self.size, int):
      target_size = (self.size, self.size)
    else:
      target_size = self.size
    return "Scale [Exact Size ({:d}, {:d})]".format(target_size[0], target_size[1])


class RandomSizedCrop(object):
  """Crop the given numpy array to random area and aspect ratio.

  A crop of random area of the original size with a random aspect ratio is made.
  This crop is finally resized to a fixed given size. This is widely used
  as data augmentation for training image classification models.

  Args:
      size (sequence or int): size of target image. If size is a sequence like
          (w, h), output size will be matched to this. If size is an int,
          output size will be (size, size).
      interpolations (list of int, optional): Desired interpolation.
      Default is ``CV2.INTER_NEAREST|CV2.INTER_LINEAR|CV2.INTER_CUBIC``
      area_range (list of int): range of the areas to sample from
      ratio_range (list of int): range of aspect ratio to sample from
      num_trials (int): number of sampling trials
  """

  def __init__(self, size, interpolations=_DEFAULT_INTERPOLATIONS,
               area_range=(0.25, 1.0), ratio_range=(0.8, 1.2), num_trials=10):
    self.size = size
    if interpolations is None:
      interpolations = [cv2.INTER_LINEAR]
    assert isinstance(interpolations, collections.abc.Iterable)
    self.interpolations = interpolations
    self.num_trials = int(num_trials)
    self.area_range = area_range
    self.ratio_range = ratio_range

  def __call__(self, img):
    # sample interpolation method
    interpolation = random.sample(self.interpolations, 1)[0]

    for attempt in range(self.num_trials):

      # sample target area / aspect ratio from area range and ratio range
      area = img.shape[0] * img.shape[1]
      target_area = random.uniform(self.area_range[0], self.area_range[1]) * area
      aspect_ratio = random.uniform(self.ratio_range[0], self.ratio_range[1])

      # Two cases based on the aspect ratio
      if aspect_ratio <= 1:
        # width <= height
        w = int(round(math.sqrt(target_area * aspect_ratio)))  
        h = int(round(math.sqrt(target_area / aspect_ratio)))
      else:
        # width > height
        w = int(round(math.sqrt(target_area / aspect_ratio)))  
        h = int(round(math.sqrt(target_area * aspect_ratio)))  

      if 0 < h <= img.shape[0] and 0 < w <= img.shape[1]:
        # cropping
        top = random.randint(0, img.shape[0] - h)
        bottom = top + h
        left = random.randint(0, img.shape[1] - w)
        right = left + w
        img_crop = img[top : bottom, left : right, :]
        
        # resizing
        if isinstance(self.size, int):
          # output matched to (size, size)
          im_scale = Scale((self.size, self.size), interpolations=[self.interpolations[interpolation]])
        else:
          # output matched to size
          im_scale = Scale(self.size, interpolations=[self.interpolations[interpolation]]) 
        
        img_crop_resized = im_scale(img_crop)
        return img_crop_resized

    # Fall back
    if isinstance(self.size, int):
      im_scale = Scale(self.size, interpolations=self.interpolations)
      img_resized = im_scale(img)
      # after all trials fail the default is to crop the patch in the center with a square sized output 
      height, width, _ = img_resized.shape

      top = (height - self.size)/2
      bottom = (height + self.size)/2
      left = (width - self.size)/2
      right = (width + self.size)/2

      img_cropped = img_resized[top : bottom, left : right, :]
      return img_cropped
    else:
      # with a pre-specified output size, the default crop is the image itself
      im_scale = Scale(self.size, interpolations=self.interpolations)
      img_resized = im_scale(img)
      return img_resized

  def __repr__(self):
    if isinstance(self.size, int):
      target_size = (self.size, self.size)
    else:
      target_size = self.size
    return "Random Crop" + \
           "[Size ({:d}, {:d}); Area {:.2f} - {:.2f}; Ratio {:.2f} - {:.2f}]".format(
            target_size[0], target_size[1],
            self.area_range[0], self.area_range[1],
            self.ratio_range[0], self.ratio_range[1])


class RandomColor(object):
  """Perturb color channels of a given image
  Sample alpha in the range of (-r, r) and multiply 1 + alpha to a color channel.
  The sampling is done independently for each channel.

  Args:
      color_range (float): range of color jitter ratio (-r ~ +r) max r = 1.0
  """
  def __init__(self, color_range):
    self.color_range = color_range

  def __call__(self, img):
    r_alpha = random.uniform(-self.color_range, self.color_range)
    g_alpha = random.uniform(-self.color_range, self.color_range)
    b_alpha = random.uniform(-self.color_range, self.color_range)
    img_colored = img
    img_colored[:, :, 0] = img[:, :, 0] * (1 + r_alpha)
    img_colored[:, :, 1] = img[:, :, 1] * (1 + g_alpha)
    img_colored[:, :, 2] = img[:, :, 2] * (1 + b_alpha)
    img_colored = np.uint8(img_colored)
    return img_colored

  def __repr__(self):
    return "Random Color [Range {:.2f} - {:.2f}]".format(
            1-self.color_range, 1+self.color_range)


class RandomRotate(object):
  """Rotate the given numpy array (around the image center) by a random degree.

  Args:
      degree_range (float): range of degree (-d ~ +d)
  """
  def __init__(self, degree_range, interpolations=_DEFAULT_INTERPOLATIONS):
    self.degree_range = degree_range
    if interpolations is None:
      interpolations = [cv2.INTER_LINEAR]
    assert isinstance(interpolations, collections.abc.Iterable)
    self.interpolations = interpolations

  def __call__(self, img):
    # sample interpolation method
    interpolation = random.sample(self.interpolations, 1)[0]
    # sample rotation
    degree = random.uniform(-self.degree_range, self.degree_range)
    # ignore small rotations
    if np.abs(degree) <= 1.0:
      return img

    height, width, _ = img.shape
    center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, degree, 1.0)
    image_rotated = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=self.interpolations[interpolation])

    W = width #X
    H = height #Y

    angle_a = math.radians(abs(degree))
    angle_b = math.radians(90 - degree)

    angle_a_sin = math.sin(angle_a)
    angle_b_sin = math.sin(angle_b)

    E = (math.sin(angle_a))/(math.sin(angle_b))* (H-W*(math.sin(angle_a)/math.sin(angle_b)))
    E = E/ 1 - (math.sin(angle_a)**2/math.sin(angle_b)**2)
    B = W-E
    A = (math.sin(angle_a)/math.sin(angle_b))*B

    image_rotated = image_rotated[int(round(E)) : int(round(A)), int(round(W-E)) : int(round(H-A)), :]
    image_rotated_resized = resize_image(img, (width,height), interpolation)    

    return image_rotated_resized


  def __repr__(self):
    return "Random Rotation [Range {:.2f} - {:.2f} Degree]".format(
            -self.degree_range, self.degree_range)


#################################################################################
# Additional helper functions. No need to modify.
#################################################################################
class ToTensor(object):
  """Convert a ``numpy.ndarray`` image to tensor.
  Converts a numpy.ndarray (H x W x C) image in the range
  [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
  """
  def __call__(self, img):
    assert isinstance(img, np.ndarray)
    # convert image to tensor
    assert (img.ndim > 1) and (img.ndim <= 3)
    if img.ndim == 2:
      img = img[:, :, None]
      tensor_img = torch.from_numpy(np.ascontiguousarray(
        img.transpose((2, 0, 1))))
    if img.ndim == 3:
      tensor_img = torch.from_numpy(np.ascontiguousarray(
        img.transpose((2, 0, 1))))
    # backward compatibility
    if isinstance(tensor_img, torch.ByteTensor):
      return tensor_img.float().div(255.0)
    else:
      return tensor_img


class SimpleDataset(data.Dataset):
  """
  A simple dataset using PyTorch dataloader
  """
  def __init__(self, root_folder, file_ext, transforms=None):
    # root folder, split
    self.root_folder = root_folder
    self.transforms = transforms
    self.file_ext = file_ext

    # load all labels
    file_list = glob.glob(os.path.join(root_folder, '*.{:s}'.format(file_ext)))
    self.file_list = file_list

  def __len__(self):
    return len(self.file_list)

  def __getitem__(self, index):
    # load img and label (from file name)
    filename = self.file_list[index]
    img = load_image(filename)
    label = os.path.basename(filename)
    label = label.rstrip('.{:s}'.format(self.file_ext))
    # apply data augmentation
    if self.transforms is not None:
      img  = self.transforms(img)
    return img, label
