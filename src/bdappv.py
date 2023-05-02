import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import random
import torchvision.transforms.functional as TF
import torchvision

"""
Pytorch DataLoader classes for the BDAPPV dataset. These dataloaders fit the 
format of the dataset. 

Images have a default size of 400*400 and are automatically resized.

The label in the case of classification and the mask in case of segmentation are updated accordingly.
The user should not pass cropping transforms for the segmentation, as these are automatically assigned.
The user should not pass cropping or rotation transforms for the classification, as these are automatically assigned.
If the user wants to disable cropping or rotation transforms, it can be passed as argument.
"""

# Helper functions

random.seed(42)

def rotations(image, mask):
    """
    applies a series of rotations to the image and the mask
    avoids flips that lead to a panel pointing upwards (i.e. north)
    """
    # vertical flip
    if random.random() > 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)

    # rotations (90, -90)
    if random.random() > 0.5:
        image = TF.rotate(image, 90.)
        mask = TF.rotate(mask, 90.)

    if random.random() > 0.5:
        image = TF.rotate(image, -90.)
        mask = TF.rotate(mask, -90.)   
        
    return image, mask
    
def crop_mask_and_image(image, mask, size, randomize):
    """
    crops the image and the mask
    """
    
    max_x, max_y = image.shape[1], image.shape[2] # get the size of the image
    x_span, y_span = max(0,max_x - size), max(0,max_y - size) # coordinates of the anchor box

    # randomly draw points in this box, which determine the top left corner
    if randomize:
        x, y = random.randint(0, x_span), random.randint(0, y_span)    
    else:
        x, y = int(x_span / 2), int(y_span / 2)
    
    return torchvision.transforms.functional.crop(image, y, x, size, size), torchvision.transforms.functional.crop(mask, y, x, size, size), (x,y)
 
def apply_crop(image, size, randomize = True):
    """
    crops the image by randomly picking a point
    in an anchor corner
    returns the transformed image and the x, y anchor
    """            
    max_x, max_y = image.shape[1], image.shape[2] # get the size of the image
    x_span, y_span = max(0,max_x - size), max(0,max_y - size) # coordinates of the anchor box

    # randomly draw points in this box, which determine the top left corner
    if randomize:
        x, y = random.randint(0, x_span), random.randint(0, y_span)    
    else:
        # otherwise, the center of the image is considered
        x, y = int(x_span / 2), int(y_span / 2)
    
    return torchvision.transforms.functional.crop(image, y, x, size, size), x, y

def update_label(x, y, name, size, mask_dir):
    """
    checks on the mask that the label is unchanged.
    """

    # open the mask
    mask = transforms.ToTensor()(Image.open(os.path.join(mask_dir, name)).convert("RGB"))
    mask_cropped = torchvision.transforms.functional.crop(mask, y, x, size, size)

    # check whether the panel is still on the mask
    # by counting the number of activated pixels on the cropped mask.
    return int(torch.sum(mask_cropped) > 5)    

# Classes

class BDAPPVClassification(Dataset):
    def __init__(self, img_dir, transform = None, size = 224, random = True, opt = None, specialized = False):
        """
        Args:
            img_dir (str): directory with all the images. Should have the following structure: 
            img_dir/
              |
               -- img/
              |
               -- mask/
               
            transforms (callable, optional): optional image transforms to be applied on the image.  
                                             (!) avoid crop transforms
            size (int or None) : indicates the size of the crop. Applies transforms accordingly.
            random (bool) : whether the cropping should be random or made at the center of the image.
            opt (dict). a dictionnary with optional data. used for specialized training should contain :
                - metadata (pd.Dataframe) the dataframe of installations' metadata
                - cutoff (int) the desired cutoff
            specialized (bool) : whether specialized training should be made. Default : false
            """
        self.img_dir = os.path.join(img_dir, 'img')
        self.mask_dir = os.path.join(img_dir, 'mask')        
        
        self.transform = transform
        self.size = size
        self.random = random


        # image and mask folders : filtering only the elements that end with .png.
        self.img_folder = [img for img in os.listdir(self.img_dir) if img[-4:] == '.png']
        self.mask_folder = [mask for mask in os.listdir(self.mask_dir) if mask[-4:] == '.png'] 

        if specialized:

            # get the cutoff and the metadata
            metadata = opt['metadata']
            cutoff = opt['cutoff']
            case = opt['case']

            if case == "above":
                installations = metadata[metadata['kWp'] >= cutoff].values
            elif case == 'below':
                installations = metadata[metadata['kWp'] < cutoff].values

            # update the attributes
            # filter the images by those corresponding to the installations below or above the cutoff
            self.img_folder = [img for img in self.img_folder if img[:-4] in installations]
            self.mask_folder = [img for img in self.mask_folder if img[:-4] in installations]
        

    def __len__(self):        
        return len(self.img_folder)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.img_dir, self.img_folder[idx])        
        image = transforms.ToTensor()(Image.open(img_path).convert("RGB"))
        
        label = int(self.img_folder[idx] in self.mask_folder) # check whether the image name is in the mask folder
                                                         # to assign a label.
        name = self.img_folder[idx][:-4] # Name : remove the extension
        
        self.name = self.img_folder[idx] # add the name in the attributes of the class
            
        if self.size is not None:
            # apply crop transforms. 
            # updates the label if necessary.
            image, label = self.crop(image, label)    
            
        # apply the additional transforms to the image.
        if self.transform:            
            image = self.transform(image)
            
        return image, label
    
    # the function below is a helper function, called in the __getitem__            
    def crop(self, image, label):
        """
        function that randomly crops the image
        and updates the label accordingly
        """        
        if label == 0:
            # no need to worry about a possible
            # change in the label        
            image, _, _ = apply_crop(image, self.size, self.random)
            
        elif label == 1:
            # in this case, we need to check whether the crop
            # leaves the label unchanged
            
            image, x, y = apply_crop(image, self.size)
            
            # check that the label does not change.
            label = update_label(x, y, self.name, self.size, self.mask_dir)
            
        return image, label

class BDAPPVClassification_path(Dataset):
    def __init__(self, img_dir, transform = None, size = 224, random = True, opt = None, specialized = False):
        """
        Args:
            img_dir (str): directory with all the images. Should have the following structure: 
            img_dir/
              |
               -- img/
              |
               -- mask/
               
            transforms (callable, optional): optional image transforms to be applied on the image.  
                                             (!) avoid crop transforms
            size (int or None) : indicates the size of the crop. Applies transforms accordingly.
            random (bool) : whether the cropping should be random or made at the center of the image.
            opt (dict). a dictionnary with optional data. used for specialized training should contain :
                - metadata (pd.Dataframe) the dataframe of installations' metadata
                - cutoff (int) the desired cutoff
            specialized (bool) : whether specialized training should be made. Default : false
            """
        self.img_dir = os.path.join(img_dir, 'img')
        self.mask_dir = os.path.join(img_dir, 'mask')        
        
        self.transform = transform
        self.size = size
        self.random = random


        # image and mask folders : filtering only the elements that end with .png.
        self.img_folder = [img for img in os.listdir(self.img_dir) if img[-4:] == '.png']
        self.mask_folder = [mask for mask in os.listdir(self.mask_dir) if mask[-4:] == '.png'] 

        if specialized:

            # get the cutoff and the metadata
            metadata = opt['metadata']
            cutoff = opt['cutoff']
            case = opt['case']

            if case == "above":
                installations = metadata[metadata['kWp'] >= cutoff].values
            elif case == 'below':
                installations = metadata[metadata['kWp'] < cutoff].values

            # update the attributes
            # filter the images by those corresponding to the installations below or above the cutoff
            self.img_folder = [img for img in self.img_folder if img[:-4] in installations]
            self.mask_folder = [img for img in self.mask_folder if img[:-4] in installations]
        

    def __len__(self):        
        return len(self.img_folder)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.img_dir, self.img_folder[idx])        
        image = transforms.ToTensor()(Image.open(img_path).convert("RGB"))
        
        label = int(self.img_folder[idx] in self.mask_folder) # check whether the image name is in the mask folder
                                                         # to assign a label.
        name = self.img_folder[idx][:-4] # Name : remove the extension
        
        self.name = self.img_folder[idx] # add the name in the attributes of the class
            
        if self.size is not None:
            # apply crop transforms. 
            # updates the label if necessary.
            image, label = self.crop(image, label)    
            
        # apply the additional transforms to the image.
        if self.transform:            
            image = self.transform(image)
            
        return image, label, img_path
    
    # the function below is a helper function, called in the __getitem__            
    def crop(self, image, label):
        """
        function that randomly crops the image
        and updates the label accordingly
        """        
        if label == 0:
            # no need to worry about a possible
            # change in the label        
            image, _, _ = apply_crop(image, self.size, self.random)
            
        elif label == 1:
            # in this case, we need to check whether the crop
            # leaves the label unchanged
            
            image, x, y = apply_crop(image, self.size)
            
            # check that the label does not change.
            label = update_label(x, y, self.name, self.size, self.mask_dir)
            
        return image, label

class BDAPPVSegmentation(Dataset):
    def __init__(self, img_dir, transform = None, size = 299, random = True, opt = None, specialized = False):
        """
        Args:
            img_dir (str): directory with all the images. Should have the following structure: 
            img_dir/
              |
               -- img/
              |
               -- mask/
               
            transforms (callable, optional): optional image transforms to be applied on the image.  
                                             (!) avoid crop and rotation transforms
            size (int or None) : indicates the size of the crop. Applies transforms accordingly.
            random (bool) : whether random transforms should be applied. Otherwise, no rotations and center crop only.
            """
        self.img_dir = os.path.join(img_dir, 'img')
        self.mask_dir = os.path.join(img_dir, 'mask')        
        
        self.transform = transform
        self.size = size
        self.random = random
        
        # image and mask folders : filtering only the elements that end with .png.
        self.img_folder = [img for img in os.listdir(self.img_dir) if img[-4:] == '.png']
        self.mask_folder = [mask for mask in os.listdir(self.mask_dir) if mask[-4:] == '.png'] 

        if specialized == True:

            # get the cutoff and the metadata
            metadata = opt['metadata']
            cutoff = opt['cutoff']
            case = opt['case']

            if case == "above":
                installations = metadata[metadata['kWp'] >= cutoff].values
            elif case == 'below':
                installations = metadata[metadata['kWp'] < cutoff].values

            # update the attributes
            # filter the images by those corresponding to the installations below or above the cutoff
            self.img_folder = [img for img in self.img_folder if img[:-4] in installations]
            self.mask_folder = [img for img in self.mask_folder if img[:-4] in installations]



    def __len__(self):      
        # the length is computed as the number of masks in the masks directory
        return len(self.mask_folder)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.img_dir, self.mask_folder[idx]) # look for the images in the mask folder only        
        image = transforms.ToTensor()(Image.open(img_path).convert("RGB"))
        mask = transforms.ToTensor()(Image.open(os.path.join(self.mask_dir, self.mask_folder[idx]))) # open the corresponding mask
        
        name = self.mask_folder[idx][:-4] # Name : remove the extension        
        self.name = self.mask_folder[idx] # add the name in the attributes of the class
        
        # apply the transforms

        # those passed in the constructor
        if self.transform:
            image = self.transform(image)
            
        # apply the random rotations to the mask and image
        if self.random :             
            image, mask = rotations(image, mask)
        
        # finally crop the image and the mask
        image, mask, shift = crop_mask_and_image(image, mask, self.size, self.random)
            
        return image, mask, name, shift