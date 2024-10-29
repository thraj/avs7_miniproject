#import sys
#sys.path.append('/home/maria/Documents/projects/avs7_miniproject/')

from typing import Optional
import numpy as np
import random

from utils.visualization import * 

class PatchSampling:

    def __init__(self, bounds: np.ndarray, background_brightness: int, brightness_threshold: int,
                 min_object_threshold: int, resize_bounds: tuple, num_patches: Optional[int] = 1,
                 plain_background: Optional[bool] = True, agco_threshold=None, index: int = 0, path: str = '',
                 verbose: Optional[bool] = None, texture: bool = False):
        self.bounds = bounds
        self.background_brightness = background_brightness
        self.brightness_threshold = brightness_threshold
        self.min_object_threshold = min_object_threshold
        self.resize_bounds = resize_bounds
        self.num_patches = num_patches
        self.plain_background = plain_background
        self.agco_threshold = agco_threshold
        self.index = index
        self.path = path
        self.verbose = verbose
        self.texture = texture

    def get_mask(self, image, background_brightness, brightness_threshold, plain_background:bool=True):

        if plain_background:
            grey = image.mean(axis=2).astype(np.uint8)
            mask = (np.abs(grey - background_brightness) > brightness_threshold).astype(np.uint8) * 255
        else:
            pass
        return grey, mask
    
    def check_contains_object(self, x, y, patch_size, min_object_ratio, mask):

        mask_patch = mask[y:y+patch_size[0], x:x+patch_size[1]]
        
        object_pixels = np.sum(mask_patch == 255)
        total_pixels = patch_size[0] * patch_size[1]
        object_ratio = object_pixels / total_pixels

        return bool(object_ratio >= min_object_ratio), object_ratio

    def check_overlap_objects(self, y, x, patch_size, dest_mask, src_mask, src_patch_centre, min_overlap=0.25):

        y_src = src_patch_centre[0] - patch_size[0]//2
        x_src = src_patch_centre[1] - patch_size[1]//2

        patch_mask = src_mask[y_src:y_src+patch_size[0], x_src:x_src+patch_size[1]]
        location_mask = dest_mask[y:y+patch_size[0], x:x+patch_size[1]]
        
        return np.logical_and(patch_mask, location_mask).sum() / np.sum(patch_mask/255) > min_overlap 

    def get_patch_centre(self, source_image, min_patch, mask, patch_size, agco_threshold=None, verbose:bool=False,
                     min_object_ratio=0.25, max_iterations=100):

        source_image_shape = source_image.shape

            # low and high limit of the Uniform distribution
        if agco_threshold is not None:
            y_range = min_patch[0], agco_threshold - min_patch[0]
            x_range = min_patch[1], source_image_shape[1] - min_patch[1]

        else:
            y_range = min_patch[0], (source_image_shape[0] - min_patch[0])
            x_range = min_patch[1], (source_image_shape[1] - min_patch[1])
        
        if verbose==1: plot_rectangule_img(rect_coods=(y_range,x_range), image=source_image)
        
        found_patch=False
        idx=0
        while (not found_patch) and (idx!=max_iterations):
        
            centre_y = random.randint(y_range[0], y_range[1]) # H
            centre_x = random.randint(x_range[0], x_range[1]) # W

            # find the region in the mask 
            y = centre_y - patch_size[0]//2
            x = centre_x - patch_size[1]//2

            # constrain
            found_patch, object_ratio = self.check_contains_object(x=x, y=y, patch_size=patch_size, min_object_ratio=min_object_ratio, mask=mask)
            
            if verbose==1: plot_patch_mask(x=x,y=y,patch_size1=patch_size, mask=mask, image=source_image, titles=["Image with Patch Location", f"Mask with Patch Location (perc={round(object_ratio*100, 2)}%)"])

            idx += 1
        return centre_y, centre_x
    
    def get_scalar(self, patch_size, min_patch_dim, max_patch_dim, resize_bounds):
        rs = np.clip(np.random.normal(loc=1, scale=0.25), resize_bounds[0], resize_bounds[1])
        min1 = np.minimum(max_patch_dim[1]/patch_size[1], max_patch_dim[0]/patch_size[0])
        min2 = np.minimum(rs, min1)
        max1 = np.maximum(min_patch_dim[1]/patch_size[1], min_patch_dim[0]/patch_size[0])

        return np.maximum(max1, min2)

    def resize_patch(self, patch_size, resize_bounds, min_patch_dim, max_patch_dim):

        s = self.get_scalar(patch_size=patch_size, min_patch_dim=min_patch_dim, max_patch_dim=max_patch_dim, resize_bounds=resize_bounds)
        h = np.clip((s*patch_size[0]).astype(int), min_patch_dim[0], max_patch_dim[0])
        w = np.clip((s*patch_size[1]).astype(int), min_patch_dim[1], max_patch_dim[1])
        
        return h, w

    def get_dest_centre(self, destination_image, src_mask, dest_mask, patch_size, src_patch_centre, agco_threshold=None, verbose:bool=False,
                        min_object_ratio=0.25, max_iterations=100):

        dest_image_shape = destination_image.shape

        if agco_threshold:
            y_range = dest_image_shape[0] - patch_size[0], agco_threshold
            x_range = patch_size[1]//2+1, dest_image_shape[1] - patch_size[1]//2+1

        else:
            y_range = patch_size[0]//2+1, dest_image_shape[0] - patch_size[0]//2+1
            x_range = patch_size[1]//2+1, dest_image_shape[1] - patch_size[1]//2+1
        
        if verbose==1: plot_rectangule_img(rect_coods=(y_range,x_range), image=destination_image)
        
        found_patch=False
        idx=0
        while (not found_patch) and (idx!=max_iterations):
        
            centre_y = random.randint(y_range[0], y_range[1]) # H
            centre_x = random.randint(x_range[0], x_range[1]) # W

            # find the region in the mask 
            y = centre_y - patch_size[0]//2
            x = centre_x - patch_size[1]//2

            # constrains
            found_patch, object_ratio = self.check_contains_object(x=x, y=y, patch_size=patch_size, min_object_ratio=min_object_ratio, mask=dest_mask)
            found_patch &= self.check_overlap_objects(y=y, x=x, patch_size=patch_size, src_patch_centre=src_patch_centre, dest_mask=dest_mask, src_mask=src_mask, min_overlap=0.25)

            if verbose==1: plot_patch_mask(x=x,y=y,patch_size1=patch_size, mask=dest_mask, image=destination_image, titles=["Image with Patch Location", f"Mask with Patch Location (perc={round(object_ratio*100, 2)}%)"])

            idx += 1

        return centre_y, centre_x
    
    def add_blending_ellipses(self, mask, num_ellipses=5):

        min_dim=5
        max_dim=15
        overlap_ratio=0.5
        coords = np.argwhere(mask > 0)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        ellipse_mask = np.zeros_like(mask, dtype=np.uint8)

        # TODO: there are parts not being totally covered by the ellipse
        for _ in range(num_ellipses):

            side = np.random.choice(['top', 'bottom', 'left', 'right'])
            if side == 'top':
                y0 = np.random.randint(y_min - max_dim * (1 - overlap_ratio), y_min + max_dim * overlap_ratio)
                x0 = np.random.randint(x_min, x_max)
            elif side == 'bottom':
                y0 = np.random.randint(y_max - max_dim * overlap_ratio, y_max + max_dim * (1 - overlap_ratio))
                x0 = np.random.randint(x_min, x_max)
            elif side == 'left':
                x0 = np.random.randint(x_min - max_dim * (1 - overlap_ratio), x_min + max_dim * overlap_ratio)
                y0 = np.random.randint(y_min, y_max)
            elif side == 'right':
                x0 = np.random.randint(x_max - max_dim * overlap_ratio, x_max + max_dim * (1 - overlap_ratio))
                y0 = np.random.randint(y_min, y_max)

            a = np.random.randint(min_dim, max_dim)
            b = np.random.randint(min_dim, max_dim)
            theta = np.random.uniform(0, np.pi)

            y, x = np.ogrid[:mask.shape[0], :mask.shape[1]]
            ellipse = (((x - x0) * np.cos(theta) + (y - y0) * np.sin(theta)) / a) ** 2 + \
                    (((x - x0) * np.sin(theta) - (y - y0) * np.cos(theta)) / b) ** 2 <= 1

            ellipse_mask |= ellipse

        final_ellipse_mask = np.clip(mask + ellipse_mask, 0, 1)  
        
        return final_ellipse_mask

    def create_patch_mask(self,center, size, mask_shape, num_ellipses=None):

        center_y, center_x = center
        patch_height, patch_width = size
        
        y_min = max(0, center_y - patch_height // 2)
        y_max = min(mask_shape[0], center_y + patch_height // 2)
        x_min = max(0, center_x - patch_width // 2)
        x_max = min(mask_shape[1], center_x + patch_width // 2)
        
        mask = np.zeros(mask_shape[:2], dtype=np.uint8)
        mask[y_min:y_max, x_min:x_max] = 255

        if num_ellipses:
            mask = self.add_blending_ellipses(mask=mask/255, num_ellipses=num_ellipses)
        
        return mask

    def run(self, dest_image, src_image):
        
        H, W, _ = np.array(dest_image.shape)
        (hmin, hmax), (wmin, wmax) = bounds

        min_patch_dim = (bounds[:, 0] * np.array([H,W])).round().astype(int)
        max_patch_dim = (bounds[:, 1] * np.array([H,W])).round().astype(int)

        if self.verbose==2:
            print(f'Size of the destination image is: H={H} x W={W}')
            print(f'Percentages of the patch for the {identifier} class: hmin={hmin}, hmax={hmax}, wmin={wmin}, wmax={wmax}')
            print(f'Patch can have size the can have is h={min_patch_dim[0]} x w={min_patch_dim[0]}')
            print(f'Maximum size the can have is h={max_patch_dim[1]} x w={max_patch_dim[1]}')

        rh, rw = np.random.gamma(shape=GAMMA_PARAMS['shape'], scale=GAMMA_PARAMS['scale'], size=2)

        h_perc = np.minimum(np.maximum(hmin, 0.06+rh), hmax)
        w_perc = np.minimum(np.maximum(wmin, 0.06+rw), wmax)

        h = (H*h_perc).round().astype(int)
        w = (W*w_perc).round().astype(int)

        if self.verbose==2:
            print(f'Sampled values from Gamma distribution: rh={rh} and rw={rw}')
            print(f'Sampled percentage sampled: h_perc={h_perc} x w_perc={w_perc}')
            print(f'Sampled size for the patch that will be taken from the source image: h={h} x w={w}')


        src_image_grey, src_mask = self.get_mask(image=src_image, background_brightness=self.background_brightness, brightness_threshold=self.brightness_threshold, plain_background=self.plain_background)
        if self.verbose==1: display_images_side_by_side(images=[src_image, src_image_grey, src_mask], titles=['Source Image - RGB', 'Source Image - Greyscale', 'Source Image Mask'], config_cmap=[None, 'grey', 'grey'])

        dest_image_grey, dest_mask = self.get_mask(image=dest_image, background_brightness=self.background_brightness, brightness_threshold=self.brightness_threshold, plain_background=self.plain_background)
        if self.verbose==1:  display_images_side_by_side(images=[dest_image, dest_image_grey, dest_mask], titles=['Destination Image - RGB', 'Destination Image - Greyscale', 'Destination Image Mask'], config_cmap=[None, 'grey', 'grey'])

        src_centre_y, src_centre_x = self.get_patch_centre(source_image=src_image, min_patch=min_patch_dim, mask=src_mask, patch_size=(h,w), agco_threshold=None, verbose=1)

        h_prime, w_prime = self.resize_patch(patch_size=(h,w), resize_bounds=resize_bounds, min_patch_dim=min_patch_dim, max_patch_dim=max_patch_dim)

        if self.verbose== 2: print(f'The new patch size after resizing is: {h_prime} x {w_prime}, before was {h} x {w}')

        y = src_centre_y - h//2
        x = src_centre_x - w//2

        if self.verbose==1: plot_patch_mask(x=x, y=y, patch_size1=(h,w), mask=src_image, patch_size2=(h_prime, w_prime), image=src_image, titles=[f"Patch Size Before Scaling: h={h} x w={w}", f"Patch Size After Scaling: h={h_prime} x w={w_prime}"])

        dest_centre_y, dest_centre_x = self.get_dest_centre(destination_image=dest_image, src_mask=src_mask, dest_mask=dest_mask, patch_size=(h_prime, w_prime), src_patch_centre=(src_centre_y, src_centre_x), agco_threshold=None, verbose=1, min_object_ratio=0.25, max_iterations=100)

        mask = self.create_patch_mask(center=(src_centre_y, src_centre_x), size=(h_prime, w_prime), mask_shape=dest_image.shape, num_ellipses=None)

        if self.verbose==1: plt.imshow(mask, cmap='gray')