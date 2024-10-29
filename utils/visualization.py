import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from constants import GAMMA_PARAMS

def display_images_side_by_side(images: list, titles: list=None, config_cmap: list[None, str]=None):

    num_images = len(images)
    
    if num_images < 1 or num_images > 4:
        raise ValueError("This function supports between 1 and 4 images.")
    
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    
    if num_images == 1:
        axes = [axes]
    
    for i, image in enumerate(images):
        if config_cmap[i] is not None:
            axes[i].imshow(image, cmap=config_cmap[i])
        else:
            axes[i].imshow(image)
            
        axes[i].axis('off')  
        if titles and i < len(titles):  
            axes[i].set_title(titles[i])

    plt.show()

def plot_gamma_distribution(alpha, beta):

    samples = np.random.gamma(alpha, beta, 100000)  

    random_value = np.random.choice(samples)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=100, density=True, alpha=0.6, color='grey', edgecolor='black') 

    x = np.linspace(0, np.max(samples), 1000)
    pdf = (x ** (alpha - 1) * np.exp(-x / beta)) / (beta ** alpha * np.math.gamma(alpha))

    plt.plot(x, pdf, color='blue', lw=2, label='Theoretical PDF')  

    plt.axvline(random_value, color='red', linestyle='--', linewidth=2, label=f'Randomly Selected Value: {random_value:.2f}')
    plt.scatter(random_value, 0, color='red', s=200, zorder=5, edgecolor='grey', linewidth=2)

    plt.annotate(f'{random_value:.2f}', 
                 xy=(random_value, 0), 
                 xytext=(random_value + 7 * beta, 20 * beta),  
                 arrowprops=dict(facecolor='blue', shrink=0.03),
                 fontsize=12,
                 color='red')

    plt.title(f'Samples from Gamma Distribution (α={alpha}, β={beta})')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid()
    plt.show()

def compute_hw_distributions(hmin, hmax, wmin, wmax, H, W, gamma_params=GAMMA_PARAMS, iterations=1000):

    h_values = []
    w_values = []


    for _ in range(iterations):
        rh, rw = np.random.gamma(shape=gamma_params['shape'], scale=gamma_params['scale'], size=2)

        h_perc = np.minimum(np.maximum(hmin, 0.06 + rh), hmax)
        w_perc = np.minimum(np.maximum(wmin, 0.06 + rw), wmax)
        
        h = (H * h_perc).round().astype(int)
        w = (W * w_perc).round().astype(int)
        
        h_values.append(h)
        w_values.append(w)
        
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(h_values, bins=30, color='grey', edgecolor='black')
    plt.title("Distribution of h")
    plt.xlabel("h values")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(w_values, bins=30, color='grey', edgecolor='black')
    plt.title("Distribution of w")
    plt.xlabel("w values")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

def plot_patch_mask(x,y, patch_size, mask, mask_patch, image, perc):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))

    center_x = patch_size[1] // 2
    center_y = patch_size[0] // 2

    ax1.imshow(mask, cmap='gray')
    rect = patches.Rectangle((x, y), patch_size[1], patch_size[0], linewidth=2, edgecolor='red', facecolor='none')
    ax1.add_patch(rect)
    ax1.set_title("Original Mask with Patch Location")
    ax1.axis('off')

    ax2.imshow(image)
    rect = patches.Rectangle((x, y), patch_size[1], patch_size[0], linewidth=2, edgecolor='red', facecolor='none')
    ax2.add_patch(rect)
    ax2.set_title("Original Mask with Patch Location")
    ax2.axis('off')

    ax3.imshow(mask_patch, cmap='gray')
    ax3.set_title(f"Extracted Patch percentage={round(perc*100, 2)}")
    ax3.axis('off')

    plt.show()

def plot_rectangule_img(rect_coods, image):

    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    #plt.axis('off')

    #rect_coods = (y_range, x_range) -> this sequence
    rect = patches.Rectangle(
        (rect_coods[1][0], rect_coods[0][0]), 
        rect_coods[1][1] - rect_coods[1][0],   
        rect_coods[0][1] - rect_coods[0][0],  
        linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    
    plt.gca().add_patch(rect)

    plt.title("Search Area for Patch (Red Rectangle)")
    plt.show()

def plot_patch_mask(x,y, patch_size1, mask, image, titles, patch_size2=None):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(image)
    rect = patches.Rectangle((x, y), patch_size1[1], patch_size1[0], linewidth=1, edgecolor='red', facecolor='none')
    ax1.add_patch(rect)
    ax1.set_title(titles[0])
    #ax2.axis('off')


    ax2.imshow(mask, cmap='gray')
    if patch_size2 is None:
        rect = patches.Rectangle((x, y), patch_size1[1], patch_size1[0], linewidth=1, edgecolor='red', facecolor='none')
    else:
        rect = patches.Rectangle((x, y), patch_size2[1], patch_size2[0], linewidth=1, edgecolor='red', facecolor='none')

    ax2.add_patch(rect)
    ax2.set_title(titles[1])
    #ax1.axis('off')
    
    plt.show()
