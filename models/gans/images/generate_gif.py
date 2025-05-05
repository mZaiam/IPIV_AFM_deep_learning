from PIL import Image
import glob

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ld", type=int)
args = parser.parse_args()
latent_dim = args.ld

dataset = 'ausio2'

image_files = sorted(glob.glob(f"images/images_epoch*.png"), 
                    key=lambda x: int(x.split('_epoch')[1].split('.png')[0]))

images = [Image.open(image) for image in image_files]

images[0].save(
    f'gan_ld{latent_dim}_{dataset}.gif',
    save_all=True,
    append_images=images[1:],
    duration=250, 
    loop=1
)  
