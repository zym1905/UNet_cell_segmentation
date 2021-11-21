import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from torchvision import transforms

from uNetData import BasicDataset
from uNetModel import UNet
#from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', default='./', metavar='OUTPUT', help='dir for output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1,
                        help='Scale factor for the input images')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))

image_size = ()
def cutImage(tiff_image, outdir):
    #tiff_image = '/Users/zhangyong/work/MLIM/data/20210426-T215-Z3-L-M019-01_regist.tif'
    #outdir = '/Users/zhangyong/work/MLIM/data/slices'

    image = Image.open(tiff_image)
    width, height = image.size
    global image_size
    image_size = image.size

    files = []
    step_size = 256
    upper = 0
    left = 0
    right = step_size
    lower = step_size
    count = 0

    while lower < height:
        bbox = (left, upper, right, lower)
        image_slice = image.crop(bbox)
        path = os.path.join(outdir, "slice_20210426-T215-Z3-L-M019-01_regist." + str(left) + "_" + str(upper) + ".png")
        files.__add__(path)
        image_slice = image_slice.convert('L')
        image_slice.save(path)

        if right > width:
            upper += step_size
            lower += step_size
            left = 0
            right = step_size
        else:
            left += step_size
            right += step_size

        count += 1

    print('split image into:' + str(count) + ' iamges.')
    return files


if __name__ == '__main__':
    args = get_args()
    in_files = args.input
    #out_files = get_output_filenames(args)
    outdir = args.output
    global image_size

    net = UNet(n_channels=1, n_classes=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        new_iamge = Image.new('L', image_size)
        step_size = 256
        count = 0
        for slices in cutImage(filename, outdir):
            logging.info(f'\nPredicting image {filename} ...')
            img = Image.open(slices)
            mask = predict_img(net=net,
                               full_img=img,
                               scale_factor=args.scale,
                               out_threshold=args.mask_threshold,
                               device=device)
            str_tmp = os.path.splitext(os.path.splitext(slices))
            str_list = str_tmp[1].split('_')
            left = int(str_list[0])
            upper = int(str_list[1])
            right = left + step_size
            lower = upper + step_size
            bbox = (left, upper, right, lower)
            new_iamge.paste(mask, bbox)

        new_iamge.save(os.path.join(str_tmp[0] + '.png'))

'''
    for i, filename in enumerate(in_files):
        logging.info(f'\nPredicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            #plot_img_and_mask(img, mask)
'''