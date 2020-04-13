import cv2
import math
import copy
import torch
import kornia
import numpy as np
import torchvision
import codebase.debug as db
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont, Image

#
# Make font a global variable since reloading many times can cause segfaults.
font = ImageFont.truetype('./FreeMono.ttf', 12)
font_large = ImageFont.truetype('./FreeMonoBold.ttf', 20)
# torchvision.set_image_backend('accimage')
#
# Create a grid from the input tensor.
def tensor2grid(images_batch):
    if isinstance(images_batch, np.ndarray):
        images_batch = torch.from_numpy(images_batch.transpose((0, 3, 1, 2)))
    grid = torchvision.utils.make_grid(images_batch, nrow=math.ceil(math.sqrt(images_batch.shape[0])))
    return grid
#
# Display all images from a batch.
def showIm(grid, block=True, save=False, name='', show=True):
    plt.figure(figsize=(10, 10), facecolor='w', edgecolor='k', linewidth=0.05)
    plt.axis('off')
    plt.imshow(grid.numpy().transpose((1, 2, 0)), aspect='auto')
    # plt.tight_layout(pad=0.01)
    plt.subplots_adjust(0.01, right=0.99, top=0.99, bottom=0.01)
    # if name is not None:
    #     plt.savefig(name + '.png')
    if show:
        plt.show(block=block)
#
# PLot an image.
def showTensor(t):
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    if t.dtype != np.uint8:
        t = np.uint8(t * 255)
    plt.imshow(t)
    plt.show()
#
#
def showBatch(imgs, show=False, denorm=False, title='', nrow=5):
    if isinstance(imgs, list):
        imgs = torch.stack(imgs, 0)
    elif isinstance(imgs, tuple):
        denorm = False
        imgs_t = []
        tt = torchvision.transforms.ToTensor()
        for im in imgs:
            imgs_t.append(tt(im))
        imgs = torch.stack(imgs_t, 0)
    if imgs.numel() == 0:
        db.printWarn('No images to show')
        return
    if denorm:
        imgs = imgs.mul(0.5).add(0.5)
    imgs = imgs.view(imgs.size(0), -1, imgs.size(-2), imgs.size(-1))
    g = torchvision.utils.make_grid(imgs, nrow=nrow)
    npimg = g.detach().cpu().numpy()
    plt.figure()
    plt.title(title)
    db.printTensor(npimg)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if show:
        plt.show()
#
#
def show():
    plt.show()
#
#
def getFigs():
    return [plt.figure(n) for n in plt.get_fignums()]

def saveFigs(figs=None):
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    import pathlib
    save_dir = str(pathlib.Path().cwd()) + '/plts/'
    db.printInfo(save_dir)
    pathlib.Path(save_dir).mkdir(exist_ok=True)
    for fig in figs:
        title = fig.axes[0].get_title()
        for a in fig.axes:
            a.axis('off')
            a.set_title('')
        db.printInfo(title)
        if title == '.png':
            title = 'noName'
        save_file = save_dir + title + '.pdf'
        fig.savefig(save_file.replace(' ', '_'), bbox_inches = 'tight', pad_inches = 0)
