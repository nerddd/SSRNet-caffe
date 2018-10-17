import numpy as np
import cv2
import scipy.io
import argparse
from tqdm import tqdm
from TYY_utils import get_meta


def get_args():
    parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
                                                 "and creates database for training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", "-o", type=str, default="megaage_aligned",
                        help="path to output database mat file")
    parser.add_argument("--db", type=str, default="megaage_aligned",
                        help="dataset; wiki or imdb")
    parser.add_argument("--img_size", type=int, default=64,
                        help="output image size")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    output_path = args.output
    db = args.db
    img_size = args.img_size

    out_ages = []
    out_imgs = []

    infile=open('/home/lc/SSR-Net/data/megaage_aligned_imglist.txt','r')
    DATA_PATH='/home/lc/SSR-Net/data/megaage_aligned/'
    lines=infile.readlines()
    i=1
    for line in lines:
        imgpath=DATA_PATH+line.strip().split()[0]
        age=int(line.strip().split()[1])
        out_ages.append(age)
        print imgpath
        img = cv2.imread(imgpath)
        out_imgs.append(img)
        print 'iter:',i
        i=i+1

    np.savez(output_path,image=np.array(out_imgs), age=np.array(out_ages), img_size=img_size)

if __name__ == '__main__':
    main()
