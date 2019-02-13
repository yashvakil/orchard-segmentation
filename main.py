import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt
import importlib

from keras import backend as K
from keras.utils import print_summary
from os.path import join, exists
from os import makedirs, environ
from config import args
from datahandler import shuffle_dataset
from capsNet import build_arch, build_arch2, build_arch4
from utils import combine_images
from datahandler import load_satellite_images, gen_patches, shuffle_dataset
from osgeo import gdal,osr


def test(model, test_list, verbose=False):
    (x_test, y_test) = test_list

    if verbose:
        print("="*51)
        print("TESTING")
        print("="*51)

    predicted, reconstructed = model.predict(x_test, batch_size=1, verbose=1)

    makedirs(join(args.final_img_dir, "Predicted"), exist_ok=True)
    makedirs(join(args.final_img_dir, "Y_test"), exist_ok=True)

    if verbose:
        print("="*51)
        print("SAVING IMAGES")
        print("="*51)

    for i in range(x_test.shape[0]):
        tif.imsave(join(args.final_img_dir, "Predicted", "pred_" + str(i) + ".tif"), predicted[i])
        tif.imsave(join(args.final_img_dir, "Y_test", "y_test_" + str(i) + ".tif"), y_test[i])

    np.save(join(args.final_img_dir, "predicted.npy"), predicted)
    np.save(join(args.final_img_dir, "y_test.npy"), y_test)

    if verbose:
        print("="*51)
        print("Images saved at " + str(args.final_img_dir))
        print("="*51)

    return predicted, reconstructed


if __name__ == "__main__":
    
    if not (exists(join(args.patch_dataset_dir, "satellite_images.npy"))
            and exists(join(args.patch_dataset_dir, "label_images.npy"))):
        if not (exists(join(args.dataset, "satellite_image_whole.npy"))
                and exists(join(args.dataset, "label_image_whole.npy"))):
            satellite_image_whole, label_image_whole = load_satellite_images(args.dataset, save=True,
                                                                            verbose=args.debug)
        else:
            satellite_image_whole = np.load(join(args.dataset, "satellite_image_whole.npy"))
            label_image_whole = np.load(join(args.dataset, "label_image_whole.npy"))

        satellite_images, label_images = gen_patches(satellite_image_whole, label_image_whole, patch_size=256,
                                                    stride=128, aug_times=0, save=True,
                                                    save_at=args.patch_dataset_dir, verbose=args.debug)
    else:
        satellite_images = np.load(join(args.patch_dataset_dir, "satellite_images.npy"))
        label_images = np.load(join(args.patch_dataset_dir, "label_images.npy"))

        if args.debug:
            print("=" * 19 + "PATCHED IMAGES" + "=" * 19)
            print("Satellite".ljust(20) + "|" + str(satellite_images.shape))
            print("Label".ljust(20) + "|" + str(label_images.shape))
            print("=" * 52)

    x_test = satellite_images
    y_test = label_images

    _, eval_model = build_arch4(x_test.shape[1:], n_class=args.num_class)
    print_summary(model=eval_model, positions=[.38, .65, .75, 1.])

    eval_model.load_weights(join(args.models_dir, args.weights))
    if args.debug:
        print("="*51)
        print("model loaded from", join(args.models_dir, args.weights))
        print("="*51)

    predicted, reconstructed = test(eval_model, (x_test, y_test), verbose=args.debug)
    predicted = predicted[: ,64:-64, 64:-64, :]

    temp = tif.imread(join(args.dataset, "orchard.tif"))
    predicted_image = np.zeros(shape=[temp.shape[0], temp.shape[1], 1], dtype=np.float16)
    print(predicted_image.shape)

    if args.debug:
        print("="*51)
        print("COMBINING TO FORM IMAGE")
        print("="*51)

    patch_size = 128
    k=0
    for i in range(64, predicted_image.shape[0]-patch_size+1-64, patch_size):
        for j in range(64, predicted_image.shape[1]-patch_size+1-64, patch_size):
            predicted_image[i:i + patch_size, j:j + patch_size,:] = predicted[k].astype(np.float16)
            if k%1000 == 0:
                print(k)
            k = k+1

    tif.imsave(join(args.final_img_dir,"predicted.tif"), predicted_image)

    if args.debug:
        print("="*51)
        print("Image saved at", join(args.final_img_dir, "predicted.tif"))
        print("="*51)

    refrenced_image = gdal.Open(join(args.dataset, 'orchard.tif'))
    to_refrence_image = gdal.Open(join(args.final_img_dir, 'predicted.tif'))
    
    if args.debug:
        print("="*51)
        print("GEO REFRENCING IMAGE")
        print("="*51)

    driver = gdal.GetDriverByName('GTiff')
    to_refrence_image = driver.CreateCopy(join(args.final_img_dir, 'predicted_refrenced.tif'), to_refrence_image, 1)
    refrenced_image = driver.CreateCopy(join(args.final_img_dir, 'original_refrenced.tif'), refrenced_image, 1)

    proj = osr.SpatialReference(wkt=refrenced_image.GetProjection())
    epsg = proj.GetAttrValue('AUTHORITY', 1)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(int(epsg))
    dest_wkt = srs.ExportToWkt()
    gt = np.asarray(refrenced_image.GetGeoTransform()).astype(np.float32)

    to_refrence_image.SetGeoTransform(gt)
    to_refrence_image.SetProjection(dest_wkt)
    to_refrence_image.FlushCache()
    refrenced_image.FlushCache()
    to_refrence_image = None
    refrenced_image = None