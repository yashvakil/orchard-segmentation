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
from capsNet import build_arch, build_arch2
from utils import combine_images
from datahandler import load_satellite_images, gen_patches, shuffle_dataset

environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

def test(model, test_list, verbose=False):
    (x_test, y_test) = test_list

    if verbose:
        print("="*51)
        print("TESTING")
        print("="*51)

    predicted, reconstructed = model.predict(x_test, batch_size=1, verbose=verbose)

    makedirs(join(args.tests_dir, "Predicted"), exist_ok=True)
    makedirs(join(args.tests_dir, "Reconstructed"), exist_ok=True)
    makedirs(join(args.tests_dir, "X_test"), exist_ok=True)
    makedirs(join(args.tests_dir, "Y_test"), exist_ok=True)
    
    band_means = np.load(join(args.dataset, "band_means.npy"))
    band_stds = np.load(join(args.dataset, "band_stds.npy"))
    
    if verbose:
        print("="*51)
        print("SAVING IMAGES")
        print("="*51)
    
    reconstructed = np.array([((image*band_stds)+band_means).astype(np.int16) for image in reconstructed])
    x_test = np.array([((image*band_stds)+band_means).astype(np.int16) for image in x_test])

    for i in range(x_test.shape[0]):
        tif.imsave(join(args.tests_dir, "Predicted", "pred_" + str(i) + ".tif"), predicted[i])
        tif.imsave(join(args.tests_dir, "Reconstructed", "recon_" + str(i) + ".tif"), reconstructed[i])
        tif.imsave(join(args.tests_dir, "X_test", "x_test_" + str(i) + ".tif"), x_test[i])
        tif.imsave(join(args.tests_dir, "Y_test", "y_test_" + str(i) + ".tif"), y_test[i])

    np.save(join(args.tests_dir, "predicted.npy"), predicted)
    np.save(join(args.tests_dir, "reconstructed.npy"), reconstructed)
    np.save(join(args.tests_dir, "x_test.npy"), x_test)
    np.save(join(args.tests_dir, "y_test.npy"), y_test)

    if verbose:
        print("="*51)
        print("Images saved at " + str(args.tests_dir))
        print("="*51)

    return predicted, reconstructed


if __name__ == "__main__":

    x_test = np.load(join(args.patch_dataset_dir, "x_val.npy"))
    y_test = np.load(join(args.patch_dataset_dir, "y_val.npy"))

    _, eval_model = build_arch2(x_test.shape[1:], n_class=args.num_class)
    print_summary(model=eval_model, positions=[.38, .65, .75, 1.])

    eval_model.load_weights(join(args.models_dir, args.weights))
    if args.debug:
        print("="*51)
        print("model loaded from", join(args.models_dir, args.weights))
        print("="*51)

    test(eval_model, (x_test, y_test), verbose=args.debug)