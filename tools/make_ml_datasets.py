#!/usr/bin/env python
"""
make_ml_datasets.py -- (re)create the image data files used by notebook 56 (Chapter 13):

    data/mnist_14x14.npz            first 20000 training and all 10000 test images of MNIST
    data/fashion_mnist_14x14.npz    first 10000 training and first 5000 test images of Fashion-MNIST

Each 28x28 image is reduced to 14x14 by averaging blocks of 2x2 pixels and rounding to uint8 (0..255); labels are kept.
Sources, checksums and licences are listed in data/README.md.

This script needs network access and is run by hand, once.  The notebook build never runs it: notebooks only read
the .npz files from data/.

Usage
-----
    python tools/make_ml_datasets.py                       # download into a temporary directory, write data/*.npz
    python tools/make_ml_datasets.py --cache DIR           # keep (or reuse) the downloaded originals in DIR
"""
import argparse
import gzip
import hashlib
import pathlib
import tempfile
import urllib.request

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
FASHION_BASE = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
FASHION_FILES = {
    "train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
    "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
    "t10k-images-idx3-ubyte.gz": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
    "t10k-labels-idx1-ubyte.gz": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url, path, expected):
    """Download url to path (unless a file with the expected sha256 is already there) and check the checksum."""
    if not (path.exists() and sha256(path) == expected):
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path)
    got = sha256(path)
    if got != expected:
        raise SystemExit(f"checksum mismatch for {path}: {got} != {expected}")
    return path


def read_idx(path):
    """Read a gzipped idx file (the MNIST file format) into a uint8 array."""
    with gzip.open(path) as f:
        b = f.read()
    nd = b[3]
    dims = [int.from_bytes(b[4 + 4 * i:8 + 4 * i], "big") for i in range(nd)]
    return np.frombuffer(b, np.uint8, offset=4 + 4 * nd).reshape(dims)


def reduce_14x14(x):
    """(M, 28, 28) uint8 -> (M, 14, 14) uint8: mean over 2x2 blocks, rounded."""
    return np.rint(x.reshape(-1, 14, 2, 14, 2).astype(np.float64).mean(axis=(2, 4))).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None, help="directory for the downloaded originals")
    args = ap.parse_args()
    cache = pathlib.Path(args.cache) if args.cache else pathlib.Path(tempfile.mkdtemp(prefix="ml_datasets_"))
    cache.mkdir(parents=True, exist_ok=True)

    d = np.load(fetch(MNIST_URL, cache / "mnist.npz", MNIST_SHA256))
    info = (f"MNIST handwritten digits. Source: {MNIST_URL}. Licence: CC BY-SA 3.0. "
            "Reduced to 14x14 by 2x2 averaging; first 20000 training and all 10000 test images. See data/README.md.")
    np.savez_compressed(DATA / "mnist_14x14.npz",
                        x_train=reduce_14x14(d["x_train"][:20000]), y_train=d["y_train"][:20000].astype(np.uint8),
                        x_test=reduce_14x14(d["x_test"]), y_test=d["y_test"].astype(np.uint8), info=np.array(info))
    print(f"wrote {DATA / 'mnist_14x14.npz'}")

    f = {name: read_idx(fetch(FASHION_BASE + name, cache / ("fashion-" + name), h)) for name, h in FASHION_FILES.items()}
    info = (f"Fashion-MNIST clothing images. Source: {FASHION_BASE}. Licence: MIT. "
            "Reduced to 14x14 by 2x2 averaging; first 10000 training and first 5000 test images. "
            "Classes: 0 T-shirt/top, 1 trouser, 2 pullover, 3 dress, 4 coat, 5 sandal, 6 shirt, 7 sneaker, 8 bag, "
            "9 ankle boot. See data/README.md.")
    np.savez_compressed(DATA / "fashion_mnist_14x14.npz",
                        x_train=reduce_14x14(f["train-images-idx3-ubyte.gz"][:10000]),
                        y_train=f["train-labels-idx1-ubyte.gz"][:10000],
                        x_test=reduce_14x14(f["t10k-images-idx3-ubyte.gz"][:5000]),
                        y_test=f["t10k-labels-idx1-ubyte.gz"][:5000], info=np.array(info))
    print(f"wrote {DATA / 'fashion_mnist_14x14.npz'}")


if __name__ == "__main__":
    main()
