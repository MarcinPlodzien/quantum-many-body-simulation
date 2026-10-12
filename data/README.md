# Data files

The notebooks read these files offline. `tools/make_ml_datasets.py` recreates them. It needs network access, is run by hand, and is never run during a notebook build.

Both image files are reduced from the originals in the same way. Each 28×28 grey-scale image becomes 14×14 by averaging every block of 2×2 pixels and rounding to an integer 0…255 (uint8). Labels are unchanged. Images keep their original order.

Each `.npz` file contains the arrays `x_train` (M, 14, 14), `y_train` (M,), `x_test`, `y_test` and a short `info` string.

## `mnist_14x14.npz`

- **Content:** the first 20000 training images and all 10000 test images of the MNIST database of handwritten digits.
- **Source:** https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
- **sha256 of the downloaded original:** `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`
- **Licence:** Creative Commons Attribution-Share Alike 3.0 (CC BY-SA 3.0), https://creativecommons.org/licenses/by-sa/3.0/
- **Attribution:** the MNIST database is by Yann LeCun, Corinna Cortes and Christopher J. C. Burges. Yann LeCun and Corinna Cortes hold its copyright; it is a derivative work of the original NIST data sets (statement on the Keras dataset page, https://keras.io/api/datasets/mnist/).
- **Changes:** reduced to 14×14 pixels as described above; subset of the training set.
- **This derived file is distributed under the same licence, CC BY-SA 3.0.**

## `fashion_mnist_14x14.npz`

- **Content:** the first 10000 training images and the first 5000 test images of Fashion-MNIST (Zalando Research).
- **Source:** http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/ (files `train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`, `t10k-images-idx3-ubyte.gz`, `t10k-labels-idx1-ubyte.gz`)
- **sha256 of the downloaded originals:**
  - `train-images-idx3-ubyte.gz`: `3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84`
  - `train-labels-idx1-ubyte.gz`: `a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845`
  - `t10k-images-idx3-ubyte.gz`: `346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073`
  - `t10k-labels-idx1-ubyte.gz`: `67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5`
- **Classes:** 0 T-shirt/top, 1 trouser, 2 pullover, 3 dress, 4 coat, 5 sandal, 6 shirt, 7 sneaker, 8 bag, 9 ankle boot.
- **Changes:** reduced to 14×14 pixels as described above; subsets of the training and test sets.
- **Licence:** MIT. The notice of the original repository (https://github.com/zalandoresearch/fashion-mnist) follows.

```
The MIT License (MIT) Copyright © 2017 Zalando SE, https://tech.zalando.com

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## `santafe.txt`

- **Content:** the first 4000 points of data set A of the Santa Fe time-series competition, the intensity of a
  far-infrared NH₃ laser in a chaotic state: the competition file `A.dat` (1000 points) followed by its continuation
  `A.cont` (9093 points). Used by notebook 46.
- **Source:** the competition page, http://www-psych.stanford.edu/~andreas/Time-Series/SantaFe.html, retrieved from
  the Internet Archive (`http://web.archive.org/web/2001id_/http://www-psych.stanford.edu/~andreas/Time-Series/SantaFe/A.dat`
  and `.../A.cont`).
- **sha256 of the downloaded originals:**
  - `A.dat`: `14faec41629ead4f45d9c055a4494269dd9122dd559594502c0c522dc98a9eb2`
  - `A.cont`: `feebf2460cb8c945b87dde368cccde95a05ee06955827c4bec93f2043db6ec5e`
- **Changes:** the 4000 values minus their mean, divided by their sample standard deviation, one value per line with
  5 significant digits. The series in the `reservoirpy` package (`reservoirpy.datasets.santafe_laser`) equals `A.dat`
  followed by `A.cont` point by point.
- **Credit:** the data set was published with the competition proceedings, A. S. Weigend and N. A. Gershenfeld (eds.),
  *Time Series Prediction: Forecasting the Future and Understanding the Past* (Addison-Wesley, 1994). It was
  contributed by U. Hübner from data collected primarily by N. B. Abraham and C. O. Weiss. The competition page states
  no licence; the data have been distributed freely for research since 1991.
