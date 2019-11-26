This readme contains the steps to run the various
algorithms used in our project. It assumes that
the folder movieratepredictions (not include with
submission due to space limitations) is at the
top level of the directory.

###Collaborative Filtering with the Surprise library
files required:
surprise.ipynb
train_ratings.csv
val_ratings.csv
test_ratings.csv

libraries required:
numpy
pandas
surprise
csv

installation:
```
conda install -c conda-forge scikit-surprise
```
for more information, please visit:
http://surpriselib.com/

running:
run each jupyter notebook cell from top to bottom.

output:
submission.csv - contains the prediction ratings in the kaggle_sample_submission format

###Collaborative Filtering with the Keras Library
files required:
Keras2.ipynb
train_ratings.csv
val_ratings.csv
test_ratings.csv

libraries required:
numpy
pandas
keras

installation:
```
conda install keras
```
note: for this project we used a tensorflow backend,
keras can run under theano as well, but we have not tested it.
for tensorflow and cudnn installation:
https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html
https://www.tensorflow.org/install

for more information, please visit:
https://keras.io/#installation
https://www.linkedin.com/pulse/set-up-gpu-accelerated-tensorflow-keras-windows-10-anaconda-bhatia

running:
run each jupyter notebook cell from top to bottom.
WARNING: WILL TAKE A VERY LONG TIME

output:
submission.csv - contains the prediction ratings in the kaggle_sample_submission format

###Collaborative Filtering with the FastAi Library
files required:
fastai.ipynb
train_ratings.csv
val_ratings.csv
test_ratings.csv

libraries required:
pandas
fastai

installation:
```
conda install -c fastai fastai
```

installation of pytorch is also necessary for fastai
if using conda install, this dependency should already be taken care of
for more information, please see:
https://docs.fast.ai/install.html

running:
run each jupyter notebook cell from top to bottom.

output:
submission.csv - contains the prediction ratings in the kaggle_sample_submission format

### Matrix Factorization with MCMC
We use the libFm library to perform matrix factorization
with MCMC.

LibFm does not take data in CSV format, so data must
first be converted. Fortunately, the library provides
a tool for converting CSV into its format. It does, however,
require that the CSVs have no headers. Running the file
```python3 remix_dataset.py``` will create a few reshuffled training and validation
CSVs without headers, in addition to a CSV with training and validation
combined. We will use this combined CSV, ```combined_ratings.csv``` in
this readme.

LibFm, strangely, requires that our test CSV be labelled with dummy ratings,
so we must also add them into a new column. ```python3 convert_test_to_triple_format.py movieratepredictions/test_ratings.csv```
will take in the test file and produce a new file, ```test_ratings_triple.csv```.

Now we must install the libFm library. In the project directory, we type:

```
curl -O  http://www.libfm.org/libfm-1.42.src.tar.gz
tar -xzvf libfm-1.42.src.tar.gz
cd libfm-1.42.src
make all
```

Move ```combined_ratings.csv``` and ```test_ratings_triple.csv``` into the
folder ```libfm-1.42.src/scripts```, then ```cd libfm-1.42.src/scripts```. In
this folder, run the command:

```
./triple_format_to_libfm.pl -in combined_ratings.csv,test_ratings_triple.csv -target 2 -separator ","
```

This produces the outputs ```combined_ratings.csv.libfm``` and ```test_ratings_triple.csv.libfm```,
which is finally in the required format.

Move those files into ```libfm-1.42.src/bin``` and run:

```
./libFM -task r -train combined_ratings.csv.libfm -test test_ratings_triple.csv.libfm -dim '1,1,35' -iter 100 -method mcmc -init_stdev 0.6 -out results.txt
```

We then move the output ```results.txt``` into the top level folder and run ```python3 make_submission.py``` for our final outcome. LibFm manual: http://www.libfm.org/libfm-1.42.manual.pdf


### Average Predictor and Per Movie Average Predictor
Running ```python3 movieaveragepredictor.py``` will produce ```movieaveragepredictor.csv```, our baseline predictions.
