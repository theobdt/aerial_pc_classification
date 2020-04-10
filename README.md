# Aerial point cloud classification

This project requires python >= 3.5

## Installation

```
$ git clone https://github.com/theobdt/aerial_pc_classification.git
$ cd aerial_pc_classification
$ pip3 install -r requirements.txt
```

## Getting the data
To get the data, you will first have to request access on the [ISPRS website](http://www2.isprs.org/commissions/comm3/wg4/detection-and-reconstruction.html), it is totally free.
Fill out the questionnaire to receive the credentials.
Then download data using the following script :

```
$ chmod +x fetch_data.sh
$ ./fetch_data.sh
```

## 1. Preprocessing

Convert files to PLY and center/scale the point clouds.
```
$ python3 preprocessing.py -f data/vaihingen3D_train.pts data/vaihingen3D_test.pts --centering
```
Output files will be saved to `data/preprocessed/`.

## 2. Computing features

To compute all features at once, run the following command :

```
$ python3 compute_features.py -f data/preprocessed/vaihingen3D_train.ply data/preprocessed/vaihingen3D_test.ply --full_pipeline
```

Otherwise, features can be computed incrementally as detailed below.
Output files will be saved in `data/features/`.

### 2.1 Local descriptors
```
$ python3 compute_features.py -f path/to/data.ply --compute_descriptors --descriptors all --radius_descriptors 2 --preferred_orientation +z
```

### 2.2 Region growing
Grows `n_regions` regions, trying to maximize/minimize `criterion_region`:
```
$ python3 compute_features.py -f path/to/data.ply --region_growing --n_regions 50 --radius_region 1 --criterion_region max planarity --thresh_height 0.1 --thresh_angle 0.1 --thresh_descriptor 0.1
```

### 2.3 Ground extraction

Stitches regions together to extract the ground :
```
$ python3 compute_features.py -f path/to/data.ply --ground_extraction --slope_intra 0.1
 --slope_inter 0.2 --percentile_closest 0.1
```
This will also output a point cloud of the ground only in `data/ground_only/`


### 2.4 Rasterize ground
To visualize the underlying ground used to compute `height_above_ground` as a grid :
```
$ python3 compute_features.py -f path/to/data.ply --rasterize_ground --step_rasterize 0.5 --method_rasterize delaunay
```
This will output the rasterized ground as a point cloud in `data/ground_rasterized/`


### 2.5 Height above ground

Uses the extracted ground to compute `height_above_ground` for each point of the cloud.
```
$ python3 compute_features.py -f path/to/data.ply --height_above_ground
```

## 3. Training Bi LSTM network

To train the network, run the following command :

```
$ python3 train.py 
```

## 4. Predict with Bi LSTM network
```
$ python3 test.py -f data/features/vaihingen3D_test.ply --ckpt ckpts/2020-04-10_12-25-36
```

## Notes
To (re)compute one feature only, you can use other features previously computed :
```
$ python3 compute_features.py -f data/features/vaihingen3D_train.ply --ground_extraction
```
In this example, only the `ground` field of the point cloud will be modified.
