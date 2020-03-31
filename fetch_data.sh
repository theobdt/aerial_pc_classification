#! /bin/bash
echo "To download data, please request access at http://www2.isprs.org/commissions/comm3/wg4/detection-and-reconstruction.html (it is totally free).\n"
echo -n "Login: "; read login
echo -n "Password: "; stty -echo; read passwd; stty echo; echo

URL_TRAIN="ftp://ftp.ipi.uni-hannover.de/ISPRS_BENCHMARK_DATASETS/Vaihingen/3DLabeling/Vaihingen3D_Traininig.pts"
URL_TEST="ftp://ftp.ipi.uni-hannover.de/ISPRS_BENCHMARK_DATASETS/Vaihingen/3DLabeling/Vaihingen3D_EVAL_WITH_REF.pts.gz"
PATH_TRAIN_DL="data/vaihingen3D_train.pts"
PATH_TEST_DL="data/vaihingen3D_test.pts"

mkdir -p data
wget --user="$login" --password="$passwd" "$URL_TRAIN" -O "$PATH_TRAIN_DL" && \
    echo "Downloaded training data to $PATH_TRAIN_DL"
wget --user="$login" --password="$passwd" "$URL_TEST" -O - | gzip -cd > "$PATH_TEST_DL" && \
    echo "Downloaded test data to $PATH_TEST_DL"
