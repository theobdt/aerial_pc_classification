#! /bin/bash

mkdir -p data
#download data from google drive
function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}
echo "Fetching data.."
gdrive_download 15dUAvbDdq0tMUgIE9jEc3SV9cQGVYO05 data/vaihingen.zip && \
    echo "Data successfully downloaded to data/vaihingen.zip"
