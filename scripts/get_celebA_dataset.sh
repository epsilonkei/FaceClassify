#!/usr/bin/env bash

# Download data zip file
FILE_ID=0B7EVK8r0v71pZjFTYXZWM3FlRnM
FILE_NAME=img_align_celeba.zip
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o ${FILE_NAME}

# Unzip file
unzip img_align_celeba.zip

# Get annotation file
ANNO_ID=0B7EVK8r0v71pblRyaVFSWGxPY0U
ANNO_NAME=list_attr_celeba.txt
wget "https://drive.google.com/uc?export=download&id=${ANNO_ID}" -O ${ANNO_NAME}
