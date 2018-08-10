#!/usr/bin/env bash

# Create datasets folder
DATASETS_FOLDER=CelebA
if [ -d "$DATASETS_FOLDER" ]; then
    echo "Error[mkdir]: $DATASETS_FOLDER directory already exists"
else
    mkdir $DATASETS_FOLDER
fi
cd $DATASETS_FOLDER

# Download data zip file
FILE_ID=0B7EVK8r0v71pZjFTYXZWM3FlRnM
FILE_NAME=img_align_celeba.zip
if [ -e "$FILE_NAME" ]; then
    echo "Error[curl] : $FILE_NAME file already exists"
else
    curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
    CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
    curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o ${FILE_NAME}
fi

# Unzip file
UNZIP_FOLDER=img_align_celeba
if [ -d "$UNZIP_FOLDER" ]; then
    echo "Error[unzip]: $UNZIP_FOLDER directory already exists"
else
    unzip img_align_celeba.zip
fi

# Get annotation file
ANNO_ID=0B7EVK8r0v71pblRyaVFSWGxPY0U
ANNO_NAME=list_attr_celeba.txt
if [ -e "$FILE_NAME" ]; then
    echo "Error[wget] : $ANNO_NAME file already exists"
else
    wget "https://drive.google.com/uc?export=download&id=${ANNO_ID}" -O ${ANNO_NAME}
fi
