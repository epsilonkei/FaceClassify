#!/usr/bin/env bash

echo "==== Download dlib model file ====="
# Create dlib folder
DLIB_FOLDER=dlib
if [ -d "$DLIB_FOLDER" ]; then
    echo "Error[mkdir]: $DLIB_FOLDER directory already exists"
else
    mkdir $DLIB_FOLDER
fi
cd $DLIB_FOLDER

# Download data br2 file & unzip
FILE_NAME=shape_predictor_68_face_landmarks.dat.bz2
UNZIP_FILE=shape_predictor_68_face_landmarks.dat
if [ -e "$UNZIP_FILE" ]; then
    echo "Error[download] : $UNZIP_FILE file already exists"
else
    if [ -e "$FILE_NAME" ]; then
        echo "Error[download] : $FILE_NAME file already exists"
    else
        curl -L "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" -o ${FILE_NAME}
        # Unzip file
        bzip2 -d ${FILE_NAME}
    fi
fi
