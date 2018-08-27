# FaceClassify 
A multi-category face classify application using Deep Convolution Neural Network
* This project created an application classifying face based on Gender, Wearing Eyeglasses or not, Wearing Hat or not, Young or not. 
* This application was trained and validated by using [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) datasets 

## How to use:
### Environment: Run this command to install necessary packages. You also need OpenCV (recommend 3.1.0) for this project. You also may need to add some minor fixes for cascade path (if OpenCV's Face Detection is used).
```
# Install neccessary packages
pip install -r requirements.txt
```

### Run scripts to download CelebA datasets and preprocess data:
```
./scripts/get_celebA_dataset.sh
```

### Training:
```
python train_classify.py --gpu 0 #for using GPU
# or python train_classify.py for only using CPU
```

### Test with single image:
```
python classify.py --image images/ManHat.jpg
```

### Test with web-camera:
```
python capture_face_with_dlib.py
```
