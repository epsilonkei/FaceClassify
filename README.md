# FaceClassify 
A multi-category face classify application using Deep Convolution Neural Network
* This project created an application classifying face based on Gender, Wearing Eyeglasses or not, Wearing Hat or not, Young or not. 
* This application was trained and validated by using [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) datasets 

## How to use:
### Environment: 
Run this command to install necessary packages. You also need OpenCV (recommend 3.1.0) for this project. You also may need to add some minor fixes for cascade path (if OpenCV's Face Detection is used).
```
# Install neccessary packages
pip install -r requirements.txt
# Download dlib model
./scripts/download_dlib_model.sh
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
python classify.py --image images/GirlWearingHat.jpg
```
Here are some examples for my classification results
<p float="left">
  <img src="https://github.com/epsilonkei/FaceClassify/blob/master/images/GirlWearingHat_result.jpg" alt="GirlWearingHat"
  width="whatever" height=520>
  <img src="https://github.com/epsilonkei/FaceClassify/blob/master/images/IwamatsuRyo_result.jpg" alt="IwamatsuRyo"
  width="whatever" height=520>
</p>

<p float="left">
  <img src="https://github.com/epsilonkei/FaceClassify/blob/master/images/ManyGirls_result.jpg" alt="ManyGirls"
  width="whatever" height=520>
  <img src="https://github.com/epsilonkei/FaceClassify/blob/master/images/RomaPeople_result.jpg" alt="RomaPeople"
  width="whatever" height=520>
</p>

### Test with web-camera:
```
python demo_with_camera.py
```
 Example demo:
<img src="https://github.com/epsilonkei/FaceClassify/blob/master/images/FaceClassifyDemo.gif" alt="FaceClassifyDemo"
  width="whatever" height=450> 
