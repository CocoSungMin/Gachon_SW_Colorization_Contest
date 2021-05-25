# Colorization with Deep learning competitions in Multimedia&lab lecture, Department of Software Gachon Uni.
-----------------------
### This Project code ranked 1st place in Colorization competitions in Multimedia&lab lecture ( From CVIP Lab in Gachon Uni. ).    
![Screen Shot 2021-05-25 at 10 26 49 AM](https://user-images.githubusercontent.com/57583574/119442232-5f679280-bd62-11eb-8893-37da1ecc4d70.png)
Main architecture based on U-net. we replace the [ Convolution -> Batch Normalization -> ReLu ] to Residual Block same architecture in resnet.    

Referenced Paper
1. U-Net : https://arxiv.org/abs/1505.04597.   
2. Attention U-Net : https://arxiv.org/abs/1804.03999      

Our model input has [ l . ab_hint , mask ] as a input.    
So 4 channels go in, out output comes with 3 channels so that it can be converted to ".png" format.    
Also we augment the train data to expand training dataset.

------------------------
## Team Member
1. Sungmin Lee ( Department of Software , undergraduated Research Assistant in VMRLAB Gachon Uni.)
2. Taeho Oh ( Department of Software , Senior student in Gachon Uni.)
3. Hyeju Yoon ( Department of Software , Senior student in Gachon Uni.)
4. Soeun Lee ( Department of Software , Senior student in Gachon Uni.)
-----------------------
## Model Architecture

### Basic U-net Architecture
![Picture1](https://user-images.githubusercontent.com/57583574/119441049-68effb00-bd60-11eb-98c1-4877df56fb21.png)

### Our Model Architecture
![Model_arch](https://user-images.githubusercontent.com/57583574/119439859-2f1df500-bd5e-11eb-9316-8fcf48f01c48.png)

### Residual Block Architecture
![Recurrent_Residual_Block](https://user-images.githubusercontent.com/57583574/119463055-55528d80-bd7c-11eb-8063-09ef2857df7d.png)

### Attention Block Architecture
![Atten_block](https://user-images.githubusercontent.com/57583574/119462193-69e25600-bd7b-11eb-842d-87e7c3c6cc52.png)

### Upsample Block Architecture
![Upsample Block](https://user-images.githubusercontent.com/57583574/119459857-0a834680-bd79-11eb-8f75-a0a5b149c9da.png)

### Model Results
![Model output](https://user-images.githubusercontent.com/57583574/119458995-42d65500-bd78-11eb-9b78-4dfddbc22363.png)



### Specific Information about model
Learning Rate : 25e-5.   
Epoch : 150.    
Loss Function : L1 loss.   
Optimizer : Adam.    
Input data size : ( 128 X 128 ).    


-----------------------
## Required Library
1. numpy
2. pillow 8.2.0
3. pytorch 1.8.1
4. torchvision 0.9.1
5. python-cv2
6. tqdm
7. cuda 10.2

-----------------
## Code Usage
1. Before training, please use "aug.py" to automatically augmentation training data ( flip , rotate 180 )
2. For training 
```
python main.py
```

3. For testing ( predicting )
```
python test.py
```
-----------------
You can get best model files via ( epoch 122 weight files )
```
https://drive.google.com/drive/folders/1GYLHydb1wabJKbSkf7c0NUSDalQ3JX9x?usp=sharing
```
training image ( contest provided) 
```
https://drive.google.com/file/d/1nCaLNE644mz7JEYeyAbvyO0njpGdD4s1/view?usp=sharing
```

test image ( contest provided )
```
https://drive.google.com/drive/folders/1GYLHydb1wabJKbSkf7c0NUSDalQ3JX9x?usp=sharing
```
--------------
## License
This project belongs to Multimedia & Lab Team 6 members , Gachon Uni ( 2021 spring semester )
