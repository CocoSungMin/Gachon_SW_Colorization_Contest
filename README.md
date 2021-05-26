# Colorization with Deep learning competitions in Multimedia&lab lecture, Department of Software Gachon Uni.
-----------------------
### This Project code ranked 1st place in Colorization Challenge competitions in Multimedia&lab lecture ( From CVIP Lab in Gachon Uni. ).    
![Screen Shot 2021-05-25 at 10 26 49 AM](https://user-images.githubusercontent.com/57583574/119442232-5f679280-bd62-11eb-8893-37da1ecc4d70.png)
  

![git](https://user-images.githubusercontent.com/57583574/119586484-c7bc7f80-be07-11eb-960b-2a3ffb4f4c10.gif)

Referenced Paper
1. U-Net Convolutional Networks for Biomedical Image Segmentation : https://arxiv.org/abs/1505.04597.   
2. Attention U-Net Learning Where to Look for the Pancreas : https://arxiv.org/abs/1804.03999      
3. Real-Time User-Guided Image Colorization with Learned Deep Priors : https://arxiv.org/abs/1705.02999.  
4. Depp Residual Learning for Image Recognition : https://arxiv.org/abs/1512.03385.  


Basic model based on "Hyeju Yoon" standard u-net : 
```
https://drive.google.com/file/d/1P0Vbt_V5FdcjWVyFcgQiK496nRUOWHzI/view?usp=sharing. 
``` 
"Sungmin Lee" replace the [ Conv - BatchNorm - ReLU ] blocks to Residual block and add attention layer in skip connection.   
 
   
Our model input has [ l . ab_hint , mask ] as a input.   
Most of Fully open github project for colorization via deep learning adapted 4 channels as a input.   
Their 4 channels consisted with 3 color channel ( e.g. Lab format L and ab chanel ) and mask as a channel.   
The mask channel leads models to train provided hint with location of the hint pixel color.   
This work shows better performance than other works in our team.
So 4 channels go in, output comes with 3 channels so that it can be converted to ".png" format.    
Also we augment the training data ( Flip horizontal , Rotate 180 ) to expand training datasets.    


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
![Screen Shot 2021-05-26 at 11 12 05 AM](https://user-images.githubusercontent.com/57583574/119592333-3a7f2800-be13-11eb-9d99-bd4fe0bee3e2.png)


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
