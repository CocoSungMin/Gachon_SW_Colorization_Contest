# Colorization with Deep learning contest in Multimedia&lab lecture, Department of Software Gachon Uni.
-----------------------
This Project code ranked 1st place in Colorization contest in Multimedia&lab lecture

------------------------
## Team Member
1. Sungmin Lee ( Department of Software , undergraduated Research Assistant in VMRLAB Gachon Uni.)
2. Taeho Oh ( Department of Software , Senior student in Gachon Uni.)
3. Hyeju Yoon ( Department of Software , Senior student in Gachon Uni.)
4. Soeun Lee ( Department of Software , Senior student in Gachon Uni.)
-----------------------
## Model Architecture
### Model Architecture
![Model_arch](https://user-images.githubusercontent.com/57583574/119439859-2f1df500-bd5e-11eb-9316-8fcf48f01c48.png)
### Recurrent Residual Block Architecture
![Recurrent_Residual_Block](https://user-images.githubusercontent.com/57583574/119439945-54aafe80-bd5e-11eb-8c1a-f7b78660dfd0.png)
### Attention Block Architecture
![Attenion_Block](https://user-images.githubusercontent.com/57583574/119439971-6096c080-bd5e-11eb-8ee1-8912ddd96e25.png)

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
1. Before training please use "aug.py" to automatically augmentation train data ( flip , rotate 180 )
2. For training 
```
python main.py
```

3. For testing ( predicting )
```
python test.py
```
-----------------
You can get best model files via
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
This project blongs to Multimedia & Lab Team 6 members , Gachon Uni ( 2021 spring semester )
