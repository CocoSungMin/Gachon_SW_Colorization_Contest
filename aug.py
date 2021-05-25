import os

import cv2
root_path = "data/"
train_dir = os.path.join(root_path, "train")
examples = [os.path.join(root_path, "train", dirs) for dirs in os.listdir(train_dir)]

print(len(examples))
for f in examples:
    file = cv2.imread(f)
    file = cv2.cvtColor(file , cv2.COLOR_BGR2RGB)
    conv_file = cv2.flip(file,1)
    rot_file = cv2.rotate(file , cv2.ROTATE_180)
    cv2.imwrite(os.path.join(train_dir, 'rot_image{}.png'.format(examples.index(f))), conv_file)
    cv2.imwrite(os.path.join(train_dir, 'rot_image{}.png'.format(examples.index(f))),rot_file)