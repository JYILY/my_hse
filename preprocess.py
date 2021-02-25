import os
import shutil
import time


start_time = time.time()

BASE_DIR = "F:/desktop/CUB_200_2011/CUB_200_2011/"
train_dir = "data/mini_cub/train/"
val_dir = "data/mini_cub/val/"
all_dir = [val_dir, train_dir]

num = [0,0]

is_val = 0

with open(BASE_DIR+"images.txt","r") as images, open(BASE_DIR+"train_test_split.txt","r") as tags:
    for image, tag in zip(images,tags):
        path = image.split(" ")[1].rstrip()
        class_name, file_name = path.split("/")
        idx, is_train = [int(s) for s in tag.split(" ")]
        num[is_train] = num[is_train] + 1
        save_dir = all_dir[is_train] + class_name
        save_path = save_dir + "/" +file_name
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        shutil.copy(BASE_DIR + "images/" + path, save_path)
        if idx == 100:
            break
end_time = time.time()


print('CUB200训练集和测试集划分完毕, 耗时 : %s' % (end_time - start_time))
print(f"num of train: {num[1]} \n"
      f"num of test: {num[0]} \n")