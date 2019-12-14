# coding=utf-8
import os
import time
import random
import cv2
import matplotlib.pyplot as plt
from shutil import copyfile

# 设定根目录
basedir = './'
save_dir = './chosen'

# 设定相关文件夹
dirs = ['unet', 'sun', 'hrdn']

# 指定想要统计的文件类型
whitelist = ['png']


# 随机选取某一张图片
def getPic():
    files = os.listdir("./hrdn")
    file = random.choice(files)
    return file


def visPic(file):
    filelists = []
    for dir in dirs:
        # filelists.append(os.path.join("./"+dir,file))
        filelists.append("./" + dir + "/" + file)
    plt.figure(figsize=(8, 6))
    for i in range(len(filelists)):
        file = filelists[i]
        dir_name = file.split('/')[1]
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(2, 2, i + 1)
        plt.imshow(img)
        plt.title(dir_name)
    plt.show()
    return filelists


def savePic(pics_chosen):
    # 判断保存路径是否存在，若不存在则创建
    isExists = os.path.exists(save_dir)
    if not isExists:
        os.mkdir(save_dir)
    # 遍历保存文件夹中的文件 找到新序号
    files = os.listdir(save_dir)
    if len(files) == 0:
        num = 1
    else:
        files.sort()
        latest = files[-1]
        num = int(latest.split('-')[0]) + 1
    # 保存新图片
    for pic in pics_chosen:
        base, dir_name, pic_name = pic.split('/')
        pic_type = pic_name.split('.')[1]  # 类型
        old_path = os.path.abspath('./' + dir_name)
        save_path = os.path.abspath('./' + save_dir)
        old = os.path.join(old_path, pic_name)
        new_name = str(num) + '-' + dir_name + '.' + pic_type
        new = os.path.join(save_path, new_name)
        copyfile(old, new)


if __name__ == '__main__':
    while True:
        pic_chosen = getPic()
        pics_chosen = visPic(pic_chosen)
        choice = input("save or not?[Y/N]:")
        if choice.lower() not in ['y', 'n']:
            continue
        elif choice.lower() == 'y':
            savePic(pics_chosen)
            print("save succeeded")
        while True:
            flag = input("continue choosing?[Y/N]:")
            if flag.lower() not in ['y', 'n']:
                continue
            else:
                break
        if flag.lower() == 'n':
            break
        else:
            pass

