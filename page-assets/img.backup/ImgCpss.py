# coding=utf-8

''' 
    ·利用 PIL 库进行图片压缩
    ·分为两种模式：文件夹压缩、单个图片压缩
    ·鸣谢：https://www.cnblogs.com/li1992/p/10675769.html
'''
import os
from PIL import Image

def get_size(file):
    # 获取文件大小:KB
    size = os.path.getsize(file)
    return size / 1024

def get_outfile(infile, outfile):
    if outfile:
        return outfile
    dir, suffix = os.path.splitext(infile)
    outfile = '{}-out{}'.format(dir, suffix)
    return outfile

def compress_image(infile, outfile='', mb=100, step=1, quality=80):
    """不改变图片尺寸压缩到指定大小
    :param infile: 压缩源文件
    :param outfile: 压缩文件保存地址
    :param mb: 压缩目标，KB
    :param step: 每次调整的压缩比率
    :param quality: 初始压缩比率
    :return: 压缩文件地址，压缩文件大小
    """
    o_size = get_size(infile)
    if o_size <= mb:
        return "SKIP IAMGE TOO SMALL"
    outfile = get_outfile(infile, outfile)
    while o_size > mb:
        im = Image.open(infile)
        im.save(outfile, quality=quality)
        if quality - step < 0:
            break
        quality -= step
        o_size = get_size(outfile)
        print(f'已将 {infile} 压缩至 {round(get_size(outfile), 1)} Kb，输出文件为 {outfile} 。')

if __name__ == "__main__":
    mod = input("对给定路径所有图片进行压缩\t【1】\n对单个图片进行压缩\t\t【2】\n<请输入选项>")
    path = input("输入文件夹路径或单个图片路径：")
    size = input("请输入大小上限（Kb）：")
    if mod == '1':
        img_list = []
        os.chdir(path)
        tmp_list = os.listdir()
        for f in tmp_list:
            if '.jpg' in f or '.png' in f:
                img_list.append(f)
        for p in img_list:
            compress_image(p, mb=int(size))
    else:
        compress_image(path, mb=int(size))
