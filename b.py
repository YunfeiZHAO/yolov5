from PIL import Image, ExifTags

im = Image.open('../Cityscapes/images/train/aachen_000000_000019_leftImg8bit.png')
print(im.verify())  # PIL verify