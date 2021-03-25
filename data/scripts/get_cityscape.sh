#!/bin/bash
# cityscape dataset dataset https://www.cityscapes-dataset.com
# Download command: bash data/scripts/get_cityscapes.sh
# Train command: python train.py --data cityscapes.yaml
# Default dataset location is next to /yolov5:
#   /parent_folder
#     /coco
#     /cityscape
# you need to download dataset manually!!!!
# Download/unzip labels

d='../cityscapes/' # unzip directory
# url=https://www.cityscapes-dataset.com/downloads/

mkdir $d

#f1='https://www.cityscapes-dataset.com/file-handling/?packageID=28' # gtBbox_cityPersons_trainval.zip (2.2MB)
#f2='https://www.cityscapes-dataset.com/file-handling/?packageID=3' # leftImg8bit_trainvaltest.zip (11GB)
#urls=( $f1 $f2 )
names=( 'gtBbox_cityPersons_trainval.zip' 'leftImg8bit_trainvaltest.zip')


# Download/unzip images
for i in "${!names[@]}"; do
  #echo 'Downloading' "${names[i]}" '...'
  #curl -L "${urls[i]}" -o $d"${names[i]}"7
  mv '../'"${names[i]}" $d
  unzip -q $d"${names[i]}" -d $d
  # rm $d"${names[i]}" & # download, unzip, remove in background
done


python 'citytscape_tools.py'
