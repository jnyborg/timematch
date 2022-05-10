#!/bin/bash

SOURCE_MODEL=pseltae_32VNH_shift_aug
SOURCE_TILE=32VNH
SOURCE=denmark/$SOURCE_TILE/2017

# Train
python train.py -e $SOURCE_MODEL --source $SOURCE --target $SOURCE --with_shift_aug --shift_aug_p 1.0

# Evaluate on new domains
TARGET_TILE=30TXT
TARGET=france/$TARGET_TILE/2017

python train.py -e $SOURCE_MODEL --source $SOURCE --target $TARGET --eval  # baseline result

TARGET_TILE=31TCJ
TARGET=france/$TARGET_TILE/2017

python train.py -e $SOURCE_MODEL --source $SOURCE --target $TARGET --eval  # baseline result

TARGET_TILE=33UVP
TARGET=austria/$TARGET_TILE/2017

python train.py -e $SOURCE_MODEL --source $SOURCE --target $TARGET --eval  # baseline result


SOURCE_MODEL=pseltae_30TXT_shift_aug
SOURCE_TILE=30TXT
SOURCE=france/$SOURCE_TILE/2017

# Train
python train.py -e $SOURCE_MODEL --source $SOURCE --target $SOURCE --with_shift_aug --shift_aug_p 1.0


# Evaluate on new domains
TARGET_TILE=32VNH
TARGET=denmark/$TARGET_TILE/2017
python train.py -e $SOURCE_MODEL --source $SOURCE --target $TARGET --eval  # baseline result

TARGET_TILE=31TCJ
TARGET=france/$TARGET_TILE/2017
python train.py -e $SOURCE_MODEL --source $SOURCE --target $TARGET --eval  # baseline result
