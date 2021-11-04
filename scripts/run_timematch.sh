#!/bin/bash

SOURCE_MODEL=pseltae_32VNH
SOURCE_TILE=32VNH
SOURCE=denmark/$SOURCE_TILE/2017

# Source-only
python train.py -e $SOURCE_MODEL --source $SOURCE --target $SOURCE

# TimeMatch
TARGET_TILE=30TXT
TARGET=france/$TARGET_TILE/2017

python train.py -e $SOURCE_MODEL --source $SOURCE --target $TARGET --eval  # baseline result
python train.py -e timematch_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET timematch --weights outputs/pseltae_$SOURCE_TILE

TARGET_TILE=31TCJ
TARGET=france/$TARGET_TILE/2017

python train.py -e $SOURCE_MODEL --source $SOURCE --target $TARGET --eval  # baseline result
python train.py -e timematch_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET timematch --weights outputs/pseltae_$SOURCE_TILE

TARGET_TILE=33UVP
TARGET=austria/$TARGET_TILE/2017

python train.py -e $SOURCE_MODEL --source $SOURCE --target $TARGET --eval  # baseline result
python train.py -e timematch_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET timematch --weights outputs/pseltae_$SOURCE_TILE



SOURCE_MODEL=pseltae_30TXT
SOURCE_TILE=30TXT
SOURCE=france/$SOURCE_TILE/2017

# Source-only
python train.py -e $SOURCE_MODEL --source $SOURCE --target $SOURCE

# TimeMatch
TARGET_TILE=32VNH
TARGET=denmark/$TARGET_TILE/2017
python train.py -e $SOURCE_MODEL --source $SOURCE --target $TARGET --eval  # baseline result
python train.py -e timematch_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET timematch --weights outputs/pseltae_$SOURCE_TILE

TARGET_TILE=31TCJ
TARGET=france/$TARGET_TILE/2017
python train.py -e $SOURCE_MODEL --source $SOURCE --target $TARGET --eval  # baseline result
python train.py -e timematch_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET timematch --weights outputs/pseltae_$SOURCE_TILE
