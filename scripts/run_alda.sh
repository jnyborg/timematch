#!/bin/bash

SOURCE_MODEL=pseltae_32VNH
SOURCE_TILE=32VNH
SOURCE=denmark/$SOURCE_TILE/2017

TARGET_TILE=30TXT
TARGET=france/$TARGET_TILE/2017
# python train.py -e alda_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET alda --weights outputs/pseltae_$SOURCE_TILE

python train.py -e alda_threshold=0.9 --source $SOURCE --target $TARGET alda --weights outputs/pseltae_$SOURCE_TILE
python train.py -e alda_threshold=0.8 --source $SOURCE --target $TARGET alda --weights outputs/pseltae_$SOURCE_TILE --pseudo_threshold 0.8
python train.py -e alda_threshold=0.7 --source $SOURCE --target $TARGET alda --weights outputs/pseltae_$SOURCE_TILE --pseudo_threshold 0.7



# TARGET_TILE=31TCJ
# TARGET=france/$TARGET_TILE/2017
# python train.py -e dann_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET dann --weights outputs/pseltae_$SOURCE_TILE

# TARGET_TILE=33UVP
# TARGET=austria/$TARGET_TILE/2017
# python train.py -e dann_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET dann --weights outputs/pseltae_$SOURCE_TILE


# SOURCE_MODEL=pseltae_30TXT
# SOURCE_TILE=30TXT
# SOURCE=france/$SOURCE_TILE/2017

# TARGET_TILE=32VNH
# TARGET=denmark/$TARGET_TILE/2017
# python train.py -e dann_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET dann --weights outputs/pseltae_$SOURCE_TILE

# TARGET_TILE=31TCJ
# TARGET=france/$TARGET_TILE/2017
# python train.py -e dann_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET dann --weights outputs/pseltae_$SOURCE_TILE
