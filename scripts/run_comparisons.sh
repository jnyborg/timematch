#!/bin/bash

SOURCE_MODEL=pseltae_32VNH
SOURCE_TILE=32VNH
SOURCE=denmark/$SOURCE_TILE/2017

TARGET_TILE=30TXT
TARGET=france/$TARGET_TILE/2017
python train.py -e dann_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET dann --weights outputs/pseltae_$SOURCE_TILE
python train.py -e cdan+e_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET dann --adv_loss CDAN+E --weights outputs/pseltae_$SOURCE_TILE
python train.py -e mmd_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET mmd --weights outputs/pseltae_$SOURCE_TILE
python train.py -e fixmatch_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET timematch --weights outputs/pseltae_$SOURCE_TILE --estimate_shift=False
python train.py -e jumbot_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET jumbot --weights outputs/pseltae_$SOURCE_TILE
python train.py -e upperbound_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET --train_on_target

TARGET_TILE=31TCJ
TARGET=france/$TARGET_TILE/2017
python train.py -e dann_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET dann --weights outputs/pseltae_$SOURCE_TILE
python train.py -e cdan+e_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET dann --adv_loss CDAN+E --weights outputs/pseltae_$SOURCE_TILE
python train.py -e mmd_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET mmd --weights outputs/pseltae_$SOURCE_TILE
python train.py -e fixmatch_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET timematch --weights outputs/pseltae_$SOURCE_TILE --estimate_shift=False
python train.py -e jumbot_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET jumbot --weights outputs/pseltae_$SOURCE_TILE
python train.py -e upperbound_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET --train_on_target

TARGET_TILE=33UVP
TARGET=austria/$TARGET_TILE/2017
python train.py -e dann_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET dann --weights outputs/pseltae_$SOURCE_TILE
python train.py -e cdan+e_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET dann --adv_loss CDAN+E --weights outputs/pseltae_$SOURCE_TILE
python train.py -e mmd_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET mmd --weights outputs/pseltae_$SOURCE_TILE
python train.py -e fixmatch_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET timematch --weights outputs/pseltae_$SOURCE_TILE --estimate_shift=False
python train.py -e jumbot_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET jumbot --weights outputs/pseltae_$SOURCE_TILE
python train.py -e upperbound_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET --train_on_target


SOURCE_MODEL=pseltae_30TXT
SOURCE_TILE=30TXT
SOURCE=france/$SOURCE_TILE/2017

TARGET_TILE=32VNH
TARGET=denmark/$TARGET_TILE/2017
python train.py -e dann_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET dann --weights outputs/pseltae_$SOURCE_TILE
python train.py -e cdan+e_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET dann --adv_loss CDAN+E --weights outputs/pseltae_$SOURCE_TILE
python train.py -e mmd_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET mmd --weights outputs/pseltae_$SOURCE_TILE
python train.py -e fixmatch_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET timematch --weights outputs/pseltae_$SOURCE_TILE --estimate_shift=False
python train.py -e jumbot_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET jumbot --weights outputs/pseltae_$SOURCE_TILE
python train.py -e upperbound_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET --train_on_target

TARGET_TILE=31TCJ
TARGET=france/$TARGET_TILE/2017
python train.py -e dann_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET dann --weights outputs/pseltae_$SOURCE_TILE
python train.py -e cdan+e_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET dann --adv_loss CDAN+E --weights outputs/pseltae_$SOURCE_TILE
python train.py -e mmd_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET mmd --weights outputs/pseltae_$SOURCE_TILE
python train.py -e fixmatch_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET timematch --weights outputs/pseltae_$SOURCE_TILE --estimate_shift=False
python train.py -e jumbot_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET jumbot --weights outputs/pseltae_$SOURCE_TILE
python train.py -e upperbound_$SOURCE_TILE\_to_$TARGET_TILE --source $SOURCE --target $TARGET --train_on_target
