#!/bin/bash

SOURCE_TILE=32VNH
SOURCE=denmark/$SOURCE_TILE/2017
TARGET_TILE=30TXT
TARGET=france/$TARGET_TILE/2017

# EMA rate
python train.py -e sensitivity_ema=0.0 --source $SOURCE --target $TARGET timematch --weights outputs/pseltae_$SOURCE_TILE
python train.py -e sensitivity_ema=0.9 --source $SOURCE --target $TARGET timematch --weights outputs/pseltae_$SOURCE_TILE
python train.py -e sensitivity_ema=0.99 --source $SOURCE --target $TARGET timematch --weights outputs/pseltae_$SOURCE_TILE
python train.py -e sensitivity_ema=0.999 --source $SOURCE --target $TARGET timematch --weights outputs/pseltae_$SOURCE_TILE
# python train.py -e sensitivity_ema=0.9999 --source $SOURCE --target $TARGET timematch --weights outputs/pseltae_$SOURCE_TILE
python train.py -e sensitivity_ema=1.0 --source $SOURCE --target $TARGET timematch --weights outputs/pseltae_$SOURCE_TILE

# pseudo label threshold
python train.py -e sensitivity_threshold=0.0 --source $SOURCE --target $TARGET timematch --weights outputs/pseltae_$SOURCE_TILE
python train.py -e sensitivity_threshold=0.5 --source $SOURCE --target $TARGET timematch --weights outputs/pseltae_$SOURCE_TILE
python train.py -e sensitivity_threshold=0.8 --source $SOURCE --target $TARGET timematch --weights outputs/pseltae_$SOURCE_TILE
# python train.py -e sensitivity_threshold=0.9 --source $SOURCE --target $TARGET timematch --weights outputs/pseltae_$SOURCE_TILE
python train.py -e sensitivity_threshold=0.95 --source $SOURCE --target $TARGET timematch --weights outputs/pseltae_$SOURCE_TILE

# # trade-off
python train.py -e sensitivity_tradeoff=0.5 --source $SOURCE --target $TARGET timematch --weights outputs/pseltae_$SOURCE_TILE
python train.py -e sensitivity_tradeoff=1.0 --source $SOURCE --target $TARGET timematch --weights outputs/pseltae_$SOURCE_TILE
# python train.py -e sensitivity_tradeoff=2.0 --source $SOURCE --target $TARGET timematch --weights outputs/pseltae_$SOURCE_TILE
python train.py -e sensitivity_tradeoff=5.0 --source $SOURCE --target $TARGET timematch --weights outputs/pseltae_$SOURCE_TILE
python train.py -e sensitivity_tradeoff=10.0 --source $SOURCE --target $TARGET timematch --weights outputs/pseltae_$SOURCE_TILE
