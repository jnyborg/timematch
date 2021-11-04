#!/bin/bash

SOURCE_TILE=32VNH
SOURCE=denmark/$SOURCE_TILE/2017
TARGET_TILE=30TXT
TARGET=france/$TARGET_TILE/2017

# Disable EMA
python train.py -e ablation_no_ema --source $SOURCE --target $TARGET timematch --weights outputs/pseltae_$SOURCE_TILE --ema_decay 0.0

# Disable domain-specific batch normalization
python train.py -e ablation_no_domain_specific_bn --source $SOURCE --target $TARGET timematch --weights outputs/pseltae_$SOURCE_TILE --domain_specific_bn=False

# Disable balanced source batches
python train.py -e ablation_no_balanced_source --source $SOURCE --target $TARGET timematch --weights outputs/pseltae_$SOURCE_TILE --balance_source=False

# Don't apply temporal shift to source data
python train.py -e ablation_no_shift_source --source $SOURCE --target $TARGET timematch --weights outputs/pseltae_$SOURCE_TILE --shift_source=False

# Inception Score
python train.py -e ablation_inception_score --source $SOURCE --target $TARGET timematch --weights outputs/pseltae_$SOURCE_TILE --shift_estimator=IS

# Entropy Score
python train.py -e ablation_entropy_score --source $SOURCE --target $TARGET timematch --weights outputs/pseltae_$SOURCE_TILE --shift_estimator=ENT
