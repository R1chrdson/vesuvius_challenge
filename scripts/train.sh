set -e

FOLDS="1"

for FOLD in $FOLDS
do
    echo "Training fold $FOLD"
    FOLD_IDX=$FOLD python source/train.py
done