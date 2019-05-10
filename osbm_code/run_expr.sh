for split_idx in `seq 0 9`

do
    echo $split_idx

    # DGLFRM experiments

    ## For cora dataset (has features)
    # python train.py --dataset cora --hidden 32_50 --alpha0 10 --split_idx $split_idx --reconstruct_x 1 --epochs 500

    ## For citeseer dataset (has features)
    python train.py --dataset citeseer --hidden 32_50 --alpha0 10 --split_idx $split_idx --reconstruct_x 1 --epochs 500

    ## For nips12 dataset (no features)
    # python train.py --dataset nips12 --hidden 128_64_100 --alpha0 40 --split_idx $split_idx --reconstruct_x 0 --early_stopping 0 --epochs 500

    # DGLFRM-B experiments

    ## For nips12 dataset (no features)
    # python train.py --dataset nips12 --hidden 256_256 --alpha0 30 --split_idx $split_idx --reconstruct_x 0 --model dglfrm_b --epochs 1000 --deep_decoder 0

    ## For cora dataset (has features)
    # python train.py --dataset cora --hidden 32_100 --alpha0 10 --split_idx $split_idx --epochs 500 --reconstruct_x 1 --early_stopping 0 --model dglfrm_b

    ## For citeseer dataset (has features)
    # python train.py --dataset citeseer --hidden 32_100 --alpha0 10 --split_idx $split_idx --epochs 500 --reconstruct_x 1 --early_stopping 0 --model dglfrm_b

done
echo "************************************************"

