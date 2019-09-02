#!/bin/bash

# python ./test.py --dataroot ./datasets/4for1 --direction AtoB --model bicycle_gan --name 4for1_bicycleGAN_pretrained --results_dir ./results/testWhole --num_test 144 --phase test  --no_flip --load_size 256 --gpu_ids -1
# python ./test.py --dataroot ./datasets/4for1 --direction AtoB --model bicycle_gan --name 4for1_bicycleGAN_pretrained --results_dir ./results/testWhole --num_test 144 --phase test  --no_flip --load_size 256 --gpu_ids -1 --sync
# python ./test.py --dataroot ./datasets/4for1 --direction AtoB --model bicycle_gan --name mytestL2RLV1_bicycleGAN_pretrained --results_dir ./results/testWhole --num_test 144 --phase test  --no_flip --load_size 256 --gpu_ids -1 --nz 1
# python ./test.py --dataroot ./datasets/4for1 --direction AtoB --model bicycle_gan --name mytestL2RLV1_bicycleGAN_pretrained --results_dir ./results/testWhole --num_test 144 --phase test  --no_flip --load_size 256 --gpu_ids -1 --nz 1 --sync

# python test.py --dataroot ./datasets/4for1 --direction AtoB --model bicycle_gan --name mytestL2RBorder125_bicycleGAN_pretrained --results_dir ./results/testWhole --num_test 144 --phase test  --no_flip --load_size 256 --gpu_ids -1 --input_nc 4 --dataset_mode aligned4c
# python test.py --dataroot ./datasets/4for1 --direction AtoB --model bicycle_gan --name mytestL2RBorder125_bicycleGAN_pretrained --results_dir ./results/testWhole --num_test 144 --phase test  --no_flip --load_size 256 --gpu_ids -1 --input_nc 4 --dataset_mode aligned4c --sync
# python test.py --dataroot ./datasets/4for1 --direction AtoB --model bicycle_gan --name mytestL2RBorder1LossW5_bicycleGAN_pretrained --results_dir ./results/testWhole --num_test 144 --phase test  --no_flip --load_size 256 --gpu_ids -1 --input_nc 4 --dataset_mode aligned4c
# python test.py --dataroot ./datasets/4for1 --direction AtoB --model bicycle_gan --name mytestL2RBorder1LossW5_bicycleGAN_pretrained --results_dir ./results/testWhole --num_test 144 --phase test  --no_flip --load_size 256 --gpu_ids -1 --input_nc 4 --dataset_mode aligned4c --sync
# python test.py --dataroot ./datasets/4for1 --direction AtoB --model bicycle_gan --name mytestL2RnBorder125_bicycleGAN_pretrained --results_dir ./results/testWhole --num_test 144 --phase test  --no_flip --load_size 256 --gpu_ids -1 --input_nc 4 --dataset_mode aligned4c
# python test.py --dataroot ./datasets/4for1 --direction AtoB --model bicycle_gan --name mytestL2RnBorder125_bicycleGAN_pretrained --results_dir ./results/testWhole --num_test 144 --phase test  --no_flip --load_size 256 --gpu_ids -1 --input_nc 4 --dataset_mode aligned4c --sync
# python test.py --dataroot ./datasets/4for1 --direction AtoB --model bicycle_gan --name mytestL2RnBorder125LossW10_bicycleGAN_pretrained --results_dir ./results/testWhole --num_test 144 --phase test  --no_flip --load_size 256 --gpu_ids -1 --input_nc 4 --dataset_mode aligned4c
# python test.py --dataroot ./datasets/4for1 --direction AtoB --model bicycle_gan --name mytestL2RnBorder125LossW10_bicycleGAN_pretrained --results_dir ./results/testWhole --num_test 144 --phase test  --no_flip --load_size 256 --gpu_ids -1 --input_nc 4 --dataset_mode aligned4c --sync

# python test.py --dataroot ./datasets/4for1 --direction AtoB --model bicycle_gan --name mytestL2RBorder125_bicycleGAN_pretrained --results_dir ./results/testWhole --num_test 144 --phase test  --no_flip --load_size 256 --gpu_ids -1 --input_nc 4 --dataset_mode aligned4c --sync

# python ./test.py --dataroot ./datasets/4for1 --direction AtoB --model bicycle_gan --name mytestL2RInput_bicycleGAN_pretrained --results_dir ./results/testWhole --where_add input --num_test 144 --phase test  --no_flip --load_size 256 --gpu_ids -1 --sync

# python ./test.py --dataroot ./datasets/4for1 --direction AtoB --model bicycle_gan --name z1_bicycleGAN_pretrained --results_dir ./results/testWhole --num_test 144 --phase test  --no_flip --load_size 256 --gpu_ids -1 --sync

# python test.py --dataroot ./datasets/4for1 --direction AtoB --model bicycle_gan --name 16z_bicycleGAN_pretrained --results_dir ./results/testWhole --num_test 144 --phase test  --no_flip --load_size 256 --gpu_ids -1 --sync --where_add input

# python test.py --dataroot ./datasets/4for1 --direction AtoB --model bicycle_ganBorderD --name mytestL2RBDL_bicycleGAN_pretrained --results_dir ./results/testWhole --num_test 144 --phase test  --no_flip --load_size 256 --gpu_ids -1 --sync

# python test.py --dataroot ./datasets/4for1 --direction AtoB --model bicycle_ganAZ --name az_bicycleGAN_pretrained --results_dir ./results/testWhole --num_test 144 --phase test  --no_flip --load_size 256 --gpu_ids -1 --sync --where_add input
# python test.py --dataroot ./datasets/z --direction AtoB --model bicycle_ganAZ --name az_bicycleGAN_pretrained --results_dir ./results/testWhole --num_test 144 --phase val1  --no_flip --load_size 256 --gpu_ids -1 --sync --where_add input

# python test.py --dataroot ./datasets/4for1 --direction AtoB --model bicycle_ganEZ --name ez_bicycleGAN_pretrained --results_dir ./results/testWhole --num_test 144 --phase test  --no_flip --load_size 256 --gpu_ids -1 --sync --where_add input
# python test.py --dataroot ./datasets/z --direction AtoB --model bicycle_ganEZ --name ez_bicycleGAN_pretrained --results_dir ./results/testWhole --num_test 144 --phase val1  --no_flip --load_size 256 --gpu_ids -1 --sync --where_add input

python test.py --dataroot ./datasets/z --direction AtoB --model bicycle_gan4TZ --name 4tz_bicycleGAN_pretrained --results_dir ./results/4tz --num_test 144 --phase val1  --no_flip --load_size 256 --gpu_ids -1 --sync --where_add input --n_samples 1

echo "generate finish!"

# python ./whole.py ./results/testWhole/test/images 3072 0.png
# python ./whole.py ./results/testWhole/test_sync/images 3072 0_sync.png
# python ./whole.py ./results/testWhole/test/images 3072 mytestL2RLV1.png
# python ./whole.py ./results/testWhole/test_sync/images 3072 mytestL2RLV1_sync.png
# python ./whole.py ./results/testWhole/test/images 3072 mytestL2RBorder125.png
# python ./whole.py ./results/testWhole/test_sync/images 3072 mytestL2RBorder125_sync.png
# python ./whole.py ./results/testWhole/test/images 3072 mytestL2RBorder1LossW5.png
# python ./whole.py ./results/testWhole/test_sync/images 3072 mytestL2RBorder1LossW5_sync.png
# python ./whole.py ./results/testWhole/test/images 3072 mytestL2RnBorder125.png
# python ./whole.py ./results/testWhole/test_sync/images 3072 mytestL2RnBorder125_sync.png
# python ./whole.py ./results/testWhole/test/images 3072 mytestL2RnBorder125LossW10.png
# python ./whole.py ./results/testWhole/test_sync/images 3072 mytestL2RnBorder125LossW10_sync.png

# python ./whole.py ./results/testWhole/test_sync/images 3072 testZ.png

# python ./whole.py ./results/testWhole/test_sync/images 3072 mytestL2RInput_sync.png

# python ./whole.py ./results/testWhole/test_sync/images 3072 mytestL2Rz1_sync.png

# python ./whole.py ./results/testWhole/test_sync/images 3072 16zb.png
# python ./whole.py ./results/16z/train_sync/images 1024 16z.png
# python ./whole.py ./results/16z/test_sync/images 1024 16zt.png

# python ./whole.py ./results/testWhole/test_sync/images 3072 mytestL2RBDL_sync.png

# python ./whole.py ./results/testWhole/test_sync/images 3072 az.png
# python ./whole.py ./results/testWhole/val1_sync/images 3072 az_z.png

# python ./whole.py ./results/testWhole/test_sync/images 3072 ez.png
# python ./whole.py ./results/ez/val1_sync/images 3072 ez_z.png

python ./whole.py ./results/4tz/val1_sync/images 3072 4tz_z.png
# python ./whole.py ./results/svz/val1_sync/images 3072 svz_z.png
# python ./whole.py ./results/z/val1_sync/images 3072 z_z.png

echo "combine images finish!"