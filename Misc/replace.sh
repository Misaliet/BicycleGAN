# python extractSeams.py ./replace/mytestL2RBorder125_sync.png ./replace/1.png
# python extractSeams.py ./replace/0_sync.png ./replace/1.png
echo "extract finish!"

cp -r ./test1 ./datasets/seamsL6/
echo "copy finish!"

# python test.py --dataroot ./datasets/seamsL6 --direction AtoB --model bicycle_gan --name seamsL7Cross1Loss_bicycleGAN_pretrained --results_dir ./results/replace --num_test 121 --phase test1  --no_flip --load_size 256 --gpu_ids -1 --input_nc 7 --dataset_mode aligned7c --sync
# python test.py --dataroot ./datasets/seamsL6 --direction AtoB --model bicycle_gan --name seamsL7Cross2Loss_bicycleGAN_pretrained --results_dir ./results/replace --num_test 121 --phase test1  --no_flip --load_size 256 --gpu_ids -1 --input_nc 7 --dataset_mode aligned7c --sync
# python test.py --dataroot ./datasets/seamsL6 --direction AtoB --model bicycle_gan --name seamsL7Cross3Loss_bicycleGAN_pretrained --results_dir ./results/replace --num_test 121 --phase test1  --no_flip --load_size 256 --gpu_ids -1 --input_nc 7 --dataset_mode aligned7c --sync
# python test.py --dataroot ./datasets/seamsL6 --direction AtoB --model bicycle_gan --name seamsL7CircleLossW200_bicycleGAN_pretrained --results_dir ./results/replace --num_test 121 --phase test1  --no_flip --load_size 256 --gpu_ids -1 --input_nc 7 --dataset_mode aligned7c --sync
# python test.py --dataroot ./datasets/seamsL6 --direction AtoB --model bicycle_gan --name seamsL4nBorder125_bicycleGAN_pretrained --results_dir ./results/replace --num_test 121 --phase test1  --no_flip --load_size 256 --gpu_ids -1 --input_nc 4 --dataset_mode aligned4c --sync
# python test.py --dataroot ./datasets/seamsL6 --direction AtoB --model bicycle_gan --name seamsL4nBorder125Loss_bicycleGAN_pretrained --results_dir ./results/replace --num_test 121 --phase test1  --no_flip --load_size 256 --gpu_ids -1 --input_nc 4 --dataset_mode aligned4c --sync
# python test.py --dataroot ./datasets/seamsL6 --direction AtoB --model bicycle_gan --name seamsL4Border125_bicycleGAN_pretrained --results_dir ./results/replace --num_test 121 --phase test1  --no_flip --load_size 256 --gpu_ids -1  --input_nc 4 --dataset_mode aligned4c --sync
# python test.py --dataroot ./datasets/seamsL6 --direction AtoB --model bicycle_gan --name seamsL4Border1Loss_bicycleGAN_pretrained --results_dir ./results/replace --num_test 121 --phase test1  --no_flip --load_size 256 --gpu_ids -1  --input_nc 4 --dataset_mode aligned4c --sync
# python test.py --dataroot ./datasets/seamsL6 --direction AtoB --model bicycle_gan --name seamsL7Cross2Loss_bicycleGAN_pretrained --results_dir ./results/replace --num_test 121 --phase test1  --no_flip --load_size 256 --gpu_ids -1 --input_nc 7 --dataset_mode aligned7c --sync
# python test.py --dataroot ./datasets/seamsL6 --direction AtoB --model bicycle_gan --name seamsLnewLoss3_bicycleGAN_pretrained --results_dir ./results/replace --num_test 121 --phase test1  --no_flip --load_size 256 --gpu_ids -1 --sync

# python test.py --dataroot ./datasets/seamsL6 --direction AtoB --model bicycle_gan --name seamsL7m1_bicycleGAN_pretrained --results_dir ./results/replace --num_test 121 --phase test1  --no_flip --load_size 256 --gpu_ids -1  --input_nc 7 --dataset_mode aligned7c --sync --mask_name nborder2_25.png --n_samples 1
# python test.py --dataroot ./datasets/seamsL6 --direction AtoB --model bicycle_gan --name seamsL7m3_bicycleGAN_pretrained --results_dir ./results/replace --num_test 121 --phase test1  --no_flip --load_size 256 --gpu_ids -1  --input_nc 7 --dataset_mode aligned7c --sync --mask_name nborder1_25.png --n_samples 1
# python test.py --dataroot ./datasets/seamsL6 --direction AtoB --model bicycle_gan --name seamsL7m5_bicycleGAN_pretrained --results_dir ./results/replace --num_test 121 --phase test1  --no_flip --load_size 256 --gpu_ids -1  --input_nc 7 --dataset_mode aligned7c --sync --mask_name nborder_p.png --n_samples 1
# python test.py --dataroot ./datasets/seamsL6 --direction AtoB --model bicycle_gan --name seamsL7m6_bicycleGAN_pretrained --results_dir ./results/replace --num_test 121 --phase test1  --no_flip --load_size 256 --gpu_ids -1  --input_nc 7 --dataset_mode aligned7c --sync --mask_name border2_25.png --n_samples 1
# python test.py --dataroot ./datasets/seamsL6 --direction AtoB --model bicycle_gan --name seamsL7m8_bicycleGAN_pretrained --results_dir ./results/replace --num_test 121 --phase test1  --no_flip --load_size 256 --gpu_ids -1  --input_nc 7 --dataset_mode aligned7c --sync --mask_name border1_25.png --n_samples 1
# python test.py --dataroot ./datasets/seamsL6 --direction AtoB --model bicycle_gan --name seamsL7m10_bicycleGAN_pretrained --results_dir ./results/replace --num_test 121 --phase test1  --no_flip --load_size 256 --gpu_ids -1  --input_nc 7 --dataset_mode aligned7c --sync --mask_name border_p.png --n_samples 1
# python test.py --dataroot ./datasets/seamsL6 --direction AtoB --model bicycle_gan --name seamsL7m11_bicycleGAN_pretrained --results_dir ./results/replace --num_test 121 --phase test1  --no_flip --load_size 256 --gpu_ids -1  --input_nc 7 --dataset_mode aligned7c --sync --mask_name cross1.png --n_samples 1
# python test.py --dataroot ./datasets/seamsL6 --direction AtoB --model bicycle_gan --name seamsL7m13_bicycleGAN_pretrained --results_dir ./results/replace --num_test 121 --phase test1  --no_flip --load_size 256 --gpu_ids -1  --input_nc 7 --dataset_mode aligned7c --sync --mask_name cross3.png --n_samples 1
# python test.py --dataroot ./datasets/seamsL6 --direction AtoB --model bicycle_gan --name seamsL7m14_bicycleGAN_pretrained --results_dir ./results/replace --num_test 121 --phase test1  --no_flip --load_size 256 --gpu_ids -1  --input_nc 7 --dataset_mode aligned7c --sync --mask_name circle.png --n_samples 1

# python test.py --dataroot ./datasets/seamsL6 --direction AtoB --model bicycle_ganML --name seamsL7m1l_bicycleGAN_pretrained --results_dir ./results/replace --num_test 121 --phase test1  --no_flip --load_size 256 --gpu_ids -1  --input_nc 7 --dataset_mode aligned7c --sync --mask_name nborder2_25.png --n_samples 1
# python test.py --dataroot ./datasets/seamsL6 --direction AtoB --model bicycle_ganML --name seamsL7m3l_bicycleGAN_pretrained --results_dir ./results/replace --num_test 121 --phase test1  --no_flip --load_size 256 --gpu_ids -1  --input_nc 7 --dataset_mode aligned7c --sync --mask_name nborder1_25.png --n_samples 1
# python test.py --dataroot ./datasets/seamsL6 --direction AtoB --model bicycle_ganML --name seamsL7m5l_bicycleGAN_pretrained --results_dir ./results/replace --num_test 121 --phase test1  --no_flip --load_size 256 --gpu_ids -1  --input_nc 7 --dataset_mode aligned7c --sync --mask_name nborder_p.png --n_samples 1
# python test.py --dataroot ./datasets/seamsL6 --direction AtoB --model bicycle_ganML --name seamsL7m6l_bicycleGAN_pretrained --results_dir ./results/replace --num_test 121 --phase test1  --no_flip --load_size 256 --gpu_ids -1  --input_nc 7 --dataset_mode aligned7c --sync --mask_name border2_25.png --n_samples 1
# python test.py --dataroot ./datasets/seamsL6 --direction AtoB --model bicycle_ganML --name seamsL7m8l_bicycleGAN_pretrained --results_dir ./results/replace --num_test 121 --phase test1  --no_flip --load_size 256 --gpu_ids -1  --input_nc 7 --dataset_mode aligned7c --sync --mask_name border1_25.png --n_samples 1
# python test.py --dataroot ./datasets/seamsL6 --direction AtoB --model bicycle_ganML --name seamsL7m10l_bicycleGAN_pretrained --results_dir ./results/replace --num_test 121 --phase test1  --no_flip --load_size 256 --gpu_ids -1  --input_nc 7 --dataset_mode aligned7c --sync --mask_name border_p.png --n_samples 1
# python test.py --dataroot ./datasets/seamsL6 --direction AtoB --model bicycle_ganML --name seamsL7m11l_bicycleGAN_pretrained --results_dir ./results/replace --num_test 121 --phase test1  --no_flip --load_size 256 --gpu_ids -1  --input_nc 7 --dataset_mode aligned7c --sync --mask_name cross1.png --n_samples 1
python test.py --dataroot ./datasets/seamsL6 --direction AtoB --model bicycle_ganML --name seamsL7m13l_bicycleGAN_pretrained --results_dir ./results/replace --num_test 121 --phase test1  --no_flip --load_size 256 --gpu_ids -1  --input_nc 7 --dataset_mode aligned7c --sync --mask_name cross3.png --n_samples 1
# python test.py --dataroot ./datasets/seamsL6 --direction AtoB --model bicycle_ganML --name seamsL7m14l_bicycleGAN_pretrained --results_dir ./results/replace --num_test 121 --phase test1  --no_flip --load_size 256 --gpu_ids -1  --input_nc 7 --dataset_mode aligned7c --sync --mask_name circle.png --n_samples 1
echo "generate finish!"

# python replace.py ./replace/mytestL2RBorder125_sync.png ./results/replace/test1_sync/images/ ./replace/mytestL2RBorder125_sync_l7c1l.png
# python replace.py ./replace/mytestL2RBorder125_sync.png ./results/replace/test1_sync/images/ ./replace/mytestL2RBorder125_sync_l7c2l.png
# python replace.py ./replace/mytestL2RBorder125_sync.png ./results/replace/test1_sync/images/ ./replace/mytestL2RBorder125_sync_l7c3l.png
# python replace.py ./replace/mytestL2RBorder125_sync.png ./results/replace/test1_sync/images/ ./replace/mytestL2RBorder125_sync_l4nb125.png
# python replace.py ./replace/mytestL2RBorder125_sync.png ./results/replace/test1_sync/images/ ./replace/mytestL2RBorder125_sync_l4nb125l.png
# python replace.py ./replace/mytestL2RBorder125_sync.png ./results/replace/test1_sync/images/ ./replace/mytestL2RBorder125_sync_l4b125.png
# python replace.py ./replace/mytestL2RBorder125_sync.png ./results/replace/test1_sync/images/ ./replace/mytestL2RBorder125_sync_l4b125l.png
# python replace.py ./replace/testZ.png ./results/replace/test1_sync/images/ ./replace/testZ_dl_Z.png

# python replace.py ./replace/0_sync.png ./results/replace/test1_sync/images/ ./replace/seamsL7m1.png
# python replace.py ./replace/0_sync.png ./results/replace/test1_sync/images/ ./replace/seamsL7m3.png
# python replace.py ./replace/0_sync.png ./results/replace/test1_sync/images/ ./replace/seamsL7m5.png
# python replace.py ./replace/0_sync.png ./results/replace/test1_sync/images/ ./replace/seamsL7m6.png
# python replace.py ./replace/0_sync.png ./results/replace/test1_sync/images/ ./replace/seamsL7m8.png
# python replace.py ./replace/0_sync.png ./results/replace/test1_sync/images/ ./replace/seamsL7m10.png
# python replace.py ./replace/0_sync.png ./results/replace/test1_sync/images/ ./replace/seamsL7m11.png
# python replace.py ./replace/0_sync.png ./results/replace/test1_sync/images/ ./replace/seamsL7m13.png
# python replace.py ./replace/0_sync.png ./results/replace/test1_sync/images/ ./replace/seamsL7m14.png

# python replace.py ./replace/0_sync.png ./results/replace/test1_sync/images/ ./replace/seamsL7m1l.png
# python replace.py ./replace/0_sync.png ./results/replace/test1_sync/images/ ./replace/seamsL7m3l.png
# python replace.py ./replace/0_sync.png ./results/replace/test1_sync/images/ ./replace/seamsL7m5l.png
# python replace.py ./replace/0_sync.png ./results/replace/test1_sync/images/ ./replace/seamsL7m6l.png
# python replace.py ./replace/0_sync.png ./results/replace/test1_sync/images/ ./replace/seamsL7m8l.png
# python replace.py ./replace/0_sync.png ./results/replace/test1_sync/images/ ./replace/seamsL7m10l.png
# python replace.py ./replace/0_sync.png ./results/replace/test1_sync/images/ ./replace/seamsL7m11l.png
python replace.py ./replace/0_sync.png ./results/replace/test1_sync/images/ ./replace/seamsL7m13l.png
# python replace.py ./replace/0_sync.png ./results/replace/test1_sync/images/ ./replace/seamsL7m14l.png
echo "replace finish!"