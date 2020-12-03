CUDA_VISIBLE_DEVICES=3 python test_xdib_image.py \
--dataroot ./datasets/dsm2lod \
--name cgan_dsm_w0 \
--model pix2pix \
--which_model_netG Coupled_UResNet50 \
--which_direction AtoB \
--dataset_mode aligned \
--dataset_loader myDSM2Out \
--resize_or_crop crop \
--norm batch \
--overlap 128 \
--gpu_ids 0 \
--input_nc 1 \
--output_nc 1 \
--output_func none \
--fusion True \
--log_lvl info \
result_dsm

