CUDA_VISIBLE_DEVICES=0 python test_xdib_image.py \
--dataroot ./datasets/dsmANDmask2lod \
--name /space/export/data/davy_ks/pytorch/Coupled_cGAN/checkpoints/Coupled_UResNet_v2_nearest_from_u_net_resnet_50 \
--model pix2pix \
--which_model_netG Coupled_UResNet \
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
result_Coupled_UResNet50_nearest_from_u_net_resnet_50

