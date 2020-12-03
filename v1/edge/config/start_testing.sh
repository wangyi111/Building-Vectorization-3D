CUDA_VISIBLE_DEVICES=4 python test_xdib_image.py \
--dataroot ./datasets/dsm2lod \
--name test_Yi_edge2 \
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
--task dsm_edges \
--test_outdir test_output \
result_Coupled_UResNet50_nearest_from_u_net_resnet_50

