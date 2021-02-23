CUDA_VISIBLE_DEVICES=3 python test_xdib_image.py \
--dataroot ./datasets/dsm2lod \
--name attn_edge \
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
--test_outdir test_output/attn_edge2 \
result_dsm

