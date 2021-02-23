# 3D Building vectorization v2

dsm refinement (GAN) --> edge detection (UResNet) --> building vectorization (graph)

1. go to folder `attn_dsm` for dsm refinement (multi-loss: cGAN + L1 + SN)

2. go to folder `attn_edge` for edge detection (CRE)

3. run `vectorize.py` for sample area vectorization (threshold + non-maximum suppression --> corners, connectivity --> edges, polygonize --> polygons)

4. run `threshold.py` for sample area visualization and threshold evaulation

5. run `visualize.py` to draw vectorization result on orthophoto and dsm


6. DSM --> DTM --> nDSM


7. run `plot3d.m` for 3D modeling (vectorized roofs --> vectorized walls and grounds --> 3D model)
