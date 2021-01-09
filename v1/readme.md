# 3D Building vectorization v1

dsm refinement (GAN) --> edge detection (UResNet) --> building vectorization (graph)

1. go to folder `dsm` for dsm refinement (multi-loss: cGAN + L1 + SN)

2. go to folder `edge` for edge detection (CRE)

3. run `vectorize.py` for sample area vectorization (threshold + non-maximum suppression --> corners, connectivity --> edges, polygonize --> polygons)

4. run `threshold.py` for sample area visualization and threshold evaulation

5. run `visualize.py` to draw vectorization result on orthophoto and dsm

(to do)
6. DSM --> DTM --> nDSM

(to do)
7. (currently) vectorized roofs --> vectorized walls and grounds
