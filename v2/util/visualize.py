import xdibias
import numpy as np
from PIL import Image
import pdb


img_e = Image.open('sample1/edge_out.png')
img_o = Image.open('sample1/sample_ortho.png')

data_e = np.asarray(img_e)
data_o = np.asarray(img_o.convert('RGB'))

out1 = data_o.copy()
out1[data_e[:,:,0]==255] = (255,0,0)
out1[data_e[:,:,1]==255] = (0,255,0)
img_oe = Image.fromarray(out1)
img_oe.save('sample1/sample_oe.png')
print 'orthor+edge saved.'

img_dsm = Image.open('sample1/sample_dsm.png')
data_dsm = np.asarray(img_dsm.convert('RGB'))
out2 = data_dsm.copy()
out2[data_e[:,:,0]==255] = (255,0,0)
out2[data_e[:,:,1]==255] = (0,255,0)
img_de = Image.fromarray(out2)
img_de.save('sample1/sample_de.png')
print 'dsm+edge saved.'


out3 = data_e.copy()
out3[data_e[:,:,0]==255] = (255,255,0)
out3[data_e[:,:,1]==255] = (0,255,0)
img_ce = Image.fromarray(out3)
img_ce.save('sample1/sample_ce.png')
print 'continuous edge saved.'
