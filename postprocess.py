import os
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
import PIL.Image as Image
from skimage.segmentation import slic
import cv2



class SlicCRF(object):
    def __init__(self, img_root, prob_root):
        self.img_root = img_root
        self.prob_root = prob_root

    def myfunc(self):
        files = os.listdir(self.prob_root)
        for i, img_name in enumerate(files):
            img = Image.open(os.path.join(self.img_root, img_name[:-4] + '.jpg')).convert('RGB')
            W, H = img.size
            img = np.array(img, dtype=np.uint8)
            probs = Image.open(os.path.join(self.prob_root, img_name[:-4] + '.png')).convert('L')
            probs = probs.resize((W, H))
            probs = np.array(probs, dtype=np.uint8)

            probs[probs > 20] = 255

            probs = probs.astype(np.float) / 255.0

            # superpixel
            img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float) / 255.0
            sp_label = slic(img_lab, n_segments=200, compactness=20)
            # in case of empty superpixels
            sp_onehot = np.arange(sp_label.max() + 1) == sp_label[..., None]
            sp_onehot = sp_onehot[:, :, sp_onehot.sum(0).sum(0) > 0]
            rs, cs, num = np.where(sp_onehot)
            for i, n in enumerate(num):
                sp_label[rs[i], cs[i]] = n
            sp_num = sp_label.max() + 1
            sp_prob = []
            for i in range(sp_num):
                sp_prob.append(probs[sp_label == i].mean())
            sp_prob = np.array(sp_prob)
            msk = np.zeros(probs.shape)
            for i in range(sp_num):
                msk[sp_label == i] = sp_prob[i]
            probs = msk

            probs = np.concatenate((1 - probs[None, ...], probs[None, ...]), 0)

            d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], 2)

            U = unary_from_softmax(probs)
            d.setUnaryEnergy(U)

            # This creates the color-dependent features and then add them to the CRF
            feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                              img=img, chdim=2)
            d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
            d.addPairwiseEnergy(feats, compat=10,
                                kernel=dcrf.DIAG_KERNEL,
                                normalization=dcrf.NORMALIZE_SYMMETRIC)

            # Run five inference steps.
            Q = d.inference(5)

            # Find out the most probable class for each pixel.

            MAP = np.array(Q)[1].reshape((H, W))
            MAP = (MAP * 255).astype(np.uint8)
            msk = Image.fromarray(MAP)
            msk.save(os.path.join(self.prob_root, img_name), 'png')
