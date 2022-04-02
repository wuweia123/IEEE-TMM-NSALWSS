import os
import numpy as np
import PIL.Image
import torch
from torch.utils import data


class MyTrainData(data.Dataset):
    """
    load data in a folder
    """
    mean_rgb = np.array([0.447, 0.407, 0.386])
    std_rgb = np.array([0.244, 0.250, 0.253])

    def __init__(self, root, stage, transform=False):
        super(MyTrainData, self).__init__()
        self.root = root
        self.stage = stage
        self._transform = transform
        img_root = os.path.join(self.root, 'DUTS-TR-Image')
        lbl_root = os.path.join(self.root, stage)
        file_names = os.listdir(img_root) 
        self.img_names = []
        self.lbl_names = []

        for i, name in enumerate(file_names):
            if not name.endswith('.jpg'):
                continue
            self.lbl_names.append(
                os.path.join(lbl_root, name[:-4]+'.png')
            )
            self.img_names.append(
                os.path.join(img_root, name)
            )

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # load image
        img_file = self.img_names[index]
        img = PIL.Image.open(img_file)
        img = img.convert('RGB')
        img = img.resize((256, 256))
        img = np.array(img, dtype=np.uint8)

        lbl_file = self.lbl_names[index]
        lbl = PIL.Image.open(lbl_file)
        lbl = lbl.convert('L')
        lbl = lbl.resize((256, 256))
        lbl = np.array(lbl)
        lbl = lbl / 255



        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl

    # Translating numpy_array into format that pytorch can use on Code.
    def transform(self, img, lbl):
        img = img.astype(np.float64)/255.0
        lbl = lbl.astype(np.float32)
        # lbl[lbl != 0] = 1
        img = img - self.mean_rgb
        img = img / self.std_rgb
        img = img.transpose(2, 0, 1)  # to verify #256*256*3 to 3*256*256
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).float()

        return img, lbl

class MyPseudoIterData(data.Dataset):
    """
    load data in a folder
    """
    mean_rgb = np.array([0.447, 0.407, 0.386])
    std_rgb = np.array([0.244, 0.250, 0.253])

    def __init__(self, root, transform=False):
        super(MyPseudoIterData, self).__init__()
        self.root = root
        self._transform = transform

        img_root = os.path.join(self.root, 'DUTS-TR-Image')
        file_names = os.listdir(img_root)
        self.img_names = []
        self.names = []

        for i, name in enumerate(file_names):
            if not name.endswith('.jpg'):
                continue
            self.img_names.append(
                os.path.join(img_root, name)
            )
            self.names.append(name[:-4])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # load image
        img_file = self.img_names[index]
        img = PIL.Image.open(img_file).convert('RGB')
        img_size = img.size
        img = img.resize((256, 256))
        img = np.array(img, dtype=np.uint8)

        if self._transform:
            img = self.transform(img)
            return img, self.names[index], img_size
        else:
            return img, self.names[index], img_size

    def transform(self, img):
        img = img.astype(np.float64)/255.0
        img -= self.mean_rgb
        img /= self.std_rgb
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        return img

class MyPostData(data.Dataset):
    """
    load data in a folder
    """
    mean_rgb = np.array([0.447, 0.407, 0.386])
    std_rgb = np.array([0.244, 0.250, 0.253])

    def __init__(self, root):
        super(MyPostData, self).__init__()
        self.root = root

        img_root = os.path.join(self.root, 'DUTS-TR-Image')
        file_names = os.listdir(img_root)
        self.img_names = []
        self.names = []

        for i, name in enumerate(file_names):
            if not name.endswith('.jpg'):
                continue
            self.img_names.append(
                os.path.join(img_root, name)
            )
            self.names.append(name[:-4])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # load image
        img_file = self.img_names[index]
        img = PIL.Image.open(img_file).convert('RGB')
        img_size = img.size
        img = img.resize((256, 256))
        img = np.array(img, dtype=np.uint8)

        if self._transform:
            img = self.transform(img)
            return img, self.names[index], img_size
        else:
            return img, self.names[index], img_size

    def transform(self, img):
        img = img.astype(np.float64)/255.0
        img -= self.mean_rgb
        img /= self.std_rgb
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        return img

class MyTestData(data.Dataset):
    """
    load data in a folder
    """
    mean_rgb = np.array([0.447, 0.407, 0.386])
    std_rgb = np.array([0.244, 0.250, 0.253])

    def __init__(self, root, transform=False):
        super(MyTestData, self).__init__()
        self.root = root
        self._transform = transform

        img_root = os.path.join(self.root, 'ECSSD-image')#test dataset document
        file_names = os.listdir(img_root)
        self.img_names = []
        self.names = []

        for i, name in enumerate(file_names):
            if not name.endswith('.jpg'):
                continue
            self.img_names.append(
                os.path.join(img_root, name)
            )
            self.names.append(name[:-4])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # load image
        img_file = self.img_names[index]
        img = PIL.Image.open(img_file).convert('RGB')
        img_size = img.size
        img = img.resize((256, 256))
        img = np.array(img, dtype=np.uint8)

        if self._transform:
            img = self.transform(img)
            return img, self.names[index], img_size
        else:
            return img, self.names[index], img_size

    def transform(self, img):
        img = img.astype(np.float64)/255.0
        img -= self.mean_rgb
        img /= self.std_rgb
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        return img
