"""
NSALWSSOD ————Wu wei
Train
"""


def main():
    import torch
    from functions import imsave
    from torch.utils.data import DataLoader
    from dataset_loader import MyTrainData, MyPseudoIterData
    from network.SalNet_dense import Net
    from network.discriminator import Dis
    import argparse
    from GANtrain import Trainer
    from postprocess import SlicCRF
    import os

    configurations = {
        1: dict(
            max_iteration=400000,
            lr=3e-6,
            lrd=1e-5,
            momentum=0.9,
            weight_decay=0.0005,
            spshot=10000,
            sshow=20,
        )
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='train', help='train or test')
    parser.add_argument('--param', type=str, default=False, help='path to pre-trained parameters')
    parser.add_argument('--train_start_stage', type=str, default='stage1', help='choose training point (stage1/stage2)')
    parser.add_argument('--train_dataroot', type=str, default=r'./datasets/DUTS_pseudo', help='path to train data')
    parser.add_argument('--stage1', type=str, default='stage1_training_map', help='path to training stage')
    parser.add_argument('--stage2', type=str, default='stage2_training_map', help='path to training stage')
    parser.add_argument('--snapshot_root', type=str, default=r'./snapshot', help='path to snapshot')
    parser.add_argument('--checkmap_root', type=str, default=r'./checkmap', help='path to test vggmap')
    parser.add_argument('--trainRGB_root', type=str, default=r'./datasets/DUTS_pseudo/DUTS-TR-Image',help='path to RGBImage')
    parser.add_argument('--stage1_result_root', type=str, default=r'./datasets/DUTS_pseudo/stage2_training_map', help='path to saliency map')
    parser.add_argument('-c', '--config', type=int, default=1, choices=configurations.keys())
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    args = parser.parse_args()
    cfg = configurations[args.config]
    cuda = torch.cuda.is_available()

    """""""""""~~~ dataset loader ~~~"""""""""

    train_dataRoot = args.train_dataroot
    stage1 = args.stage1
    stage2 = args.stage2

    if not os.path.exists(args.snapshot_root):
        os.mkdir(args.snapshot_root)


    snap_root = args.snapshot_root          # checkpoint
    MapRoot = args.stage1_result_root
    train_loader_stage1 = torch.utils.data.DataLoader(MyTrainData(train_dataRoot, stage=stage1, transform=True),
                                               batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    train_loader_stage2 = torch.utils.data.DataLoader(MyTrainData(train_dataRoot, stage=stage2, transform=True),
                                               batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    stage1_LabelIter_loader = torch.utils.data.DataLoader(MyPseudoIterData(train_dataRoot, transform=True),
                                              batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    print('data already')

    """"""""""" ~~~nets~~~ """""""""
    start_epoch = 0
    start_iteration = 0
    test = 0

    model = Net()  # load the model
    dis = Dis()

    if cuda:
        model = model.cuda()
        dis = dis.cuda()

#---------------------training stage1---------------------------------#
    if args.train_start_stage == 'stage1':
        print('stage1 begin')
        for ij in range(3):
            start_epoch = start_epoch + 5
            test = test + 1
            optimizer_model = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
            optimizer_dis = torch.optim.Adam(dis.parameters(), lr=cfg['lrd'], weight_decay=cfg['weight_decay'],
                                             betas=(0.5, 0.999))
            training = Trainer(
                cuda=cuda,
                model=model,
                dis=dis,
                optimizer_model=optimizer_model,
                optimizer_dis=optimizer_dis,
                train_loader=train_loader_stage1,
                max_iter=cfg['max_iteration'],
                snapshot=cfg['spshot'],
                outpath=snap_root,
                sshow=cfg['sshow'],
                clip=args.clip,
                test=test,
                stage=1
            )
            training.epoch = start_epoch
            training.iteration = start_iteration
            training.train()

        print('stage1 training end')

#------------------------Pseudo Label Iterating--------------------------#
        print('Pseudo Label Iterating')

        model.eval()
        for id, (data, img_name, img_size) in enumerate(stage1_LabelIter_loader):
            inputs = data.cuda()
            _, _, height, width = inputs.size()

            output, _, _ = model(inputs)  # size: batchsize*1*n*n    side5, side4, side3,

            output = output.squeeze()
            output = output.cpu().data.resize_(height, width)
            imsave(os.path.join(MapRoot, img_name[0] + '.png'), output, img_size)


        torch.cuda.empty_cache()

# --------------------------Pseudo Label postprocess --------------------------- #
        print('CRF processing')
        postprocess = SlicCRF(img_root=args.trainRGB_root, prob_root=MapRoot)
        postprocess.myfunc()

        print('Pseudo Label Updated')



#--------------------------------train from stage2---------------------------------------#
    print('stage2 begin')
    start_epoch = 0
    test = 0
    for ii in range(15): # stage training epoch
        start_epoch = start_epoch + 5
        test = test + 1
        optimizer_model = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        optimizer_dis = torch.optim.Adam(dis.parameters(), lr=cfg['lrd'], weight_decay=cfg['weight_decay'],
                                         betas=(0.5, 0.999))
        training = Trainer(
            cuda=cuda,
            model=model,
            dis=dis,
            optimizer_model=optimizer_model,
            optimizer_dis=optimizer_dis,
            train_loader=train_loader_stage2,
            max_iter=cfg['max_iteration'],
            snapshot=cfg['spshot'],
            outpath=snap_root,
            sshow=cfg['sshow'],
            clip=args.clip,
            test=test,
            stage=2
        )
        training.epoch = start_epoch
        training.iteration = start_iteration
        training.train()
    print('done!')


if __name__ == '__main__':
    main()