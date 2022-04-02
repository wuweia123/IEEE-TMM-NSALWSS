"""
NSALWSSOD ————Wu wei
Infer
"""

def main():
    import torch
    from torch.utils.data import DataLoader
    from dataset_loader import MyTestData
    from network.SalNet_dense import Net
    from functions import imsave
    import argparse
    from utils.evaluateFM import get_FM
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dataroot', type=str, default=r'D:/datasets/ECSSD', help='path to test data')
    parser.add_argument('--snapshot_root', type=str, default=r'./snapshot', help='path to snapshot')
    parser.add_argument('--salmap_root', type=str, default=r'./salmap', help='path to saliency map')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    args = parser.parse_args()
    cuda = torch.cuda.is_available()

    """""""""""~~~ dataset loader ~~~"""""""""
    test_dataRoot = args.test_dataroot

    if not os.path.exists(args.snapshot_root):
        os.mkdir(args.snapshot_root)
    if not os.path.exists(args.salmap_root):
        os.mkdir(args.salmap_root)


    MapRoot = args.salmap_root
    test_loader = torch.utils.data.DataLoader(MyTestData(test_dataRoot, transform=True),
                                              batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    print('data already')

    """"""""""" ~~~nets~~~ """""""""

    model = Net()  # load the model
    model.load_state_dict(torch.load(os.path.join(args.snapshot_root, 'snapshot_stage2_64adNRGAN16.pth')))

    if cuda:
        model = model.cuda()

        print(' testing ......... \n\n\n')

        model.eval()
        for id, (data, img_name, img_size) in enumerate(test_loader):
            inputs = data.cuda()
            _, _, height, width = inputs.size()
            output, _, _ = model(inputs)  # size: batchsize*1*n*n
            output = output.squeeze()
            output = output.cpu().data.resize_(height, width)
            imsave(os.path.join(MapRoot, img_name[0] + '.png'), output, img_size)

        # -------------------------- validation --------------------------- #
        torch.cuda.empty_cache()

        print("\n evaluating ....")
        F_measure, mae = get_FM(salpath=MapRoot + '\\', gtpath=test_dataRoot + '\\ECSSD-mask\\')  #
        print('F_measure:', F_measure), print('MAE:', mae)

if __name__ == '__main__':
    main()