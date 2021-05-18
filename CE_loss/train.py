import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from training_layers import PriorBoostLayer, NNEncLayer, ClassRebalanceMultLayer, NonGrayMaskLayer
from data_loader import TrainImageFolder
from model import Color_model
from unet import UNet

# original_transform = transforms.Compose([
#     transforms.Scale(256),
#     transforms.RandomCrop(224),
#     transforms.RandomHorizontalFlip(),
#     #transforms.ToTensor()
# ])

model_dict={
    'CIC':{'structure':Color_model,'down_rate':4},
    'UNet':{'structure':UNet,'down_rate':2}
}


def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Image preprocessing, normalization for the pretrained resnet
    train_set = TrainImageFolder(args.img_list_path,down_rate=model_dict[args.model]['down_rate'])

    # Build data loader
    data_loader = torch.utils.data.DataLoader(train_set, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)

    # Build the models
    model=nn.DataParallel(model_dict[args.model]['structure']()).cuda()

    if not args.train_from_scratch and os.path.exists(args.checkpoint_path):
        model.load_state_dict(torch.load(args.checkpoint_path))
        start_epoch=int(args.checkpoint_path.split(os.path.sep)[-1].split('.')[0].split('-')[1])
    else:
        start_epoch=1
    encode_layer=NNEncLayer()
    boost_layer=PriorBoostLayer()
    nongray_mask=NonGrayMaskLayer()
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(reduce=False).cuda()
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr = args.learning_rate)
    

    # Train the models
    print("[Start Training]")
    total_step = len(data_loader)
    for epoch in range(start_epoch,args.num_epochs):
        print("-----Epoch {}/{}-----".format(epoch,args.num_epochs))
        start_time=time.time()
        error_count=0
        for i, (images, img_ab) in enumerate(data_loader):
            try:
                # Set mini-batch dataset
                images = images.unsqueeze(1).float().cuda()
                img_ab = img_ab.float()
                encode,max_encode=encode_layer.forward(img_ab)
                targets=torch.Tensor(max_encode).long().cuda()
                boost=torch.Tensor(boost_layer.forward(encode)).float().cuda()
                mask=torch.Tensor(nongray_mask.forward(img_ab)).float().cuda()
                boost_nongray=boost*mask
                outputs = model(images)#.log()
                output=outputs[0].cpu().data.numpy()
                out_max=np.argmax(output,axis=0)

                # print('set',set(out_max.flatten()))
                loss = (criterion(outputs,targets)*(boost_nongray.squeeze(1))).mean()
                #loss=criterion(outputs,targets)
                #multi=loss*boost_nongray.squeeze(1)

                model.zero_grad()

                loss.backward()
                optimizer.step()

                # Print log info
                if i % args.log_step == 0:
                    cost_time=time.time()-start_time
                    print('Epoch [{}/{}], Step [{}/{}], Loss:{:.4f}, error_count:{}, cost_time:{:.3f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(),error_count, cost_time))
                    start_time=time.time()

                # Save the model checkpoints
                if (i + 1) % args.save_step == 0:
                    torch.save(model.state_dict(), os.path.join(
                        args.model_path, 'model-{}-{}.ckpt'.format(epoch, i + 1)))
            except:
                error_count+=1
                print("Epoch [{}/{}], Step [{}/{}] Error!,Error count:{}".format(
                    epoch, args.num_epochs, i, total_step,error_count
                ))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default = 'CIC', help = 'choose')
    parser.add_argument('--model_path', type = str, default = '../model/models/', help = 'path for saving trained models')
    parser.add_argument('--train_from_scratch',action='store_true')
    parser.add_argument('--checkpoint_path',type=str,default='../model/models/model-100-10.ckpt',help='checkpoint path')
    parser.add_argument('--crop_size', type = int, default = 224, help = 'size for randomly cropping images')
    parser.add_argument('--img_list_path', type = str, default = '../data/ILSVRC2017-DET_train.txt', help = 'directory for resized images')
    parser.add_argument('--log_step', type = int, default = 1, help = 'step size for prining log info')
    parser.add_argument('--save_step', type = int, default = 216, help = 'step size for saving trained models')

    # Model parameters
    parser.add_argument('--num_epochs', type = int, default = 200)
    parser.add_argument('--batch_size', type = int, default = 60)
    parser.add_argument('--num_workers', type = int, default = 8)
    parser.add_argument('--learning_rate', type = float, default = 1e-3)
    args = parser.parse_args()
    print(args)
    main(args)
