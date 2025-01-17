"""
Author: 吴子昂
Email: ziangwu@stu.pku.edu.cn
Date: 2025-01-13
Description: This file is used to train CNN model on CIFAR-10 with different settings.
"""
'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
import wandb
import time
from torch.amp import autocast, GradScaler
import csv
import sys



# Function to set up the directories and files for saving results
def setup_directories(run_name):
    base_dir = 'results'  # Base directory where results will be stored

    # Create the base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Create a directory for the current run inside the base directory
    run_dir = os.path.join(base_dir, run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    
    # Define paths for CSV files and log file
    train_csv = os.path.join(run_dir, 'train.csv')
    test_step_csv = os.path.join(run_dir, 'test_step.csv')
    test_epoch_csv = os.path.join(run_dir, 'test_epoch.csv')
    log_file = os.path.join(run_dir, 'log.txt')
    
    # Write headers to CSV files if they don't exist
    for csv_file, header in zip([train_csv, test_step_csv, test_epoch_csv], 
                                [['train_loss', 'train_accuracy'], 
                                 ['test_loss', 'test_accuracy'], 
                                 ['epoch', 'test_accuracy']]):
        # if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)  # Write header row to CSV
    
    # Initialize log file with header text
    with open(log_file, 'w') as file:
        file.write('Training Log\n' + '='*20 + '\n')
    
    # Open files in append mode to allow further logging during training
    train_csv = open(train_csv, mode='a', newline='')
    test_step_csv = open(test_step_csv, mode='a', newline='')
    test_epoch_csv = open(test_epoch_csv, mode='a', newline='')
    log_file = open(log_file, mode='a')
    
    # Return file objects for further writing
    return train_csv, test_step_csv, test_epoch_csv, log_file

# Function to log messages to the log file
def log_to_file(log_file, message):
    with open(log_file, 'a') as f:
        f.write(message + '\n')  # Append message to log file

# Function to redirect print output to a log file
def log_print_output(log_file):
    sys.stdout = open(log_file, 'a')  # Redirect print output to log file

# Training
def train(epoch):
    global net, scaler, optimizer, scheduler, dtype, train_writer, log_file
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    # when use amp don't let the tensor be fp16, let amp decide it
    if not args.amp:
        net = net.to(dtype)
    start=time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if not args.amp:
            inputs, targets = inputs.to(device).to(dtype), targets.to(device)
        else:
            inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        if args.amp:
            with autocast(device_type='cuda', dtype=dtype):
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            if args.loss_scale:
                # loss scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        end=time.time()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if args.progress_bar:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        else:
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if args.wandb:
            wandb.log({"train_loss": train_loss/(batch_idx+1), "train_accuracy": correct/total})
        # log train loss and acc
        train_writer.writerow([train_loss/(batch_idx+1), correct/total])
        
    end=time.time()
    if args.wandb:
        wandb.log({"time(s/epoch)": end-start})
    # log training time spent per epoch
    # log_file.write(f"epoch:{epoch}\ttime(s/epoch): {end-start}\n")

# Test
def test(epoch):
    global best_acc, net, dtype, test_step_writer, test_epoch_writer, log_file
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # when use amp don't let the tensor be fp16, let amp decide it
            if not args.amp:
                inputs, targets = inputs.to(device).to(dtype), targets.to(device)
            else:
                inputs, targets = inputs.to(device), targets.to(device)
            if args.amp:
                with autocast(device_type='cuda', dtype=dtype):
                    outputs = net(inputs)
            else:
                outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if args.progress_bar:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            else:
                print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            if args.wandb:
                wandb.log({"test_loss": test_loss/(batch_idx+1), "test_accuracy": correct/total})
            # log test loss and accuracy
            test_step_writer.writerow([test_loss/(batch_idx+1), correct/total])
    # Save checkpoint.
    acc = 100.*correct/total
    test_epoch_writer.writerow([epoch, correct/total])
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/{run_name}-ckpt.pth')
        best_acc = acc
        log_file.write(f"epoch: {epoch}\tbest_acc: {best_acc}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--progress-bar', action='store_true')
    parser.add_argument('--dtype', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'])
    parser.add_argument('--model', type=str, default='VGG19', choices=[
                        'VGG19',
                        'ResNet18',
                        'PreActResNet18',
                        'GoogLeNet',
                        'DenseNet121',
                        'ResNeXt29_2x64d',
                        'MobileNet',
                        'MobileNetV2',
                        'DPN92',
                        # 'ShuffleNetG2',
                        'SENet18',
                        'ShuffleNetV2',
                        'EfficientNetB0',
                        'RegNetX_200MF',
                        'SimpleDLA'])
    parser.add_argument('--loss-scale', action='store_true')
    parser.add_argument('--amp', action='store_true')

    args = parser.parse_args()

    run_name=f'Cifar10-{args.model}-e{args.epochs}-b{args.batch_size}-'+args.dtype
    if args.amp:
        run_name += '-amp'
    if args.loss_scale:
        run_name += '-lossscale'
    if args.wandb:
        wandb_api_key=os.getenv("WANDB_API_KEY")
        try:
            wandb.login(key=wandb_api_key)
        except:
            print("set the WANDB_API_KEY")
        wandb.init(project='MTP', name=run_name, reinit=True)

    if args.dtype=="fp32":
        dtype=torch.float32
    elif args.dtype=="fp16":
        dtype=torch.float16
    elif args.dtype=="bf16":
        dtype=torch.bfloat16

    if args.amp and args.loss_scale:
        scaler = GradScaler()
    else:
        scaler = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    model_dict = {
        'VGG19': VGG('VGG19'),
        'ResNet18': ResNet18(),
        'PreActResNet18': PreActResNet18(),
        'GoogLeNet': GoogLeNet(),
        'DenseNet121': DenseNet121(),
        'ResNeXt29_2x64d': ResNeXt29_2x64d(),
        'MobileNet': MobileNet(),
        'MobileNetV2': MobileNetV2(),
        'DPN92': DPN92(),
        # 'ShuffleNetG2': ShuffleNetG2(),
        'SENet18': SENet18(),
        'ShuffleNetV2': ShuffleNetV2(1),
        'EfficientNetB0': EfficientNetB0(),
        'RegNetX_200MF': RegNetX_200MF(),
        'SimpleDLA': SimpleDLA()
    }
    
    # net initialization
    net = model_dict.get(args.model, VGG('VGG19'))  
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(f'./checkpoint/{run_name}-ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # prepare log 
    total_st = time.time()
    train_csv, test_step_csv, test_epoch_csv, log_file = setup_directories(run_name=run_name)
    train_writer = csv.writer(train_csv)
    test_step_writer = csv.writer(test_step_csv)
    test_epoch_writer = csv.writer(test_epoch_csv)

    # start training
    for epoch in range(start_epoch, start_epoch+args.epochs):
        train(epoch)
        test(epoch)
        scheduler.step()
    total_time = time.time()-total_st
    log_file.write(f"total training and test time: {total_time} s\n")