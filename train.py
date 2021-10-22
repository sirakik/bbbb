import os
import csv
import yaml
import random
import shutil
import argparse
import numpy as np
from tqdm import tqdm
import torch

import utils.main_tools as main_tools



class Processor:
    def __init__(self, args, log_dir):

        # gpu device
        device = args.pop('device')
        output_device = device[0] if type(device) is list else device
        torch.backends.cudnn.benchmark = True # magic spell

        # model
        model = main_tools.load_model(args.pop('model_args'),
                                      log_dir,
                                      device,
                                      output_device)

        # optimizer
        optimizer, scheduler = main_tools.load_optimizer(args.pop('optimizer_args'),
                                                         model)

        # loss function
        criterion = torch.nn.CrossEntropyLoss()

        # dataset
        data_loader = load_data_tools.load_data(args.pop('feeder_args'))

        # log list
        train_loss_list = []
        train_acc_list = []
        test_loss_list = []
        test_acc_list = []

        # start
        print('\n# Start! :)')
        for epoch in range(1, args['epoch']+1):
            # print header
            if ((epoch - 1) % 10) == 0:
                main_tools.header_print(log_dir)

            train_loss, train_accuracy = main_tools.train(model,
                                                          optimizer,
                                                          criteirion,
                                                          scheduler,
                                                          data_loader['train'])
            test_loss, test_accuracy = main_tools.test(model,
                                                       criterion,
                                                       data_loader['test'])

            train_loss_list.append(train_loss)
            train_acc_list.append(train_accuracy)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_accuracy)

            main_tools.result_print(epoch, log_dir,
                                    train_loss, train_accuracy,
                                    test_loss, test_accuracy)

            # save best model
            if (len(test_acc_list) - 1) == np.argmax(test_acc_list):
                torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pt'))

        main_tools.print_log('# Best test accuracy: {:.3f}[%]'.format(np.max(test_acc_list)), log_dir)

        # plot csv
        train_loss_list.insert(0, 'train_loss')
        train_acc_list.insert(0, 'train_acc')
        test_loss_list.insert(0, 'test_loss')
        test_acc_list.insert(0, 'test_acc')
        with open(log_dir + '/loss_accuracy.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow(train_loss_list)
            writer.writerow(train_acc_list)
            writer.writerow(test_loss_list)
            writer.writerow(test_acc_list)

        print('\n# bye :)\n')


if __name__ == '__main__':
    print('# Spatial Temporal GCN Ver.Shiraki')
    parser = argparse.ArgumentParser(description='st-gcn_srk')
    parser.add_argument('--config', type=str, default=None)
    p = parser.parse_args()

    main_tools.init_seed(seed=1)

    # load config
    with open(p.config, 'r') as f:
        args = yaml.load(f, Loader=yaml.SafeLoader)

    # log dir
    log_dir = args.pop('log_dir')
    print('\n# log_dir: ', log_dir)

    # mkdir log_dir
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
        with open(log_dir + '/log.txt', 'w'):
            pass
    else:
        print('# warning: log_dir[{}] already exist.'.format(log_dir))
        answer = input('# continue? y/n : ')
        if answer == 'y':
            pass
        else:
            print('# bye :)')
            exit()

    # save config
    main_tools.print_log('# Parameters:\n{}\n'.format(str(args)), log_dir=log_dir, print_tf=False)
    shutil.copyfile(p.config, os.path.join(log_dir, 'config.yaml'))

    # start Processor
    Processor(args, log_dir)
