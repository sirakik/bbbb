import yaml
import argparse
import numpy as np
from tqdm import tqdm
import torch

import utils.main_tools as main_tools


def main():
    parser = argparse.ArgumentParser(description='Main config')
    parser.add_argument('--config', type=str, default=None)
    par = parser.parse_args()

    # load config
    with open(par.config, 'r') as f:
        args = yaml.load(f, Loader=yaml.SafeLoader)

    log_dir = args.pop('log_dir')

    # output device
    device = args.pop('device')
    output_device = device[0] if type(device) is list else device

    # model
    num_class = args['num_class']
    model = main_tools.load_model(args.pop('model_args'),
                                  log_dir,
                                  device,
                                  output_device,
                                  load_weight=True)

    # dataset
    data_loader = load_data_tools.load_data(args.pop('feeder_args'))


    correct = 0
    confusion_matrix = np.zeros((num_class, num_class))
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(tqdm(data_loader['test'])):
            data = data.to(output_device)
            label = label.to(output_device)

            output = model(data)

            _, predict = torch.max(output.data, 1)
            correct += (predict == label).sum().item()

            # confusion matrix
            for l, p in zip(label.view(-1), predict.view(-1)):
                confusion_matrix[l.long(), p.long()] += 1

    main_tools.make_confusion_matrix(confusion_matrix, log_dir)

    print('# Accuracy: {:.3f}'.format(100. * correct / len(data_loader['test'].dataset)))

if __name__=='__init__':
    main()
