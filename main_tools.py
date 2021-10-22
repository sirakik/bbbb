import random
import datetime
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch



def train(model, optimizer, criterion, scheduler, data_loader):
    model.train()
    losses = []
    correct = 0
    for batch_idx, (data, label) in enumerate(tqdm(data_loader, leave=False, desc='# train')):
        data = data.to(output_device)
        label = label.to(output_device)

        output = model(data)

        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        _, predict = torch.max(output.data, 1)
        correct += (predict == label).sum().item()

    return np.mean(losses), 100. * correct / len(data_loader)


def test(model, criterion, data_loader):
    model.eval()
    losses = []
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(tqdm(data_loader, leave=False, desc='# test')):
            data = data.to(output_device)
            label = label.to(output_device)

            output = model(data)

            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimzer.step()

            losses.append(loss.item())
            _, predict = torch.max(output.data, 1)
            correct += (predict == label).sum().item()

    return np.mean(losses), 100. * correct / len(data_loader)



def load_model(model_args, log_dir, device, output_device, load_weight=False):
    model = model_class(num_class=model_args['num_class'],
                        in_channels=model_args['in_channels'],
                        residual=model_args['residual'],
                        dropout=model_args['dropout'],
                        num_person=model_args['num_person'],
                        t_kernel_size=model_args['t_kernel_size'],
                        layout=model_args['layout'],
                        strategy=model_args['strategy'],
                        hop_size=model_args['hop_size'],)

    model = model.to(output_device)

    # load weight
    if load_weight:
        print('# --- LOAD WEIGHT!!! ---')
        weights_path = model_args['weight_path']
        print('# Load weights from: ', weights_path)
        weights = torch.load(weights_path)
        weights = OrderedDict([[k.split('module.')[-1], v.to(output_device)] for k, v in weights.items()])
        model.load_state_dict(weights)

    # Multi GPU
    if type(device) is list:
        if len(device) > 1:
            model = torch.nn.DataParallel(model, device_ids=device, output_device=output_device)

    return model


def load_optimizer(optimizer_args, model, momentum=0.9, nesterov=True, weight_decay=0.0001):
    if optimizer_args['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=optimizer_args['lr'],
                                    momentum=momentum,
                                    nesterov=nesterov,
                                    weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[],
                                                         gamma=0.1)
    elif optimizer_args['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=optimizer_args['lr'],
                                     weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=lr_step,
                                                         gamma=0.1)

    return optimizer, scheduler


def print_log(input, log_dir, print_tf=True, log_tf=True):
    if print_tf:
        print(input)
    if log_tf:
        with open('{}/log.txt'.format(log_dir), 'a') as f:
            f.write(input+'\n')


def header_print(log_dir):
    print_log(
        '# -------------------------------------------------------------------------------------------------------------------------------\n'
        '# | epoch |:|train|:| mean loss | accuracy |:|test|:| mean loss | accuracy |:| current- |:|\n'
        '# -------------------------------------------------------------------------------------------------------------------------------',
        log_dir)


def result_print(epoch, log_dir, train_loss, train_accuracy, test_loss, test_accuracy):
    # Print  [9 = len(mean loss), 8 = len(accuracy)]
    train_loss = str(round(train_loss, 4)).rjust(9, ' ')
    train_accuracy = str(round(train_accuracy, 3)).rjust(8, ' ')
    test_loss = str(round(test_loss, 4)).rjust(9, ' ')
    test_accuracy = str(round(test_accuracy, 3)).rjust(8, ' ')

    print_log('# | {:5d} |:|     |:| {} | {} |:|    |:| {} | {} |:| {} |:|'.format(
        epoch, train_loss, train_accuracy, test_loss, test_accuracy,
        datetime.datetime.now().strftime('%H:%M:%S')), log_dir)


def make_confusion_matrix(confusion_matrix, log_dir):
    len_cm = len(confusion_matrix)
    for i in range(len_cm):
        sum_cm = np.sum(confusion_matrix[i])
        for j in range(len_cm):
            confusion_matrix[i][j] = 100 * (confusion_matrix[i][j] / sum_cm)
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    figname = os.path.join(log_dir, 'confusion_matrix.pdf')
    plt.savefig(figname, bbox_inches='tight')
    plt.close()

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
