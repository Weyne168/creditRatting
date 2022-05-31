import sys

sys.path.append('.')
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import accuracy_score
from dataset.data_loader import get_loader
from models.model import CNNDLGA
from utils.torch_utils import select_device, opts_parser
from utils.mlogger import logger
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

'''
input_size=4
acc(0.3) = 0.7923
auc(0.3) = 0.8616
fpr=0.20721271393630367  
fnr=0.20547945205453863
'''
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--repeat', type=int, default=1, help='sample repeats')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=128, help='size of each image batch')
parser.add_argument('--workers', type=int, default=1, help='number of Pytorch DataLoader workers')
parser.add_argument('--gpus', default='0', type=str, help='number of gpus for training')

parser.add_argument('--resume', action='store_true', default=False, help='resume training flag')
parser.add_argument('--weight', type=str, default='ckp/model_best_acc.pkl', help='models are saved here')
parser.add_argument('--ck_pth', type=str, default='ckp', help='the models are saved here')

parser.add_argument('--trainset', type=str, default='data/ch.train.dat', help='training set')
parser.add_argument('--valset', type=str, default='data/ch.test.dat', help='test/val data set')
parser.add_argument('--seq_len', type=int, default=3, help='sequence length')
parser.add_argument('--num_feat', type=int, default=22, help='CHR is 49, US is 223, CH is 22')
parser.add_argument('--embedding_dim', type=int, default=32, help='original features` embedding dim')
parser.add_argument('--cate_nums', type=str, default='65',
                    help='number of categories for each category feature. CHR:7+1, US:11+1, CH:64+1')
parser.add_argument('--cross_feats', type=str, default='1',
                    help='number of kept high order features for each order')
parser.add_argument('--nclass', type=int, default=2, help='2 or 20(CHR), number of rattings')

parser.add_argument('--dist-url', default='tcp://127.0.0.1:9999', type=str, help='distributed training init method')
parser.add_argument('--rank', default=0, type=int, help='distributed training node rank')
parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
parser.add_argument('--backend', default='nccl', type=str, help='distributed backend')


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def freeze_bn(m):
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Dropout2d):
        # print(type(m))
        m.eval()


def train(opt, log):
    device = select_device(gpus=opt.gpus)
    # Hyper Parameters
    seq_len = opt.seq_len
    input_num_feat = opt.num_feat
    cate_nums = opt.cate_nums
    K = opt.embedding_dim
    cross_feat_nums = opt.cross_feats
    nclass = opt.nclass

    num_epochs = opt.epochs
    batch_size = opt.batch_size
    learning_rate = opt.lr

    batch_loss = 0
    best_loss = float('inf')
    best_acc, best_auc = 0, 0

    print("Loading data...")
    train_loader = get_loader(dat_file=opt.trainset, stride=opt.seq_len, repeat=opt.repeat,
                              batch_size=batch_size,
                              num_workers=opt.workers,
                              is_testing=False)

    val_loader = get_loader(dat_file=opt.valset, stride=opt.seq_len, repeat=1,
                            batch_size=batch_size // 4,
                            num_workers=opt.workers,
                            shuffle=False,
                            is_testing=True)
    print("train/val: {:d}/{:d}".format(len(train_loader), len(val_loader)))
    print("==================================================================================")
    # device, seq_len, n_num_feats, cls_feats=[2], embed_dim=64,keep_cross_feats=[128, 64, 32], nclass=19
    model = CNNDLGA(device,
                    seq_len=seq_len,
                    n_num_feats=input_num_feat,
                    cls_feats=cate_nums,
                    embed_dim=K,
                    keep_cross_feats=cross_feat_nums,
                    nclass=nclass)

    # Optimizer
    # optimizer = torch.optim.SGD(CNNDLGA.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    start_epoch = -1
    if opt.resume and os.path.exists(opt.weight):  # Load previously saved model
        print('loading model from %s...' % opt.weight)
        ck = torch.load(opt.weight)
        pretrained_net_dict = ck['model_params']
        cross_layers = ck['cross_layers']
        cls_encoders = ck['cls_encoders']
        model_dict = model.state_dict()

        for k, v in pretrained_net_dict.items():
            print(k)
            if k in model_dict.keys():
                model_dict[k] = v

        model.load_state_dict(model_dict)
        for i, cross_layer in enumerate(cross_layers):
            layer_dict = model.cross_layers[i].state_dict()
            for k, v in cross_layer.items():
                if k in layer_dict:
                    print('cross_layer', k)
                    layer_dict[k] = v
            model.cross_layers[i].load_state_dict(layer_dict)

        for i, cls_encoder in enumerate(cls_encoders):
            layer_dict = model.featEmbedding.cls_encoders[i].state_dict()
            for k, v in cls_encoder.items():
                if k in layer_dict:
                    print('cls_encoder', k)
                    layer_dict[k] = v
            model.featEmbedding.cls_encoders[i].load_state_dict(layer_dict)

        # optimizer = ck['optimizer']
        start_epoch = ck['epoch']
        best_acc = ck['best_acc']

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 400], gamma=0.1)

    if torch.cuda.is_available():
        model.to(device)
        if torch.cuda.device_count() > 0:
            # dist.init_process_group(backend=opt.backend, init_method=opt.dist_url, world_size=opt.world_size,rank=opt.rank)
            # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=list(range(num_gpus)))
            # model = torch.nn.parallel.DataParallel(model, device_ids=opt.gpus)
            # model = torch.nn.DataParallel(model, device_ids=opt.gpus)
            pass
    # test(CNNDLGA, val_loader)

    with torch.no_grad():
        testing_acc, testing_auc = test(model, val_loader, opts)

    print("==================================================================================")
    print("Training Start..")

    # Train the Model
    total_step = len(train_loader)
    # lab_weights = torch.ones(nclass).to(device)
    for epoch in range(start_epoch + 1, num_epochs):
        print('learning rate==', scheduler.get_lr())
        model.train()

        for i, (x_num, x_cls, y) in enumerate(train_loader):
            # Convert torch tensor to Variable
            if torch.cuda.is_available():
                x_num = x_num.to(device)
                x_cls = x_cls.to(device)
                y = y.to(device)

            # Forward + Backward + Optimize
            out, loss1, loss2 = model(x_cls, x_num, y)
            loss = 1.0 * loss1  # + 1.0 * loss2
            optimizer.zero_grad()  # zero the gradient buffer
            loss.backward()
            l1_regularization(model.cross_layers, 0.1)
            optimizer.step()

            batch_loss += loss.item()
            if i % 10 == 0:
                # Print log info
                result = F.softmax(out, dim=1)
                result = torch.argmax(result, dim=1)
                res = result.data.cpu().numpy()

                labels = y.cpu().numpy()
                acc = accuracy_score(res, labels)

                print(
                    'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Training_acc: %5.4f'
                    % (epoch, num_epochs, i, total_step,
                       batch_loss / 10, acc))
                log.logging(
                    'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Training_acc: %5.4f'
                    % (epoch, num_epochs, i, total_step,
                       batch_loss / 10, acc)
                )
                if best_loss > batch_loss / 10:
                    cross_layers = []
                    cls_encoders = []
                    for cross_layer in model.cross_layers:
                        cross_layers.append(cross_layer.state_dict())

                    for cls_encoder in model.featEmbedding.cls_encoders:
                        cls_encoders.append(cls_encoder.state_dict())
                    # Save the Model
                    ck = {
                        'epoch': epoch,
                        'n_iter': i,
                        'best_loss': best_loss,
                        'best_acc': best_acc,
                        'optimizer': optimizer.state_dict(),
                        'model_params': model.state_dict(),
                        'cross_layers': cross_layers,
                        'cls_encoders': cls_encoders
                    }
                    torch.save(ck, os.path.join(opt.ck_pth, 'best_model.pkl'))
                    best_loss = batch_loss / 10
                batch_loss = 0
                # break
        scheduler.step()

        log.logging("==================================================================================")
        log.logging("Testing Start..")
        with torch.no_grad():
            testing_acc, testing_auc = test(model, val_loader, opts)

        log.logging('Acc=%.4f' % (testing_acc))
        # Save the Model
        if epoch % 50 == 0 or testing_acc > best_acc or testing_auc > best_auc:
            if testing_acc > best_acc:
                best_acc = testing_acc
                model_name = 'best_acc'
            else:
                model_name = str(epoch)

            if testing_auc > best_auc:
                best_auc = testing_auc
                model_name += 'best_auc'

            if epoch % 50 == 0:
                model_name += '_' + str(epoch)

            cross_layers = []
            cls_encoders = []
            for cross_layer in model.cross_layers:
                cross_layers.append(cross_layer.state_dict())

            for cls_encoder in model.featEmbedding.cls_encoders:
                cls_encoders.append(cls_encoder.state_dict())

            ck = {
                'epoch': epoch,
                'best_loss': best_loss,
                'best_acc': best_acc,
                'best_auc': best_auc,
                'optimizer': optimizer.state_dict(),
                'model_params': model.state_dict(),
                'cross_layers': cross_layers,
                'cls_encoders': cls_encoders
            }
            torch.save(ck, os.path.join(opt.ck_pth, 'model_' + model_name + '.pkl'))
    print("==================================================================================")
    print("Training End..")


def test(model, test_loader, opts):
    model.eval()
    # CNNDLGA.apply(freeze_bn)
    # CNNDLGA.localAttentionLayer.eval()
    print("==================================================================================")
    print("Testing Start..")
    for i, (x_num, x_cls, y) in enumerate(test_loader):
        # Convert torch tensor to Variable
        x_num = to_var(x_num)
        x_cls = to_var(x_cls)
        out, feat_scores, ws = model(x_cls, x_num)
        y_scores = F.softmax(out, dim=1)
        result = torch.argmax(y_scores, dim=1)

        if i == 0:
            res = result.data.cpu().numpy()
            y_scores = y_scores.data.cpu().numpy()
            y_ps = y_scores  # [:, 1]
            labels = y.numpy()
        else:
            res = np.append(res, result.data.cpu().numpy(), axis=0)
            y_scores = y_scores.data.cpu().numpy()
            labels = np.append(labels, y.numpy(), axis=0)
            y_ps = np.append(y_ps, y_scores, axis=0)

    # fpr, tpr, thresholds = roc_curve(labels, y_ps, pos_label=1, sample_weight=None, drop_intermediate=True)
    # AUC = auc(fpr, tpr)

    if opts.nclass == 20:
        res1 = res.copy()
        labels1 = labels.copy()
        res1[res <= 5] = 0
        res1[(res > 5) & (res < 13)] = 1
        res1[res >= 13] = 2
        labels1[labels <= 5] = 0
        labels1[(labels > 5) & (labels < 13)] = 1
        labels1[labels >= 13] = 2

        y_probs = np.zeros((y_ps.shape[0], 3))
        y_probs[:, 0] = np.sum(y_ps[:, :6], axis=1)
        y_probs[:, 1] = np.sum(y_ps[:, 6:13], axis=1)
        y_probs[:, 2] = np.sum(y_ps[:, 13:], axis=1)

        print(labels.shape, y_probs.shape)
        y_one_hot = label_binarize(y=list(labels1), classes=np.arange(3))
        AUC = roc_auc_score(y_one_hot, y_probs)
        acc = accuracy_score(res1, labels1)
    else:
        AUC = roc_auc_score(labels, y_ps[:, 1])
        acc = accuracy_score(res, labels)

    print('Auc=%.4f' % (AUC))
    print('Acc=%.4f' % (acc))
    '''
    fp, fn, pp, nn = 0, 0, 0, 0
    for i, p in enumerate(res):
        if p == 1 and labels[i] == 0:
            fp += 1
        if p == 0 and labels[i] == 1:
            fn += 1
        if p == 1:
            pp += 1
        else:
            nn += 1

    fp_rate = 1.0 * fp / (pp + 1e-9)
    fn_rate = 1.0 * fn / (nn + 1e-9)
    '''
    fp_rate = {}
    fp_count = {}
    label_counts(labels)
    for i, p in enumerate(res):
        # print(p,labels[i])
        if p not in fp_count.keys():
            fp_count[p] = 0
        if p == labels[i]:
            if p not in fp_rate.keys():
                fp_rate[p] = 0
            fp_rate[p] += 1
        fp_count[p] += 1

    for fp in fp_rate.keys():
        print('rating %d:%f' % (fp, fp_rate[fp] / fp_count[fp]))

    # print(fp_rate, fn_rate)
    print("==================================================================================")
    print("Testing End..")
    return acc, AUC  # , fp_rate  # , fn_rate


def l1_regularization(cross_layers, l1_alpha=0.01):
    for layer in cross_layers:
        layer.pfs.weight.grad.data.add_(l1_alpha * torch.sign(layer.pfs.weight.data))


def label_counts(labels):
    lab_count = {}
    ln = len(labels)
    for lb in labels:
        if lb not in lab_count.keys():
            lab_count[lb] = 1
        else:
            lab_count[lb] += 1
    for lb in lab_count.keys():
        lab_count[lb] = lab_count[lb] / ln
    print(lab_count)


if __name__ == '__main__':
    opts = opts_parser(parser.parse_args())
    print(opts, end='\n\n')
    log = logger()
    train(opts, log)
