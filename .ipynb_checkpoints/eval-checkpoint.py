from __future__ import print_function


import argparse
import os
import torch
from model.resnet import resnet34, resnet50
from torch.autograd import Variable
from tqdm import tqdm
from model.basenet import AlexNetBase, VGGBase, Predictor, Predictor_deep
from utils.return_dataset import return_dataset_test

# Training settings
parser = argparse.ArgumentParser(description='Visda Classification')
parser.add_argument('--T', type=float, default=0.05, metavar='T',
                    help='temperature (default: 0.05)')
parser.add_argument('--step', type=int, default=1000, metavar='step',
                    help='loading step')
parser.add_argument('--checkpath', type=str, default='./save_model_ssda',
                    help='dir to save checkpoint')
parser.add_argument('--method', type=str, default='MME',
                    choices=['S+T', 'ENT', 'MME'],
                    help='MME is proposed method, ENT is entropy minimization,'
                         'S+T is training only on labeled examples')
parser.add_argument('--output', type=str, default='./output.txt',
                    help='path to store result file')
parser.add_argument('--net', type=str, default='resnet34', metavar='B',
                    help='which network ')
parser.add_argument('--source', type=str, default='Strain', metavar='B',
                    help='board dir')
parser.add_argument('--target', type=str, default='Ttrain', metavar='B',
                    help='board dir')
parser.add_argument('--target_test', type=str, default='Ttest', metavar='B',
                    help='board dir')
parser.add_argument('--dataset', type=str, default='multi_all',
                    choices=['multi_all', 'toy_1'],
                    help='the name of dataset, multi is large scale dataset')
parser.add_argument('--num', type=int, default=3, help='Number of labeled target data')
args = parser.parse_args()
print('dataset %s source %s target %s network %s' %
      (args.dataset, args.source, args.target, args.net))
target_loader_unl, class_list = return_dataset_test(args)
use_gpu = torch.cuda.is_available()
if args.net == 'resnet34':
    G = resnet34()
    inc = 512
elif args.net == 'resnet50':
    G = resnet50()
    inc = 2048
elif args.net == "alexnet":
    G = AlexNetBase()
    inc = 4096
elif args.net == "vgg":
    G = VGGBase()
    inc = 4096
else:
    raise ValueError('Model cannot be recognized.')


if "resnet" in args.net:
    F1 = Predictor_deep(num_class=len(class_list),
                        inc=inc)
else:
    F1 = Predictor(num_class=len(class_list), inc=inc, cosine=True, temp=args.T)
G.cuda()
F1.cuda()
G.load_state_dict(torch.load(os.path.join(args.checkpath,
                                          "G_iter_model_{}_{}_"
                                          "to_{}_step_{}.pth.tar".
                                          format(args.method, args.source,
                                                 args.target, args.step))))
F1.load_state_dict(torch.load(os.path.join(args.checkpath,
                                           "F1_iter_model_{}_{}_"
                                           "to_{}_step_{}.pth.tar".
                                           format(args.method, args.source,
                                                  args.target, args.step))))

im_data_t = torch.FloatTensor(1)
gt_labels_t = torch.LongTensor(1)

im_data_t = im_data_t.cuda()
gt_labels_t = gt_labels_t.cuda()

im_data_t = Variable(im_data_t)
gt_labels_t = Variable(gt_labels_t)
if os.path.exists(args.checkpath) == False:
    os.mkdir(args.checkpath)


def eval(loader, output_file="output.txt"):
    G.eval()
    F1.eval()
    size = 0
    with open(output_file, "w") as f:
        with torch.no_grad():
            for batch_idx, data_t in tqdm(enumerate(loader)):
                im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
                gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])
                paths = data_t[2]
                feat = G(im_data_t)
                output1 = F1(feat)
                size += im_data_t.size(0)
                pred1 = output1.data.max(1)[1]
                for i, path in enumerate(paths):
                    f.write("%s %d\n" % (path, pred1[i]))


def test(loader):
    G.eval()
    F1.eval()
    test_loss = 0
    correct = 0
    size = 0
    num_class = len(class_list)
    output_all = np.zeros((0, num_class))
    criterion = nn.CrossEntropyLoss().cuda()
    confusion_matrix = np.zeros((num_class, num_class))
    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
            gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])
            feat = G(im_data_t)
            output1 = F1(feat)
            output_all = np.r_[output_all, output1.data.cpu().numpy()]
            size += im_data_t.size(0)
            pred1 = output1.data.max(1)[1]
            for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(gt_labels_t.data).cpu().sum()
            test_loss += criterion(output1, gt_labels_t) / len(loader)
    print('Test set: Average loss: {:.4f}, '
          'Accuracy: {}/{} F1 ({:.0f}%)'.
          format(test_loss, correct, size,
                 100. * correct / size))
    print('Confusion matrix:')
    print(confusion_matrix.astype('int'))
    return test_loss.data, 100. * float(correct) / size


# +
#eval(target_loader_unl, output_file="%s_%s_%s.txt" % (args.method, args.net, args.step))
# -

test(target_loader_unl)
