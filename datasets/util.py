#import h5py
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i    

    return mapped_label

class DATA_LOADER(object):  
    def __init__(self, opt):
        self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

    def read_matdataset(self, opt):

        if opt.dataset == 'ModelNet40':
            seen_set_index = [0,3,4,5,6,7,9,10,11,13,15,16,17,18,19,20,21,24,25,26,27,28,29,31,32,34,36,37,38,39]
            train_labels = np.int16(seen_set_index)
            unseen_set_index = [1,2,8,12,14,22,23,30,33,35]
            unseen_labels =np.int16(unseen_set_index)

            matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.point_embedding + "/"  + "seen_train.mat")
            train_feature = matcontent['data']
            matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.point_embedding + "/"  + "seen_train_label.mat")
            train_label = matcontent['label'].squeeze() #label 索引
            train_label = train_labels[train_label].squeeze() #导入真正label

            matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.point_embedding + "/"  + "seen_test.mat")
            test_feature = matcontent['data']
            matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.point_embedding + "/"  + "seen_test_label.mat")
            test_label = matcontent['label'].squeeze() #label 索引
            test_label = train_labels[test_label].squeeze() #导入真正label

            matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.point_embedding + "/"  + "unseen_" + opt.dataset + ".mat")
            unseen_feature = matcontent['data']
            matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.point_embedding + "/"  + "unseen_" + opt.dataset + "_label.mat")
            unseen_label = matcontent['label'].squeeze() #label 索引
            unseen_label = unseen_labels[unseen_label].squeeze() #导入真正label #导入真正label

            matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".mat")
            self.attribute = torch.from_numpy(matcontent['word']).float()
            self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),self.attribute.size(1))
        
        if opt.dataset == 'McGill':
            seen_set_index = [0,3,4,5,6,7,9,10,11,13,15,16,17,18,19,20,21,24,25,26,27,28,29,31,32,34,36,37,38,39]
            unseen_set_index = [1,2,8,12,14,22,23,30,33,35]

            matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.point_embedding + "/"  + "seen_train.mat")
            train_feature = matcontent['data']
            matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.point_embedding + "/"  + "seen_train_label.mat")
            train_label = matcontent['label'].squeeze() 

            matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.point_embedding + "/"  + "seen_test.mat")
            test_feature = matcontent['data']
            matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.point_embedding + "/"  + "seen_test_label.mat")
            test_label = matcontent['label'].squeeze() 

            matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.point_embedding + "/"  + "unseen_" + opt.dataset + ".mat")
            unseen_feature = matcontent['data']
            matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.point_embedding + "/"  + "unseen_" + opt.dataset + "_label.mat")
            unseen_label = matcontent['label'].squeeze()
            unseen_label = unseen_label + 30

            matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".mat")
            self.attribute = torch.from_numpy(matcontent['word']).float()
            self.attribute_seen = self.attribute[seen_set_index]
            matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.dataset + "_w2v.mat")
            self.attribute = torch.from_numpy(matcontent['word']).float()
            self.attribute = torch.cat((self.attribute_seen,self.attribute), dim = 0)
            self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),self.attribute.size(1))

        if opt.dataset == 'ScanObjectNN':
            seen_set_index = [0,1,5,6,7,9,10,11,15,16,17,18,19,20,21,23,24,25,26,27,28,31,34,36,37,39]
            unseen_set_index = [3,4,5,6,7,8,9,10,12,13,14]

            matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.point_embedding + "/"  + "seen_train.mat")
            train_feature = matcontent['data'][0:4999]
            matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.point_embedding + "/"  + "seen_train_label.mat")
            train_label = matcontent['label'].squeeze() 

            matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.point_embedding + "/"  + "seen_test.mat")
            test_feature = matcontent['data']
            matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.point_embedding + "/"  + "seen_test_label.mat")
            test_label = matcontent['label'].squeeze() 

            matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.point_embedding + "/"  + "unseen_" + opt.dataset + ".mat")
            unseen_feature = matcontent['data']
            matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.point_embedding + "/"  + "unseen_" + opt.dataset + "_label.mat")
            unseen_label = matcontent['label'].squeeze()
            unseen_label = unseen_label + 26

            matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".mat")
            self.attribute = torch.from_numpy(matcontent['word']).float()
            self.attribute_seen = self.attribute[seen_set_index]
            matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.dataset + "_w2v.mat")
            self.attribute = torch.from_numpy(matcontent['word']).float()
            self.attribute_unseen = self.attribute[unseen_set_index]
            self.attribute = torch.cat((self.attribute_seen,self.attribute_unseen), dim = 0)
            self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),self.attribute.size(1))



        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()
                
                _train_feature = scaler.fit_transform(train_feature)
                _test_seen_feature = scaler.transform(test_feature)
                _test_unseen_feature = scaler.transform(unseen_feature)
                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1/mx)
                self.train_label = torch.from_numpy(train_label).long() 
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1/mx)
                self.test_unseen_label = torch.from_numpy(unseen_label).long() 
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float() 
                self.test_seen_feature.mul_(1/mx)
                self.test_seen_label = torch.from_numpy(test_label).long()

            else:
                self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
                self.train_label = torch.from_numpy(label[trainval_loc]).long() 
                self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
                self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float() 
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
        else:
            self.train_feature = torch.from_numpy(feature[train_loc]).float()
            self.train_label = torch.from_numpy(label[train_loc]).long()
            self.test_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long() 

    
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))


        self.ntrain = self.train_feature.size()[0]
        self.ntest_seen = self.test_seen_feature.size()[0]
        self.ntest_unseen = self.test_unseen_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()
        self.train_mapped_label = map_label(self.train_label, self.seenclasses) 

    def next_seen_batch(self, seen_batch):
        idx = torch.randperm(self.ntrain)[0:seen_batch]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att
