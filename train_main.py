#author: akshitac8
#tf-vaegan inductive
from __future__ import print_function

import scipy.io as sio
import random
import numpy as np

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
#import functions
import networks.CGRL_model as model
import datasets.util as util
import classifiers.classifier_pointclouds as classifier
from config import opt

import losses

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
# load data
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)


netE = model.Encoder(opt)
netG = model.Generator(opt)
netD = model.Discriminator_D1(opt)
# Init models: Feedback module, auxillary module
netF = model.Feedback(opt)
netCla = model.AttClass(opt)
# Init Contrasive Learning Network
netMap = model.Embedding_Net(opt)


print(netE)
print(netG)
print(netD)
print(netF)
print(netCla)
print(netMap)
# print(opt.feedback_loop)

###########
# Init Tensors
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize) #attSize class-embedding size
noise = torch.FloatTensor(opt.batch_size, opt.nz)
one = torch.FloatTensor([1])
mone = one * -1

input_label = torch.LongTensor(opt.batch_size)
contras_criterion = losses.SupConLoss_clear(opt.ins_temp)

##########
# Cuda
if opt.cuda:
    netD.cuda()
    netE.cuda()
    netF.cuda()
    netG.cuda()
    netCla.cuda()
    netMap.cuda()


    input_label = input_label.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    one = one.cuda()
    mone = mone.cuda()

def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x+1e-12, x.detach(),size_average=False)
    BCE = BCE.sum()/ x.size(0)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())/ x.size(0)
    return (BCE + KLD)
           
def sample():
    batch_feature, batch_label, batch_att = data.next_seen_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))

def WeightedL1(pred, gt):
    wt = (pred-gt).pow(2)
    wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0),wt.size(1))
    loss = wt * (pred-gt).abs()
    return loss.sum()/loss.size(0)

def generate_syn_feature(generator,classes, attribute,num,netF=None,netCla=None):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num) 
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        syn_noisev = Variable(syn_noise,volatile=True)
        syn_attv = Variable(syn_att,volatile=True)
        fake = generator(syn_noisev,c=syn_attv)
        if netF is not None:
            Map_out = netMap(fake) # only to call the forward function of decoder
            embed_hidden_feat = netMap.getLayersOutEmbed() #no detach layers
            feedback_out = netF(embed_hidden_feat)
            fake = generator(syn_noisev, a1=opt.a2, c=syn_attv, feedback_layers=feedback_out)
        output = fake
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)

    return syn_feature, syn_label

# setup optimizer
import itertools
optimizer          = optim.Adam(netE.parameters(), lr=opt.lr)
# optimizerD         = optim.Adam(netD.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
optimizerD = optim.Adam(itertools.chain(netD.parameters(), netMap.parameters()), lr=opt.lr,
                        betas=(opt.beta1, 0.999))
optimizerG         = optim.Adam(netG.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
optimizerF         = optim.Adam(netF.parameters(), lr=opt.feed_lr, betas=(opt.beta1, 0.999))
optimizerCla       = optim.Adam(netCla.parameters(), lr=opt.dec_lr, betas=(opt.beta1, 0.999))


def calc_gradient_penalty(netD,real_data, fake_data, input_att):
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates, Variable(input_att))
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty

best_gzsl_acc = 0
best_zsl_acc = 0
for epoch in range(0,opt.nepoch):
    for i in range(0, data.ntrain, opt.batch_size):
        for loop in range(0,opt.feedback_loop):
            #########Discriminator training ##############
            for p in netD.parameters(): #unfreeze discrimator
                p.requires_grad = True

            for p in netCla.parameters(): #unfreeze deocder
                p.requires_grad = True
        #hy################
            for p in netMap.parameters():  # reset requires_grad
                p.requires_grad = True
        #hy###############
            # Train D1 and Decoder (and Decoder Discriminator)
            gp_sum = 0 #lAMBDA VARIABLE
            for iter_d in range(opt.critic_iter):
                sample()
                netD.zero_grad()    
                netMap.zero_grad()      
                input_resv = Variable(input_res)
                input_attv = Variable(input_att)


                netCla.zero_grad()
                loss = nn.CrossEntropyLoss()
                input = netCla(input_resv)
                target = autograd.Variable(input_label)
                C_cost = opt.cls_weight * loss(input, target)
                C_cost.backward(retain_graph=True)

                optimizerCla.step()

                outz_real = netMap(input_resv)
                real_ins_contras_loss = contras_criterion(outz_real, input_label)

                
                criticD_real = netD(input_resv, input_attv)
                criticD_real = opt.gammaD*criticD_real.mean()
                criticD_real.backward(mone)
                if opt.encoded_noise:        
                    means, log_var = netE(input_resv, input_attv)
                    std = torch.exp(0.5 * log_var)
                    eps = torch.randn([opt.batch_size, opt.latent_size]).cpu()
                    eps = Variable(eps.cuda())
                    z = eps * std + means #torch.Size([64, 312])
                else:
                    noise.normal_(0, 1)
                    z = Variable(noise)
     
                if loop == 1 or loop == 2:
                # if loop == 1:
                    fake = netG(z, c=input_attv)
                    Map_out = netMap(fake)
                    embed_hidden_feat = netMap.getLayersOutEmbed() #no detach layers
                    feedback_out = netF(embed_hidden_feat)
                    fake = netG(z, a1=opt.a1, c=input_attv, feedback_layers=feedback_out)
                else:
                    fake = netG(z, c=input_attv)
                fake = netG(z, c=input_attv)

                outz_fake = netMap(fake)
                fake_ins_contras_loss = contras_criterion(outz_fake, input_label)

                criticD_fake = netD(fake.detach(), input_attv)
                criticD_fake = opt.gammaD*criticD_fake.mean()
                criticD_fake.backward(one)
                # gradient penalty
                gradient_penalty = opt.gammaD*calc_gradient_penalty(netD, input_res, fake.data, input_att)
                # if opt.lambda_mult == 1.1:
                gp_sum += gradient_penalty.data
                gradient_penalty.backward()         
                Wasserstein_D = criticD_real - criticD_fake

                D_cost = criticD_fake - criticD_real + gradient_penalty + real_ins_contras_loss + fake_ins_contras_loss#add Y here and #add vae reconstruction loss
                optimizerD.step()

            gp_sum /= (opt.gammaD*opt.lambda1*opt.critic_iter)
            if (gp_sum > 1.05).sum() > 0:
                opt.lambda1 *= 1.1
            elif (gp_sum < 1.001).sum() > 0:
                opt.lambda1 /= 1.1

            #############Generator training ##############
            # Train Generator and Decoder
            for p in netD.parameters(): #freeze discrimator
                p.requires_grad = False
            if opt.recons_weight > 0 and opt.freeze_dec:
                for p in netCla.parameters(): #freeze decoder
                    p.requires_grad = False

            netE.zero_grad()
            netG.zero_grad()
            netF.zero_grad()
            input_resv = Variable(input_res)
            input_attv = Variable(input_att)
            means, log_var = netE(input_resv, input_attv)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn([opt.batch_size, opt.latent_size]).cpu()
            eps = Variable(eps.cuda())
            z = eps * std + means #torch.Size([64, 312])

            if loop == 1 or loop == 2:
            # if loop == 1:
                recon_x = netG(z, c=input_attv)
                Map_out = netMap(recon_x)
                embed_hidden_feat = netMap.getLayersOutEmbed() #no detach layers
                feedback_out = netF(embed_hidden_feat)
                recon_x = netG(z, a1=opt.a1, c=input_attv, feedback_layers=feedback_out)
            else:
                recon_x = netG(z, c=input_attv)
            recon_x = netG(z, c=input_attv)
            ###hy
            outz_fake = netMap(recon_x)
            ###hy
            vae_loss_seen = loss_fn(recon_x, input_resv, means, log_var) # minimize E 3 with this setting feedback will update the loss as well
            errG = vae_loss_seen
            
            if opt.encoded_noise:
                criticG_fake = netD(recon_x,input_attv).mean()
                fake = recon_x 
            else:
                noise.normal_(0, 1)
                noisev = Variable(noise)
     
                if loop == 1 or loop == 2:
                # if loop == 1:
                    fake = netG(noisev, c=input_attv)
                    Map_out = netMap(recon_x) #Feedback from Decoder encoded output
                    embed_hidden_feat = netMap.getLayersOutEmbed() #no detach layers
                    feedback_out = netF(embed_hidden_feat)
                    fake = netG(noisev, a1=opt.a1, c=input_attv, feedback_layers=feedback_out)
                else:
                    fake = netG(noisev, c=input_attv)
                fake = netG(noisev, c=input_attv)
                criticG_fake = netD(fake,input_attv).mean()
                

            G_cost = -criticG_fake

            outz_real = netMap(input_resv)
            all_outz = torch.cat((outz_fake, outz_real.detach()), dim=0)
            fake_real_contras_loss = contras_criterion(all_outz, torch.cat((input_label, input_label), dim=0))
            


            errG += opt.gammaG*G_cost
            netCla.zero_grad()
            loss = nn.CrossEntropyLoss()
            input_fake = netCla(fake)
            target = autograd.Variable(input_label)
            C_cost = opt.cls_weight * loss(input, target)


            errG += opt.cls_weight * C_cost + opt.ins_weight * fake_real_contras_loss 
            errG.backward()
            # write a condition here
            optimizer.step()
            optimizerG.step()
            # if loop == 1:
            #     optimizerF.step()
            optimizerCla.step() 
        
    print('[%d/%d]  Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist:%.4f, vae_loss_seen:%.4f'% (epoch, opt.nepoch, D_cost.data[0], G_cost.data[0], Wasserstein_D.data[0],vae_loss_seen.data[0]),end=" ")
    netG.eval()
    netCla.eval()
    netF.eval()
    syn_feature, syn_label = generate_syn_feature(netG,data.unseenclasses, data.attribute, opt.syn_num,netF=netF,netCla=netCla)
    # syn_feature, syn_label = generate_syn_feature(netG,data.unseenclasses, data.attribute, opt.syn_num,netF=None,netCla=netCla)


    # Generalized zero-shot learning
    if opt.gzsl:   
        # Concatenate real seen features with synthesized unseen features
        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)
        nclass = opt.nclass_all
        # Train GZSL classifier
        gzsl_cls = classifier.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, \
                25, opt.syn_num, generalized=True, netMap =netMap, dec_size=512, dec_hidden_size=2048)
        if best_gzsl_acc < gzsl_cls.H:
            best_acc_seen, best_acc_unseen, best_gzsl_acc = gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H
        print('GZSL: seen=%.4f, unseen=%.4f, h=%.4f' % (gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H),end=" ")

    # Zero-shot learning
    # Train ZSL classifier
    zsl_cls = classifier.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), \
                    data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, \
                    generalized=False, netMap = netMap, dec_size=512, dec_hidden_size=2048)

    acc = zsl_cls.acc
    if best_zsl_acc < acc:
        best_zsl_acc = acc
    print('ZSL: unseen accuracy=%.4f' % (acc))
    # reset G to training mode
    netG.train()
    netCla.train()
    netF.train()

print('Dataset', opt.dataset)
print('the best ZSL unseen accuracy is', best_zsl_acc)
if opt.gzsl:
    print('Dataset', opt.dataset)
    print('the best GZSL seen accuracy is', best_acc_seen)
    print('the best GZSL unseen accuracy is', best_acc_unseen)
    print('the best GZSL H is', best_gzsl_acc)
    print(opt.point_embedding)



