import torch
from .base_model import BaseModel
from . import networks
import time
from PIL import Image
import numpy as np

import sys
sys.path.append("..")
from data.base_dataset import get_params, get_transform


class BiCycleGAN4TZModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        if opt.isTrain:
            assert opt.batch_size % 2 == 0  # load two images at one time.

        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'D', 'G_GAN2', 'D2', 'G_L1', 'z_L1', 'kl']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A_encoded', 'real_B_encoded', 'fake_B_random', 'fake_B_encoded']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        use_D = opt.isTrain and opt.lambda_GAN > 0.0
        use_D2 = opt.isTrain and opt.lambda_GAN2 > 0.0 and not opt.use_same_D
        use_E = opt.isTrain or not opt.no_encode
        use_vae = True
        self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.nz, opt.ngf, netG=opt.netG,
                                      norm=opt.norm, nl=opt.nl, use_dropout=opt.use_dropout, init_type=opt.init_type, init_gain=opt.init_gain,
                                      gpu_ids=self.gpu_ids, where_add=opt.where_add, upsample=opt.upsample)
        D_output_nc = opt.input_nc + opt.output_nc if opt.conditional_D else opt.output_nc
        if use_D:
            self.model_names += ['D']
            self.netD = networks.define_D(D_output_nc, opt.ndf, netD=opt.netD, norm=opt.norm, nl=opt.nl,
                                          init_type=opt.init_type, init_gain=opt.init_gain, num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)
        if use_D2:
            self.model_names += ['D2']
            self.netD2 = networks.define_D(D_output_nc, opt.ndf, netD=opt.netD2, norm=opt.norm, nl=opt.nl,
                                           init_type=opt.init_type, init_gain=opt.init_gain, num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)
        else:
            self.netD2 = None
        if use_E:
            self.model_names += ['E']
            self.netE = networks.define_E(opt.output_nc, opt.nz, opt.nef, netE=opt.netE, norm=opt.norm, nl=opt.nl,
                                          init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids, vaeLike=use_vae)

        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(gan_mode=opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionZ = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            if use_E:
                self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_E)

            if use_D:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)
            if use_D2:
                self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D2)
        
        # fix magic number
        self.tri1 = torch.zeros((1, opt.nz, opt.load_size, opt.load_size)).to(self.device)
        self.tri2 = torch.zeros((1, opt.nz, opt.load_size, opt.load_size)).to(self.device)
        self.tri3 = torch.zeros((1, opt.nz, opt.load_size, opt.load_size)).to(self.device)
        self.tri4 = torch.zeros((1, opt.nz, opt.load_size, opt.load_size)).to(self.device)
        # print(one.view(1, 1, 1, 1).expand(1, opt.nz, 1, 1).size())
        # print(tri1.size())
        for ii in range(opt.nz):
            for i in range(opt.load_size):
                for j in range(opt.load_size):
                    ri = opt.load_size - i
                    # left
                    if j <= i and j <= ri:
                        self.tri2[0][ii][i][j] = 1.0
                        # count += 1
                    # bottom
                    elif j <= i and j >= ri:
                        self.tri4[0][ii][i][j] = 1.0
                    # top
                    elif j >= i and j <= ri:
                        self.tri1[0][ii][i][j] = 1.0
                    # right
                    elif j >= i and j >= ri:
                        self.tri3[0][ii][i][j] = 1.0
        # print(self.tri1)
        # print(self.tri2)
        # print(self.tri3)
        # print(self.tri4)
        # print(self.tri1 * 3)
        # print(self.tri1 + self.tri2 + self.tri3 + self.tri4)
        # print(self.tri1[0][0][255][512])


    # Add Image Positioning System
    # magic number need to be fixed
    # test z1
    # def Positioning(self, cen):
    #     # print(cen)
    #     # 12 * column
    #     # top
    #     if cen < 4:
    #         top = -1
    #     else:
    #         top = cen - 4
    #     # left
    #     if cen % 4 == 0:
    #         left = -1
    #     else:
    #         left = cen - 1
    #     # right
    #     if (cen + 1) % 4 == 0:
    #         right = -1
    #     else:
    #         right = cen + 1
    #     # bottom
    #     # (row * 12 - 1) * 12 * column
    #     if cen >= 12:
    #     # if cen >= 180:
    #         bottom = -1
    #     else:
    #         bottom = cen + 4
    
    #     connecteddArea = [top, left, cen, right, bottom]
    #     # print(connecteddArea)
    
    #     return connecteddArea

    def Positioning(self, cen):
        # print(cen)
        # 12 * column
        # top
        if cen < 60:
            top = -1
        else:
            top = cen - 60
        # left
        if cen % 60 == 0:
            left = -1
        else:
            left = cen - 1
        # right
        if (cen + 1) % 60 == 0:
            right = -1
        else:
            right = cen + 1
        # bottom
        # (row  - 1) * 12 * 12 * column
        if cen >= 3540:
        # if cen >= 180:
            bottom = -1
        else:
            bottom = cen + 60
    
        connecteddArea = [top, left, cen, right, bottom]
        # print(connecteddArea)
    
        return connecteddArea

    def PositioningV1(self, cen):
        # print(cen)
        # 12 * column
        # top
        if cen < 3660:
            top = -1
        else:
            top = cen - 60
        # left
        if cen % 60 == 0:
            left = -1
        else:
            left = cen - 1
        # right
        if (cen + 1) % 12 == 0:
            right = -1
        else:
            right = cen + 1
        # bottom
        # (row  - 1) * 12 * 12 * column
        if cen >= 4260:
        # if cen >= 180:
            bottom = -1
        else:
            bottom = cen + 60
    
        connecteddArea = [top, left, cen, right, bottom]
        # print(connecteddArea)
    
        return connecteddArea
    
    def getCZ(self, connecteddArea):
        CZ = torch.zeros((5, 8, self.real_B.size(2), self.real_B.size(3))).to(self.device).to(self.device)
        index = 0
        for c in connecteddArea:
            # print(self.image_paths)
            if c != -1:
                path = self.image_paths[0][0: self.image_paths[0].rfind('/') + 1] + str(c).zfill(5) + ".png"
                # print(path)
                AB = Image.open(path).convert('RGB')
                w, h = AB.size
                w2 = int(w / 2)
                B = AB.crop((w2, 0, w, h))
                transform_params = get_params(self.opt, B.size)
                B_transform = get_transform(self.opt, transform_params, grayscale=(self.opt.output_nc == 1))
                B = B_transform(B).to(self.device)
                # matrix = np.zeros((self.real_B.size()[0] - 1, self.real_B.size()[1], self.real_B.size()[2], self.real_B.size()[3]), dtype=np.float32)
                # B_encoded = torch.from_numpy(matrix).to(self.device)
                if self.is_train():
                    B_encoded = torch.zeros((self.real_B.size()[0] - 1, self.real_B.size()[1], self.real_B.size()[2], self.real_B.size()[3])).to(self.device)
                else:
                    B_encoded = torch.zeros((1, self.real_B.size()[1], self.real_B.size()[2], self.real_B.size()[3])).to(self.device)
                
                B_encoded[0:1, :, :, :] = B
                mu, logvar = self.netE.forward(B_encoded)
                std = logvar.mul(0.5).exp_()
                torch.manual_seed(c)
                eps = self.get_z_random(std.size(0), std.size(1))
                # print(eps)
                z_encoded = eps.mul(std).add_(mu)
                # print(B)
                CZ[index: index + 1, :, :, :] = z_encoded.view(z_encoded.size(0), z_encoded.size(1), 1, 1).expand(z_encoded.size(0), z_encoded.size(1), B_encoded.size(2), B_encoded.size(3))
            else:
                CZ[index: index + 1, :, :, :] = torch.zeros((1, 8)).view(1, 8, 1, 1).expand(1, 8, self.real_B.size(2), self.real_B.size(3)).to(self.device)
            index += 1

        return CZ
    
    def fill_z(self, z, zz):
        # print(zz[1])
        # print(z.shape)
        # print(zz.shape)
        # print(z[0][0][0][0])
        # newZ = np.zeros((1, 8, self.real_B.size(2), self.real_B.size(3)), dtype=np.float32)
        # tz = torch.from_numpy(newZ).to(self.device)
        tz =  torch.zeros((1, 8, self.real_B.size(2), self.real_B.size(3))).to(self.device)
        # for ii in range(8):
            # l = (z[0][ii][0][0] + zz[1][ii][0][0]) / 2
            # b = (z[0][ii][0][0] + zz[4][ii][0][0]) / 2
            # t = (z[0][ii][0][0] + zz[0][ii][0][0]) / 2
            # r = (z[0][ii][0][0] + zz[3][ii][0][0]) / 2
            # print(l)
            # print(b)
            # print(t)
            # print(r)

        zz = (z + zz) / 2
        # print(zz[1:2][:][:][:])

        top = self.tri1 * zz[0:1][:][:][:]
        left = self.tri2 * zz[1:2][:][:][:]
        # print(left)
        right = self.tri3 * zz[4:5][:][:][:]
        bottom = self.tri4 * zz[3:4][:][:][:]
        tz = top + left + right + bottom
        # print(tz)
        
        return tz


    def is_train(self):
        """check if the current batch is good for training."""
        return self.opt.isTrain and self.real_A.size(0) == self.opt.batch_size

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        index1 = self.image_paths[0].rfind('/')
        index2 = self.image_paths[0].rfind('.')
        # print(self.image_paths[0][index1+1: index2])
        self.image_number = int(self.image_paths[0][index1+1: index2])

    def get_z_random(self, batch_size, nz, random_type='gauss'):
        if random_type == 'uni':
            z = torch.rand(batch_size, nz) * 2.0 - 1.0
        elif random_type == 'gauss':
            z = torch.randn(batch_size, nz)
        return z.to(self.device)

    def encode(self, input_image):
        mu, logvar = self.netE.forward(input_image)
        std = logvar.mul(0.5).exp_()
        torch.manual_seed(self.image_number)
        eps = self.get_z_random(std.size(0), std.size(1))
        # print(eps)
        z = eps.mul(std).add_(mu)
        return z, mu, logvar

    # def test(self, z0=None, encode=False):
    #     with torch.no_grad():
    #         if encode:  # use encoded z
    #             z0, _ = self.netE(self.real_B)
    #         if z0 is None:
    #             z0 = self.get_z_random(self.real_A.size(0), self.opt.nz)
    #         # self.fake_B = self.netG(self.real_A, z0)
    #         self.fake_B = self.netG(self.real_A, z0.view(z0.size(0), z0.size(1), 1, 1).expand(z0.size(0), z0.size(1), self.real_A.size(2), self.real_A.size(3)))
    #         return self.real_A, self.fake_B, self.real_B

    def test(self, z0=None, encode=False):
        self.connecteddArea = self.PositioningV1(self.image_number)
        # print(self.connecteddArea)
        # print(self.connecteddArea)
        self.CZ = self.getCZ(self.connecteddArea)
        with torch.no_grad():
            if encode:  # use encoded z
                z0, _ = self.netE(self.real_B)
                # print(z0.size())
                z0 = z0.view(z0.size(0), z0.size(1), 1, 1).expand(z0.size(0), z0.size(1), self.real_A.size(2), self.real_A.size(3))
            else:
                torch.manual_seed(self.image_number)
                z0 = self.get_z_random(self.real_A.size(0), self.opt.nz)
                z0 = z0.view(z0.size(0), z0.size(1), 1, 1).expand(z0.size(0), z0.size(1), self.real_A.size(2), self.real_A.size(3))
                z0 = self.fill_z(z0, self.CZ)
                # print(z0.shape)
            # self.fake_B = self.netG(self.real_A, z0)
            self.fake_B = self.netG(self.real_A, z0)
            return self.real_A, self.fake_B, self.real_B

    def forward(self):
        # get real images
        half_size = self.opt.batch_size // 2
        # A1, B1 for encoded; A2, B2 for random
        self.real_A_encoded = self.real_A[0:half_size]
        self.real_B_encoded = self.real_B[0:half_size]
        self.real_B_random = self.real_B[half_size:]
        # get encoded z
        self.z_encoded, self.mu, self.logvar = self.encode(self.real_B_encoded)
        # get random z
        ticks = time.time()
        torch.manual_seed(ticks)
        self.z_random = self.get_z_random(self.real_A_encoded.size(0), self.opt.nz)
        # generate fake_B_encoded
        # print(self.z_encoded)
        self.z_encoded = self.z_encoded.view(self.z_encoded.size(0), self.z_encoded.size(1), 1, 1).expand(self.z_encoded.size(0), self.z_encoded.size(1), self.real_A_encoded.size(2), self.real_A_encoded.size(3))
        # print(self.z_encoded)
        # print('-----------------')
        # print(self.CZ)
        # print('-----------------')
        self.z_encoded = self.fill_z(self.z_encoded, self.CZ)
        # print(self.z_encoded)
        # print('-----------------')
        self.fake_B_encoded = self.netG(self.real_A_encoded, self.z_encoded)
        # generate fake_B_random
        self.fake_B_random = self.netG(self.real_A_encoded, self.z_random.view(self.z_random.size(0), self.z_random.size(1), 1, 1).expand(self.z_random.size(0), self.z_random.size(1), self.real_A_encoded.size(2), self.real_A_encoded.size(3)))
        if self.opt.conditional_D:   # tedious conditoinal data
            self.fake_data_encoded = torch.cat([self.real_A_encoded, self.fake_B_encoded], 1)
            self.real_data_encoded = torch.cat([self.real_A_encoded, self.real_B_encoded], 1)
            self.fake_data_random = torch.cat([self.real_A_encoded, self.fake_B_random], 1)
            self.real_data_random = torch.cat([self.real_A[half_size:], self.real_B_random], 1)
        else:
            self.fake_data_encoded = self.fake_B_encoded
            self.fake_data_random = self.fake_B_random
            self.real_data_encoded = self.real_B_encoded
            self.real_data_random = self.real_B_random

        # compute z_predict
        if self.opt.lambda_z > 0.0:
            self.mu2, logvar2 = self.netE(self.fake_B_random)  # mu2 is a point estimate

    def backward_D(self, netD, real, fake):
        # Fake, stop backprop to the generator by detaching fake_B
        pred_fake = netD(fake.detach())
        # real
        pred_real = netD(real)
        loss_D_fake, _ = self.criterionGAN(pred_fake, False)
        loss_D_real, _ = self.criterionGAN(pred_real, True)
        # Combined loss
        loss_D = loss_D_fake + loss_D_real
        loss_D.backward()
        return loss_D, [loss_D_fake, loss_D_real]

    def backward_G_GAN(self, fake, netD=None, ll=0.0):
        if ll > 0.0:
            pred_fake = netD(fake)
            loss_G_GAN, _ = self.criterionGAN(pred_fake, True)
        else:
            loss_G_GAN = 0
        return loss_G_GAN * ll

    def backward_EG(self):
        # 1, G(A) should fool D
        self.loss_G_GAN = self.backward_G_GAN(self.fake_data_encoded, self.netD, self.opt.lambda_GAN)
        if self.opt.use_same_D:
            self.loss_G_GAN2 = self.backward_G_GAN(self.fake_data_random, self.netD, self.opt.lambda_GAN2)
        else:
            self.loss_G_GAN2 = self.backward_G_GAN(self.fake_data_random, self.netD2, self.opt.lambda_GAN2)
        # 2. KL loss
        if self.opt.lambda_kl > 0.0:
            self.loss_kl = torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp()) * (-0.5 * self.opt.lambda_kl)
        else:
            self.loss_kl = 0
        # 3, reconstruction |fake_B-real_B|
        if self.opt.lambda_L1 > 0.0:
            self.loss_G_L1 = self.criterionL1(self.fake_B_encoded, self.real_B_encoded) * self.opt.lambda_L1
        else:
            self.loss_G_L1 = 0.0

        self.loss_G = self.loss_G_GAN + self.loss_G_GAN2 + self.loss_G_L1 + self.loss_kl
        self.loss_G.backward(retain_graph=True)

    def update_D(self):
        self.set_requires_grad([self.netD, self.netD2], True)
        # update D1
        if self.opt.lambda_GAN > 0.0:
            self.optimizer_D.zero_grad()
            self.loss_D, self.losses_D = self.backward_D(self.netD, self.real_data_encoded, self.fake_data_encoded)
            if self.opt.use_same_D:
                self.loss_D2, self.losses_D2 = self.backward_D(self.netD, self.real_data_random, self.fake_data_random)
            self.optimizer_D.step()

        if self.opt.lambda_GAN2 > 0.0 and not self.opt.use_same_D:
            self.optimizer_D2.zero_grad()
            self.loss_D2, self.losses_D2 = self.backward_D(self.netD2, self.real_data_random, self.fake_data_random)
            self.optimizer_D2.step()

    def backward_G_alone(self):
        # 3, reconstruction |(E(G(A, z_random)))-z_random|
        if self.opt.lambda_z > 0.0:
            self.loss_z_L1 = torch.mean(torch.abs(self.mu2 - self.z_random)) * self.opt.lambda_z
            self.loss_z_L1.backward()
        else:
            self.loss_z_L1 = 0.0

    def update_G_and_E(self):
        # update G and E
        self.set_requires_grad([self.netD, self.netD2], False)
        self.optimizer_E.zero_grad()
        self.optimizer_G.zero_grad()
        self.backward_EG()
        self.optimizer_G.step()
        self.optimizer_E.step()
        # update G only
        if self.opt.lambda_z > 0.0:
            self.optimizer_G.zero_grad()
            self.optimizer_E.zero_grad()
            self.backward_G_alone()
            self.optimizer_G.step()

    def optimize_parameters(self):
        self.connecteddArea = self.Positioning(self.image_number)
        # print(self.connecteddArea)
        self.CZ = self.getCZ(self.connecteddArea)
        # print(self.CZ)
        self.forward()
        self.update_G_and_E()
        self.update_D()
        # print("------optimize once--------")