import torch
import torch.nn as nn

from GAN import G, D, GanData, DCGAND, DCGANG
from tqdm import tqdm

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

cuda = torch.cuda.is_available()

# torch.manual_seed(1337)
# if cuda:
#     torch.cuda.manual_seed(1337)
D = DCGAND()
G = DCGANG()

if cuda:
    D.cuda()
    G.cuda()

criterion = nn.BCELoss()  # 是单目标二分类交叉熵函数
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
dlr_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size=1, gamma=0.92)
glr_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=1, gamma=0.92)

batch_size = 256
kwargs = {'num_workers': 3, 'pin_memory': True} if cuda else {}
trainData = torch.utils.data.DataLoader(GanData(), batch_size=batch_size, shuffle=True, **kwargs)

real_label = (torch.ones((batch_size, 1)) * 0.99).cuda().cuda()
fake_label = (torch.ones((batch_size, 1)) * 0.01).cuda().cuda()

resume = True
show = False
save = True
start_epoch = 49
num_epoch = 300
rootPath = "./"
D_path = rootPath + "model/D" + str(start_epoch) + ".pt"
G_path = rootPath + "model/G" + str(start_epoch) + ".pt"

if __name__ == '__main__':
    if resume:
        D.load_state_dict(torch.load(D_path))
        G.load_state_dict(torch.load(G_path))

    g_trainTimes = 5
    for epoch in range(start_epoch + 1, num_epoch):
        lossD = []
        lossG = []
        _tqdm = tqdm(trainData)
        for i, (real_img, _) in enumerate(_tqdm):
            batch_size = len(real_img)
            # 判别器D训练 真图片label 1，假图片 0
            # 真图片损失
            real_label = (torch.ones((batch_size, 1)) * 0.99).cuda()
            fake_label = (torch.ones((batch_size, 1)) * 0.01).cuda()

            real_out = D(real_img.cuda())
            d_loss_real = criterion(real_out, real_label)
            # 假图片损失
            z = torch.rand(batch_size, 100, 1, 1)
            fake_img = G(z.cuda()).detach()
            fake_out = D(fake_img)
            d_loss_fake = criterion(fake_out, fake_label)
            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # 训练生成器G
            for t in range(g_trainTimes):
                z = torch.rand(batch_size, 100, 1, 1)
                fake_img = G(z.cuda())
                output = D(fake_img.cuda())
                g_loss = criterion(output, real_label)  # 得到的假的图片与真实的图片的label的loss
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

            # 显示生成图片
            if fake_out.data.mean() > 0.7:
                if show:
                    array = np.uint8(((fake_img.cpu().detach().numpy() + 1) / 2 * 255))
                    array = array.transpose([0, 2, 3, 1])
                    plt.imshow(Image.fromarray(array[fake_out.data.argmax()]))  # 最大fake_out下标
                    plt.show()
                if save:
                    array = np.uint8(((fake_img.cpu().detach().numpy() + 1) / 2 * 255))
                    array = array.transpose([0, 2, 3, 1])
                    Image.fromarray(array[fake_out.data.argmax()]).save(
                        rootPath + "anime_out/" + str(epoch) + "_" + str(i) + ".jpg")
            # if (i + 1) % 10000 == 0:
            #     dlr_scheduler.step()  # 降低学习率
            #     glr_scheduler.step()  # 降低学习率

            # 打印中间的损失
            _tqdm.set_description(
                'Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} D real: {:.6f},D fake: {:.6f}'.format(epoch, num_epoch,
                                                                                                d_loss.data.item(),
                                                                                                g_loss.data.item(),
                                                                                                real_out.data.mean(),
                                                                                                fake_out.data.mean()))
            lossD.append(d_loss.data.item())
            lossG.append(g_loss.data.item())
            # if fake_out.data.mean() < 0.5:
            #     g_trainTimes = 5
            # if fake_out.data.mean() > 0.8:
            #     g_trainTimes = 1
        # 保存模型参数
        torch.save(D.state_dict(), rootPath+"model/D" + str(epoch) + ".pt")
        torch.save(G.state_dict(), rootPath+"model/G" + str(epoch) + ".pt")
        # 绘制loss
        plt.subplot(211)
        plt.plot(lossD)
        #plt.title("D_loss")
        plt.subplot(212)
        plt.plot(lossG)
        #plt.title("G_loss")
        plt.show()