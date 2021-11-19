import torch
from GAN import G, D, GanData,DCGAND,DCGANG
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import time
cuda = torch.cuda.is_available()

# kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}
# trainData = torch.utils.data.DataLoader(GanData(), batch_size=1, shuffle=True, **kwargs)

D = DCGAND()
G = DCGANG()

start_epoch = 298
D_path = r"./model/D" + str(start_epoch) + ".pt"
G_path = r"./model/G" + str(start_epoch) + ".pt"
D.load_state_dict(torch.load(D_path))
G.load_state_dict(torch.load(G_path))
if cuda:
    D.cuda()
    G.cuda()

if __name__ == '__main__':
    torch.manual_seed(int(time.time()))
    z = torch.rand(100, 100, 1, 1)
    fake_img = G(z.cuda()).detach()
    fake_out = D(fake_img.cuda())
    print(fake_out)
    array = np.uint8(((fake_img.cpu().numpy() + 1) / 2 * 255))
    array = array.transpose([0, 2, 3, 1])
    out = np.zeros((960,960,3))
    for i in range(10):
        for j in range(10):
            out[96 * i:96 * i + 96, 96 * j:96 * j + 96, ] = array[i * 10 + j]
    plt.imshow(Image.fromarray(np.uint8(out)))
    plt.show()
    Image.fromarray(np.uint8(out)).save("299.png")
    # for i, (real_img, _) in enumerate(trainData):
    #         real_out = D(real_img.cuda())
    #         print(real_out)
    #         array = np.uint8(((real_img.cpu().numpy() + 1) / 2 * 255))
    #         array = array.transpose([0, 2, 3, 1])
    #         plt.imshow(Image.fromarray(array[0]))
    #         plt.show()
    #         if input() == "=":
    #             break
