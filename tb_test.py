import os

from torch.utils.tensorboard import SummaryWriter

os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

writer = SummaryWriter("tb_logs")
x = range(100)
for i in x:
    writer.add_scalar('y=2x', i * 2, i)
writer.close()