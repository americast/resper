from model import *
from utils import *
from evaluate import *
import pdb
import os
import torch

def load_data():
    bxc = [] # character sequence batch
    bxw = [] # word sequence batch
    by = [] # label batch
    data = []
    cti = load_tkn_to_idx(sys.argv[2]) # char_to_idx
    wti = load_tkn_to_idx(sys.argv[3]) # word_to_idx
    itt = load_idx_to_tkn(sys.argv[4]) # idx_to_tkn
    print("loading %s..." % sys.argv[5])
    fo = open(sys.argv[5], "r")
    for line in fo:
        line = line.strip()
        *x, y = [x.split(":") for x in line.split(" ")]
        #breakpoint()
        xc, xw = zip(*[(list(map(int, xc.split("+"))), int(xw)) for xc, xw in x])

        bxc.append(xc)
        bxw.append(xw)
        by.append(int(y[0]))
        if len(by) == BATCH_SIZE:
            bxc, bxw = batchify(bxc, bxw, True, True, max(KERNEL_SIZES))
            data.append((bxc, bxw, LongTensor(by)))
            bxc = []
            bxw = []
            by = []
    fo.close()
    print("data size: %d" % (len(data) * BATCH_SIZE))
    print("batch size: %d" % BATCH_SIZE)
    return data, cti, wti, itt

def train():
    best_f1 = 0
    num_epochs = int(sys.argv[-2])
    data, cti, wti, itt = load_data()
    model = cnn(len(cti), len(wti), len(itt))
    optim = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    print(model)
    epoch = load_checkpoint(sys.argv[1], model) if isfile(sys.argv[1]) else 0
    filename = re.sub("\.epoch[0-9]+$", "", sys.argv[1])
    print("training model...")
    for ei in range(epoch + 1, epoch + num_epochs + 1):
        loss_sum = 0
        timer = time()
        for xc, xw, y in data:
            model.zero_grad()
            loss = F.nll_loss(model(xc, xw), y) # forward pass and compute loss
            loss.backward() # compute gradients
            optim.step() # update parameters
            loss_sum += loss.item()
        timer = time() - timer
        loss_sum /= len(data)
        # if ei % SAVE_EVERY and ei != epoch + num_epochs:
        #     save_checkpoint("", None, ei, loss_sum, timer)
        # else:
        #     save_checkpoint(filename, model, ei, loss_sum, timer)
        if EVAL_EVERY and (ei % EVAL_EVERY == 0 or ei == epoch + num_epochs):
            args = [model, cti, wti, itt]
            curr_f1 = evaluate(predict(sys.argv[6], *args), True, sys.argv[6],ei)

            if curr_f1 > best_f1:
                torch.save(model.state_dict(), sys.argv[6].replace('test','cnn_res')+'.pt')
                best_f1 = curr_f1
                print('model saved')

            model.train()
            print()

if __name__ == "__main__":
    if len(sys.argv) not in [7, 8, 9 ]:
        sys.exit("Usage: %s model char_to_idx word_to_idx tag_to_idx training_data (validation_data) num_epoch" % sys.argv[0])
    if len(sys.argv) == 7:
        EVAL_EVERY = False

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    try:
        cpu_dev= int(sys.argv[-1])
    except Exception as e:
        cpu_dev=0

    torch.cuda.set_device(cpu_dev)
    #os.environ["CUDA_VISIBLE_DEVICES"]=cpu_dev
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
    train()
