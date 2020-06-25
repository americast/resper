from model import *
from utils import *

def load_model():
    cti = load_tkn_to_idx(sys.argv[2]) # char_to_idx
    wti = load_tkn_to_idx(sys.argv[3]) # word_to_idx
    itt = load_idx_to_tkn(sys.argv[4]) # idx_to_tag
    itw = idx_to_tkn(wti) # idx_to_word
    model = rnn(len(cti), len(wti), len(itt))
    print(model)
    load_checkpoint(sys.argv[1], model)
    return model, cti, wti, itt, itw

def run_model(model, itw, itt, batch):
    batch_size = len(batch) # real batch size
    while len(batch) < BATCH_SIZE:
        batch.append([-1, "", [], [], ""])
    batch.sort(key = lambda x: -len(x[3]))
    xc, xw = batchify(*zip(*[(x[2], x[3]) for x in batch]), True, True)
    result = model(xc, xw, maskset(xw))
    if VERBOSE:
        Va = model.attn.Va.tolist() # attention weights
    for i in range(batch_size):
        # here is the actual prediction going on......

        y = itt[result[i].argmax()]
        p = round(max(result[i]).exp().item(), NUM_DIGITS)
        batch[i].append(y)
        batch[i].append(p)
        if VERBOSE:
            print(batch[i][1])
            y = enumerate(result[i].exp().tolist())
            for a, b in sorted(y, key = lambda x: -x[1]):
                print(itt[a], round(b, NUM_DIGITS))
            print(heatmap(Va[i], batch[i][3], itw, sos = True, eos = True)) # attention heatmap

    # import pdb; pdb.set_trace()
    return [(x[1], *x[4:]) for x in sorted(batch[:batch_size])]

def predict(filename, model, cti, wti, itt, itw):
    data = []
    fo = open(filename)
    for idx, line in enumerate(fo):
        line = line.strip()
        # y = line[-1]
        # line = " ".join(line[0:-1])
        line, y = line.split("\t") if line.count("\t") else [line, None]
        # line = line.split()
        # y = line[-2]
        # # speaker = line[-1]
        # line2 = " ".join(line[0:-2])
        x = tokenize(line, UNIT)
        xc = [[cti[c] if c in cti else UNK_IDX for c in w] for w in x]
        xw = [wti[w] if w in wti else UNK_IDX for w in x]
        data.append([idx, line, xc, xw, y])
    fo.close()
    with torch.no_grad():
        model.eval()
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:i + BATCH_SIZE]
            for y in run_model(model, itw, itt, batch):
                yield y

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("Usage: %s model char_to_idx word_to_idx tag_to_idx test_data" % sys.argv[0])
    result = predict(sys.argv[5], *load_model())
    for x, y0, y1, p in result:
        print((x, y0, y1, p) if y0 else (x, y1, p))
