from predict import *
import pickle
from sklearn.metrics import * 

def evaluate(result, summary = False,file=None,epochs=10):
    print(file)
    filename = file.replace('test','cnn_res')
    # f= open(file+'.txt','a')
    f = open(filename,'a')
    f.write(str(epochs)+'\n')
    avg = defaultdict(float) # average
    tp = defaultdict(int) # true positives
    tpfn = defaultdict(int) # true positives + false negatives
    tpfp = defaultdict(int) # true positives + false positives
    ytrue=[]
    ypred=[]
    for _, y0, y1, _ in result: # actual value, prediction
        tp[y0] += y0 == y1
        tpfn[y0] += 1
        tpfp[y1] += 1
        ytrue.append(y0)
        ypred.append(y1)

    y={}
    y['pred']=ypred
    y['true']=ytrue

    # with open(file+str(epochs)+'.p','wb') as handle:
    #     pickle.dump(y,handle)

    print()
    for y in sorted(tpfn.keys()):
        pr = (tp[y] / tpfp[y]) if tpfp[y] else 0
        rc = (tp[y] / tpfn[y]) if tpfn[y] else 0
        avg["macro_pr"] += pr
        avg["macro_rc"] += rc
        if not summary:
            print("label = %s" % y)
            print("precision = %f (%d/%d)" % (pr, tp[y], tpfp[y]))
            print("recall = %f (%d/%d)" % (rc, tp[y], tpfn[y]))
            print("f1 = %f\n" % f1(pr, rc))
    avg["macro_pr"] /= len(tpfn)
    avg["macro_rc"] /= len(tpfn)
    avg["micro_f1"] = sum(tp.values()) / sum(tpfp.values())
    print("macro precision = %f" % avg["macro_pr"])
    print("macro recall = %f" % avg["macro_rc"])
    print("macro f1 = %f" % f1(avg["macro_pr"], avg["macro_rc"]))
    print("micro f1 = %f" % avg["micro_f1"])

    f1_val = f1_score(ytrue, ypred, average='macro')
    acc_val = accuracy_score(ytrue, ypred)

    print("macro f1 = " + str(f1_val))
    print("accuracy = " + str(acc_val))

    # f.write("macro precision = " + str(avg["macro_pr"])+'\n')
    # f.write("macro recall = " + str(avg["macro_rc"])+'\n')
    # f.write("macro f1 = " + str(f1(avg["macro_pr"], avg["macro_rc"]))+'\n')
    # f.write("micro f1 = " + str(avg["micro_f1"])+'\n')
    # f.write('\n')

    f.write("macro f1 = " + str(f1_val)+'\n')
    f.write("accuracy = " + str(acc_val)+'\n')

    f.write('\n')

    return f1_val

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("Usage: %s model char_to_idx word_to_idx tag_to_idx test_data" % sys.argv[0])
    evaluate(predict(sys.argv[5], *load_model()))
