epoch = 1000


def merge_files(file, out_file):
    with open('bev_classification/datasets/test_edited.txt', 'r') as l:
        paths = [line.rstrip() for line in l]

    with open(file, 'r') as r:
        labels = [line.rstrip().replace(" ", "") for line in r]
    
    with open(out_file, 'w') as w:
        for i in range(len(paths)):
            w.write(paths[i]+","+labels[i]+"\n")

top5 = "top5_val_"+str(epoch)+".txt"
out5 = "top5_val.txt"

top1 = "val_"+str(epoch)+".txt"
out1 = "top1_val.txt"

merge_files(top5, out5)
merge_files(top1, out1)