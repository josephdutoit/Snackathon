def print_acc(epoch):
	with open("bev_classification/datasets/test_edited.txt", 'r') as e:
    		lines = [line.rstrip() for line in e]

	truth = [line.split("/")[2] for line in lines]

	with open("val_"+str(epoch)+".txt", 'r') as f:
	# with open("test_pred_"+str(epoch)+".txt", 'r') as f:
    		preds = [line.rstrip() for line in f]

	# preds = [line.split(",")[1] for line in lines]
	# preds = lines
	correct = 0
	for i in range(len(truth)):
    		if truth[i] == preds[i]:
        		correct += 1

	print("Got " + str(correct) + " out of " + str(len(truth)))
	print("Accuracy is: ", correct / len(truth))

	return correct / len(truth)

# Generate accuracy files

files = []
accs = []
for epoch in range(0, 10000, 500):
	print("\nEpoch:" , epoch)
	acc = print_acc(epoch)
	accs.append(acc)
	files.append("val_"+str(epoch)+".txt")

files = [f[1] for f in sorted(zip(accs, files), reverse=True)]

with open("bev_classification/datasets/test_edited.txt", 'r') as e:
    		truth = [line.rstrip() for line in e]