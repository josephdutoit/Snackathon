def print_top_acc(epoch):
	with open("bev_classification/datasets/test_edited.txt", 'r') as e:
    		lines = [line.rstrip() for line in e]

	truth = [line.split("/")[2] for line in lines]

	with open("val_"+str(epoch)+".txt", 'r') as f:
	# with open("test_pred_"+str(epoch)+".txt", 'r') as f:
    		preds = [line.rstrip() for line in f]

	preds = [line.split(",") for line in preds]
	# preds = lines
	correct = 0
	for i in range(len(truth)):
    		if truth[i] in preds[i]:
        		correct += 1

	print("Got " + str(correct) + " out of " + str(len(truth)))
	print("Accuracy is: ", correct / len(truth))

	return correct / len(truth)

def print_acc(epoch):
	with open("bev_classification/datasets/test_edited.txt", 'r') as e:
    		lines = [line.rstrip() for line in e]

	truth = [line.split("/")[2] for line in lines]

	with open("val_"+str(epoch)+".txt", 'r') as f:
	# with open("test_pred_"+str(epoch)+".txt", 'r') as f:
    		preds = [line.rstrip() for line in f]

	# preds = [line.split(",") for line in preds]
	# preds = lines
	correct = 0
	for i in range(len(truth)):
    		if truth[i] in preds[i]:
        		correct += 1

	print("Got " + str(correct) + " out of " + str(len(truth)))
	print("Accuracy is: ", correct / len(truth))

	return correct / len(truth)

for epoch in range(0, 2500, 500):
	print("\nEpoch:" , epoch)
	acc = print_acc(epoch)

print("\n")

print("Top 1")
print_top_acc("top1")
print("Top 5")
print_top_acc("top5")