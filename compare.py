import pandas as pd

# read submission.csv
submission = pd.read_csv('submission.csv')

# read voted_gt
voted_gt = pd.read_csv('./voted_gt/1027-0.5982.csv')

# if the label is different between the two csv, record ids
diff = voted_gt[voted_gt['label'] != submission['label']]['node_id'].values

print(len(diff))
print(diff)