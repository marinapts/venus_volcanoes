from src.pyvov import ChipsIndex

ci = ChipsIndex()

all_experiments = ci.experiments()
EXP_NAMES = ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'C1', 'D1', 'D2', 'D3', 'D4', 'E1', 'E2', 'E3', 'E4', 'E5']
num_img = 0
training_split = []
testing_split = []
all_labels = []

for EXP_NAME in ['C1', 'D4']:
    training_split.extend(ci.training_split_for(EXP_NAME))
    testing_split.extend(ci.testing_split_for(EXP_NAME))
    labels = ci.labels_for(EXP_NAME)
    label_list = list(labels['trn'])
    label_list.extend(list(labels['tst']))
    all_labels.extend(label_list)

training_split.extend(testing_split)
full_dataset = training_split
print(len(full_dataset), len(all_labels))
