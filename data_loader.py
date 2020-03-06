from src.pyvov import ChipsIndex

ci = ChipsIndex()

all_experiments = ci.experiments()
print(all_experiments)
EXP_NAME = 'A1'

training_split = ci.training_split_for(EXP_NAME)
testing_split = ci.testing_split_for(EXP_NAME)
all = ci.all_for_exp(EXP_NAME)
labels = ci.labels_for(EXP_NAME)
