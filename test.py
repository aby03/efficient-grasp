from dataset_processing.cornell_generator import CornellDataset
import json

### TESTING
dataset = "/home/aby/Workspace/Cornell/archive"
with open(dataset+'/train_1.txt', 'r') as filehandle:
    train_data = json.load(filehandle)

train_generator = CornellDataset(
    dataset,
    train_data,
    train=True,
    shuffle=False,
    batch_size=1
)

for i in range(0, 20):
    train_generator[i]