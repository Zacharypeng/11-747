import csv

train_file_txt_path = 'topicclass/topicclass_train.txt'
train_file_csv_path = 'topicclass/topicclass_train.csv'
valid_file_txt_path = 'topicclass/topicclass_valid.txt'
valid_file_csv_path = 'topicclass/topicclass_valid.csv'
test_file_txt_path = 'topicclass/topicclass_test.txt'
test_file_csv_path = 'topicclass/topicclass_test.csv'
valid_file_onehot_path = 'topicclass/topicclass_valid_onehot.csv'
train_file_onehot_path = 'topicclass/topicclass_train_onehot.csv'
label_map_path = 'topicclass/label_map.txt'
labels_map = dict()
labels = set()

# load label map dic
with open(label_map_path, encoding='utf-8', mode='r') as f:
    data = [line.split('|') for line in f.read().splitlines()]
    for line in data:
        labels_map[line[0]] = int(line[1])

test_raw = []
with open(test_file_txt_path, encoding='utf-8', mode='r') as f:
    data = [line.split(' ||| ') for line in f.read().splitlines()]
    for row in data:
        # test_raw.append(row[1:])
        row[0] = -1

filed_names = ['LABEL', 'TEXT']
# filed_names = ['TEXT']
data.insert(0, filed_names)
# test_raw.insert(0, filed_names)
csv.register_dialect('train_data', delimiter=',')

with open(test_file_csv_path, 'w') as csv_f:
    writer = csv.writer(csv_f, dialect='train_data')
    writer.writerows(data)
    # for row in test_raw:
    #     csv_f.write(row[0]+'\n')


