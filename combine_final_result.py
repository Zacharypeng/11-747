
with open('val_label_result.txt') as f:
    predictions = f.read().splitlines()

with open('topicclass/topicclass_valid.txt', mode='r') as f:
    text = f.read().splitlines()

label_map = dict()
with open('topicclass/label_map.txt', encoding='utf-8', mode='r') as f:
        data = [line.split('|') for line in f.read().splitlines()]
        for d in data:
            label_map[d[1]] = d[0]

print(label_map)

with open('final_valid_result.txt', encoding='utf-8', mode='w') as f:
    for l, t in zip(predictions, text):
        t = t.split(' ||| ')[1]
        s = label_map[l] + '\n'
        print(s)
        f.write(s)
