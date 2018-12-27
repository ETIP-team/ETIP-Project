def stastic_len():
    # all_info = open('./dataset/train/train.data', 'r').read()
    all_info = open('./dataset/test/test.data', 'r').read()
    all_info = all_info.strip('\n').split('\n\n')
    max_sentences_len = 0
    label_static = {}
    for item in all_info:
        infos = item.split('\n')

        sentences_len = len(infos[0].split(' '))
        max_sentences_len = max(sentences_len, max_sentences_len)

        entitys = infos[1].split('|')
        for entity in entitys:
            RL, cls = entity.split(' G#')
            right, left = RL.split(',')
            label_len = int(left) - int(right)
            if cls not in label_static:
                label_static[cls] = {'max_len': label_len, 'num': 1, 'avg_len': label_len,
                                     'SLR': label_len / sentences_len}
            else:
                label_static[cls]['max_len'] = max(label_len, label_static[cls]['max_len'])
                label_static[cls]['num'] += 1
                label_static[cls]['avg_len'] += label_len
                label_static[cls]['SLR'] += label_len / sentences_len
    print('max_sentences_len:', max_sentences_len)
    for k, v in label_static.items():
        v['avg_len'] = v['avg_len'] / v['num']
        v['SLR'] = v['SLR'] / v['num']
        print(k)
        print(v)


def correct():
    # writefile = open('./dataset/train/train.data', 'w')
    # all_info = open('./GENIA/train.data', 'r').read()
    writefile = open('./dataset/test/test.data', 'w')
    all_info = open('./GENIA/test.data', 'r').read()
    all_info = all_info.strip('\n').split('\n\n')
    for item in all_info:
        infos = item.strip('\n').split('\n')
        if len(infos) != 3:
            print(infos)
            continue
        writefile.write('\n'.join([infos[0], infos[2], '\n']))
    writefile.close()


def words_count():
    train_info = open('./dataset/train/train.data', 'r').read().strip('\n').split('\n\n')
    test_info = open('./dataset/test/test.data', 'r').read().strip('\n').split('\n\n')
    wordlist = []
    for info in train_info:
        words = info.split('\n')[0].split(' ')
        for word in words:
            if word not in wordlist:
                wordlist.append(word)
    for info in test_info:
        words = info.split('\n')[0].split(' ')
        for word in words:
            if word not in wordlist:
                wordlist.append(word)
    print(len(wordlist))


def count_neg(neg):
    datatype = ['train', 'test']
    writefile = open('./count/' + neg + '_count.data', 'w')
    for type in datatype:
        train_info = open(type.join(['./dataset/', '/', '.data']), 'r').read().strip('\n').split('\n\n')
        writefile.write('-----------------------------------------\n' + type + 'data:\n')
        count = {}
        num = 0
        for info in train_info:
            infos = info.split('\n')
            words = infos[0].split(' ')
            range = infos[1].split('|')
            for r in range:
                RL, Cls = r.split(' G#')
                left, right = RL.split(',')
                if neg in words[int(left): int(right)]:
                    num += 1
                    writefile.write(' '.join(words[int(left): int(right)]) + '\n')
                    length = int(right) - int(left)
                    if length not in count:
                        count[length] = 0
                    count[length] += 1
        sorted_count = sorted(count.items(), key=lambda item: item[0])
        writefile.write('num: ' + str(num) + '\n(len,count) ' + str(sorted_count) + '\n')
    writefile.close()


if __name__ == '__main__':
    # stastic_len()
    # correct()
    # words_count()
    count_neg('complex')
