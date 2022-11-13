"""
Convert emotion indices in train.tsv/test.tsv/dev.tsv to labels
"""


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('input', type=str, help='Input .tsv file')
    parser.add_argument('output', type=str, help='Output .tsv file')
    parser.add_argument('--emotion-list', type=str, help='emotions.txt', default='data/emotions.txt')
    return parser.parse_args()


def main():
    args = get_args()

    idx2emotion = {}
    with open(args.emotion_list, encoding='utf-8') as f:
        for i, line in enumerate(f):
            emotion = line.rstrip('\n')
            idx2emotion[i] = emotion

    with open(args.input, encoding='utf-8') as f, open(args.output, 'w', encoding='utf-8') as of:
        for line in f:
            sentence, annotation, _ = line.rstrip('\n').split('\t')
            labels = [idx2emotion[int(p)] for p in annotation.split(',')]
            of.write(f'{sentence}\t{",".join(labels)}\n')


if __name__ == '__main__':
    main()
