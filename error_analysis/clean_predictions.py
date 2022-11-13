"""
Convert test.tsv.predictions.tsv to have the same format as test_label.tsv
"""


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--preds', type=str, help='prediction.tsv', default='exp/test.tsv.predictions.tsv')
    parser.add_argument('--output', type=str, default='exp/test_pred.tsv')
    parser.add_argument('--threshold', type=float, default=0.3)
    return parser.parse_args()


def main():
    args = get_args()

    # emotion mappings
    with open(args.preds, encoding='utf-8') as f, open(args.output, 'w', encoding='utf-8') as of:
        lines = list(f)
        emotions = lines[0].rstrip('\n').split()
        for line in lines[1:]:
            scores = [float(s) for s in line.rstrip('\n').split()]
            labels = [emotions[i] for i, s in enumerate(scores) if s > args.threshold]
            of.write(f'{",".join(labels)}\n')


if __name__ == '__main__':
    main()
