"""
Generate a file containing lines:   sentence   labels   predictions
"""
import json


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--preds', type=str, help='Cleaned predictions (using clean_predictions.py)',
                        default='error_analysis/test_pred.tsv')
    parser.add_argument('--labels', type=str, help='test_label.tsv (containing labels instead of indices)',
                        default='goemotions/data/test_label.tsv')
    parser.add_argument('--output', type=str, default='sentence_label_pred.tsv')
    return parser.parse_args()


def main():
    args = get_args()

    with open(
            args.preds, encoding='utf-8'
    ) as pred_f, open(
        args.labels, encoding='utf-8'
    ) as label_f, open(
        args.output, 'w', encoding='utf-8'
    ) as of:
        preds = list(pred_f)
        labels = list(label_f)
        assert len(preds) == len(labels)

        of.write(f'Sentence\tLabels\tPredictions\n')
        for i in range(len(preds)):
            p = preds[i].rstrip('\n')
            sent_label = labels[i].rstrip('\n')
            of.write(f'{sent_label}\t{p}\n')


if __name__ == '__main__':
    main()
