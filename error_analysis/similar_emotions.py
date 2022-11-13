"""
Check how much the model confuses similar emotions
"""
import json


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, help='Merged labels and predictions',
                        default='error_analysis/sentence_label_pred.tsv')
    parser.add_argument('--emotion-mapping', type=str, help='Emotion group mapping',
                        default='goemotions/data/ekman_mapping.json')
    return parser.parse_args()


def main():
    args = get_args()

    # emotion mappings
    with open(args.emotion_mapping, encoding='utf-8') as f:
        mapping = json.load(f)
    emotion2group = {}
    for group, emo in mapping.items():
        for e in emo:
            emotion2group[e] = group

    emo_confusion = {}
    emo_total = {}
    emo_error_count = {}
    exclude = {'disgust', 'neutral', ''}
    with open(args.input, encoding='utf-8') as f:
        samples = list(f)[1:]
        for line in samples:
            sentence, labels, preds = line.rstrip('\n').split('\t')
            labels = labels.split(',')
            preds = preds.split(',')
            pred_groups = list(set([
                emotion2group[p]
                for p in preds if p not in exclude  # skip them since it has no peers
            ]))

            for lab in labels:
                emo_total.setdefault(lab, 0)
                emo_total[lab] += 1
                if lab in exclude:
                    continue
                if lab not in preds:
                    emo_error_count.setdefault(lab, 0)
                    emo_error_count[lab] += 1
                    if emotion2group[lab] in pred_groups:
                        emo_confusion.setdefault(lab, 0)
                        emo_confusion[lab] += 1

    print('Num confusion / total count:', {k: v / emo_total[k] for k, v in emo_confusion.items()})
    print('Num confusion / total errors:', {k: v / emo_error_count[k] for k, v in emo_confusion.items()})


if __name__ == '__main__':
    main()
