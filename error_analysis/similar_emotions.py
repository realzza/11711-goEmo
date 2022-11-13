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
    with open(args.input, encoding='utf-8') as f:
        samples = list(f)[1:]
        for line in samples:
            sentence, labels, preds = line.rstrip('\n').split('\t')
            labels = labels.split(',')
            preds = preds.split(',')
            lab_groups = list(set([
                emotion2group[lab]
                for lab in labels if lab != 'disgust' and lab != 'neutral'  # skip them since it has no peers
            ]))

            for p in preds:
                if p not in ['', 'neutral', 'disgust'] and p not in labels and emotion2group[p] in lab_groups:
                    emo_confusion.setdefault(p, 0)
                    emo_confusion[p] += 1

    print({k: v / len(samples) for k, v in emo_confusion.items()})


if __name__ == '__main__':
    main()
