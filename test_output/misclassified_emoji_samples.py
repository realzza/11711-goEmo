import emoji
from typing import List


def extract_emojis(s):
    return ''.join(c for c in s if c in emoji.UNICODE_EMOJI['en'])


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('baseline_pred_file', type=str)
    parser.add_argument('emoji2vec_pred_file', type=str)
    return parser.parse_args()


def count_correct_preds(lines: List[str]):
    correct_emoji_count = 0
    for line in lines:
        sent, tl, pl = line.split('\t')
        if tl != pl:
            # print(sent + '\n', tl + '\n', pl + '\n')
            continue
        correct_emoji_count += 1
    return correct_emoji_count


def main():
    args = get_args()

    with open(args.baseline_pred_file, encoding='utf-8') as f:
        baseline_lines = [line.rstrip('\n') for line in f]
    with open(args.emoji2vec_pred_file, encoding='utf-8') as f:
        emoji2vec_lines = [line.rstrip('\n') for line in f]

    emoji_line_idx = [i for i, line in enumerate(emoji2vec_lines) if emoji.distinct_emoji_list(line)]

    n_emojis = len(emoji_line_idx)
    print(f'{n_emojis}/{len(emoji2vec_lines)} have emojis')

    # for i in emoji_line_idx:
    #     print(baseline_lines[i])
    #     print(emoji2vec_lines[i])

    baseline_lines = [baseline_lines[i] for i in emoji_line_idx]
    emoji2vec_lines = [emoji2vec_lines[i] for i in emoji_line_idx]

    baseline_corr_count = count_correct_preds(baseline_lines)
    emoji2vec_corr_count = count_correct_preds(emoji2vec_lines)

    print(baseline_corr_count / n_emojis, emoji2vec_corr_count / n_emojis)


if __name__ == '__main__':
    main()
