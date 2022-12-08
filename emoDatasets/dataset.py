import torch
from typing import List


class GoEmotionDataset:
    def __init__(self, texts, labels, tokenizer, max_len, replace_emoticon=False):
        self.texts = texts
        self.labels = labels

        self.tokenizer = tokenizer
        self.max_len = max_len

        if replace_emoticon:
            self.texts = self.clean_emoticons(self.texts, self.tokenizer.sep_token)

    def clean_emoticons(self, texts: List[str], sep_token: str):
        import emot
        emot_obj = emot.core.emot()

        for i, res in enumerate(emot_obj.bulk_emoticons(texts)):
            if res['flag']:
                new_str = texts[i]
                for emoticon, desc in zip(res['value'], res['mean']):
                    new_str = new_str.replace(emoticon, '')
                    new_str += sep_token + " " + desc
                texts[i] = new_str

        return texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        inputs = self.tokenizer(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def check():
    import transformers
    model_name = "squeezebert/squeezebert-uncased"
    tokenizer = transformers.SqueezeBertTokenizer.from_pretrained(
        model_name, do_lower_case=True
    )
    dataset = GoEmotionDataset(
        [':-) hello there'],
        [1],
        tokenizer,
        100,
    )

    sample = dataset[0]
    print(tokenizer.decode(sample['ids']))


if __name__ == '__main__':
    check()
