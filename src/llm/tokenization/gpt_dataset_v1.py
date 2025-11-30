import tiktoken
from torch.utils.data import DataLoader, Dataset


class GPTDatasetV1(Dataset):
    def __init__(self, text: str, tokenizer, max_length: int, stride: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.token_ids = self.tokenizer.encode(text)

        self.input_ids = []
        self.target_ids = []

        for i in range(0, len(self.token_ids) - self.max_length, stride):
            input_seq = self.token_ids[i : i + self.max_length]
            target_seq = self.token_ids[i + 1 : i + self.max_length + 1]

            self.input_ids.append(input_seq)
            self.target_ids.append(target_seq)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple:
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    text: str,
    max_length: int,
    stride: int,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader
