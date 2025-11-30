from datasets import load_dataset


def main():
    print("Hello from llm!")

    # Load WikiText-103
    dataset = load_dataset("wikitext", "wikitext-103-v1", revision="b08601e")
    train_data = dataset["train"]
    print(f"Number of training samples: {len(train_data)}")


if __name__ == "__main__":
    main()
