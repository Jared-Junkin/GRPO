from utils import GRPODataset, generate_dataset, load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

def generate_datasets_main(dataset_params: dict, n: int, k: int)->None:


    for ds in dataset_params.keys():
        print(f"generating dataset {ds}")
        _ = generate_dataset(num_graphs = dataset_params[ds]['size'],
                             writefile = dataset_params[ds]['writefile'],
                             n=n,
                             k=k)
        print(f"wrote dataset to {dataset_params[ds]['writefile']}")


def tokenize_collate_fn(tokenizer: AutoTokenizer, batch_texts: list[str])->None:
    enc = tokenizer(
        batch_texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    )
    labels = enc["input_ids"].clone() # cloning labels because labels gives the next tokens the model needs to predict. they're automatically right shifted by 1. they must be cloned because if we pass them by reference and then do any in-place modificaitons of them they will also change the toknens

    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100 # -100 is the default ignore index for attention in hugging face. we're saying 'anywhere that is just padding, don't attend to it.'

    enc["labels"] = labels
    return enc

if __name__ == "__main__":
    n = 100
    k=100
    batch_size=32
    model_path = "./models/qwen2.5-0.5b-instruct"  # change if needed

    dataset_params = {
        'train': {
            "size": 10000,
            "writefile":"./data/train.txt"
        },
        'test': {
            "size": 100,
            "writefile":"./data/test.txt"
        },
        'prompt_examples': {
            "size": 3,
            "writefile":"./data/prompt_samples.txt"
        }
    }
    # generate_datasets_main(dataset_params = dataset_params, n=n, k=k)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True, # use fast tells hugging face to load the rust implemented backend instead of the python one
        trust_remote_code=True
    )
    print(f"tokenizer is {tokenizer}")
    


    train_dataset = GRPODataset(load_dataset(dataset_params['train']['writefile']))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=lambda batch: tokenize_collate_fn(tokenizer,batch)
    )
    batch = next(iter(train_loader))
    print(f"batch is {batch}")
    

    # create dataset objehct that reads in graphs and forms prompts
    # create dataset

    # set up logging
    # make a train_step function that
        # samples from the dataset
        # broadcasts to group dim
        # passes groups through model
        # calculates group scores from groups
        # does backprop?

    # let's just start with making a dataset object. I'll also want to see how I'm referencing that in DPO
