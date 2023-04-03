import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

class SemanticDataset(Dataset):
 
    def __init__(self, file_name, tokenizer, max_len=512):
        df = pd.read_csv(file_name, sep="\t", names=["text", "label", "sem_type"])

        text = df["text"].values
        labels = df["label"].values
        text_ids = tokenizer.batch_encode_plus(text, max_length=max_len, padding='max_length', return_tensors="pt")
        label_ids = tokenizer.batch_encode_plus(labels, max_length=max_len, padding='max_length', return_tensors="pt")

        self.x_train=torch.tensor(text_ids, dtype=torch.float32)
        self.y_train=torch.tensor(label_ids,dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self,idx):
        return self.x_train[idx],self.y_train[idx]

def main():
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--train_dir", type=str, default=None)
    train_parser.add_argument("--val_dir", type=str, default=None)
    train_parser.add_argument("--test_dir", type=str, default=None)
    train_parser.add_argument("--T5_modelname", type=str, default="t5-base")
    train_parser.add_argument("--batch_size", type=int, default=8)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--learning_rate", "--lr", type=float, default=0.001)
    train_parser.add_argument("--save_dir", type=str, default=None)
    train_parser.add_argument("--cuda", action="store_true")

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--test_dir", type=str, default=None)
    eval_parser.add_argument("--checkpoint_dir", type=str, default=None)
    eval_parser.add_argument("--predictions_dir", type=str, default=None)
    train_parser.add_argument("--cuda", action="store_true")

    args = parser.parse_args()
    args.command(args)

def train(args):

    tokenizer = T5Tokenizer.from_pretrained(args.T5_modelname, model_max_length=512)
    model = T5ForConditionalGeneration.from_pretrained(args.T5_modelname)
    model.resize_token_embeddings(len(tokenizer))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    
    # Load data into a torch dataloader
    train_data = SemanticDataset(args.train_dir, tokenizer)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    val_data = SemanticDataset(args.val_dir, tokenizer)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)

    best_val_loss = float("inf")

    # Train the model
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader)
        pbar.set_description(f"Train | Epoch {epoch}")
        loss_sum = 0
        for step, (text, label) in enumerate(pbar):
            text = text.to(device)
            label = label.to(device)
            outputs = model(input_ids=text, labels=label)
            loss = outputs.loss
            loss.backward()
            loss_sum += loss.item()
            pbar.set_postfix(f"Loss: {loss.item()}, Avg Loss: {loss_sum / (step + 1)}")
            optimizer.step()
            optimizer.zero_grad()

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            val_loss_sum = 0
            pbar = tqdm(val_loader)
            pbar.set_description(f"Eval | Epoch {epoch}")
            for text, label in pbar:
                text = text.to(device)
                label = label.to(device)
                outputs = model(input_ids=text, labels=label)
                loss = outputs.loss
                val_loss_sum += loss.item()
            print(f"Validation Loss: {val_loss_sum / len(val_loader)}")
        model.train()

        # Save the model if it is the best so far
        if val_loss_sum / len(val_loader) < best_val_loss:
            best_val_loss = val_loss_sum / len(val_loader)
            if args.save_dir is not None:
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                model.save_pretrained(args.save_dir)
                tokenizer.save_pretrained(args.save_dir)
                print(f"Best model saved to {args.save_dir}")

    # Test the model
    # Load the best model
    if args.save_dir is not None:
        model = T5ForConditionalGeneration.from_pretrained(args.save_dir)
        tokenizer = T5Tokenizer.from_pretrained(args.save_dir)
        model.resize_token_embeddings(len(tokenizer))
        model.to(device)
    model.eval()
    with torch.no_grad():
        test_data = SemanticDataset(args.test_dir, tokenizer)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
        test_loss_sum = 0
        pbar = tqdm(test_loader)
        pbar.set_description("Test")
        for text, label in pbar:
            text = text.to(device)
            label = label.to(device)
            outputs = model(input_ids=text, labels=label)
            loss = outputs.loss
            test_loss_sum += loss.item()
        print(f"Test Loss: {test_loss_sum / len(test_loader)}")


def evaluate_fn(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    # Load the model
    if args.checkpoint_dir is not None:
        model = T5ForConditionalGeneration.from_pretrained(args.checkpoint_dir)
        tokenizer = T5Tokenizer.from_pretrained(args.checkpoint_dir)
        model.resize_token_embeddings(len(tokenizer))
        model.to(device)

    # Use whatever metric was listed in the paper

    val_data = SemanticDataset(args.val_dir, tokenizer)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)
    
    pbar = tqdm(val_loader)
    pbar.set_description("validation loop")

    exact_match_metric = load("exact_match")
    

    predictions = []
    labels = []

    outputs = []
    with open('predictions.json', 'w') as f:
        # save the predictions to a file
        for text, label in pbar:
            
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            label_ids = tokenizer(label, return_tensors="pt").input_ids.to(device)
            generated_ids = model.generate(input_ids, max_length=2048)

            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            outputs.append({
                'text': text,
                'label': label,
                'generated': generated_text
            })

            predictions.append(generated_ids)
            labels.append(label_ids)

            json_object = json.dumps(outputs)
            f.write(json_object)
            f.write('\n')

    #return exact match 
    results = exact_match_metric.compute(predictions=predictions, references=references)
    return results

if __name__ == "__main__":
    main()
