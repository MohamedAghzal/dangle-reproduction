import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import evaluate
import json

class SemanticDataset(Dataset):
 
    def __init__(self, file_name, tokenizer, max_len=512):
        df = pd.read_csv(file_name, sep="\t", names=["text", "label", "sem_type"])
        
        text = list(df["text"].values)
        labels = list(df["label"].values)
        text_encoding = tokenizer(text, truncation=True, padding=True)
        label_encoding = tokenizer(labels, truncation=True, padding=True)
        self.text_ids = torch.tensor(list(text_encoding.values()), dtype=torch.int64)[0]
        self.label_ids = torch.tensor(list(label_encoding.values()), dtype=torch.int64)[0]
        
    def __len__(self):
        return len(self.label_ids)

    def __getitem__(self,idx):
        return self.text_ids[idx], self.label_ids[idx]

class Dangle_Model(nn.Module):
    def __init__(self, T5_modelname, tokenizer, device, max_len=512):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(T5_modelname)
        self.model.resize_token_embeddings(len(tokenizer))
        self.max_len = max_len
        tokenizer.add_special_tokens({'sep_token':'<s>'})
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
    
    def save_checkpoint(self, save_dir):
        self.model.save_pretrained(save_dir)
    
    def generate(self, input_ids):
        # Put sep_token at the end of the input_ids
        input_ids = torch.cat([input_ids, torch.tensor([self.tokenizer.sep_token_id]).repeat([input_ids.shape[0], 1]).to(self.device)], dim=-1)
        target_offset = input_ids.shape[1]
        current_target_len = 1
        total_target_len = self.max_len
        model_out = self.model.generate(input_ids=input_ids, max_length=total_target_len)
        # Add the next generated token to the input
        input_ids = torch.cat([input_ids, model_out.argmax(dim=-1)[:, current_target_len].unsqueeze(-1)], dim=-1)
        current_target_len += 1
        while current_target_len <= total_target_len:
            # Repeat until we have generated the max length
            model_out = self.model.generate(input_ids=input_ids, max_length=total_target_len)
            input_ids = torch.cat([input_ids, model_out.argmax(dim=-1)[:, current_target_len].unsqueeze(-1)], dim=-1)
            current_target_len += 1
        return input_ids[:, target_offset:]
    
    def forward(self, input_ids, labels):
        print("Input Ids: ", input_ids.shape)
        print("Labels: ", labels.shape)
        # Put sep_token at the end of the input_ids
        input_ids = torch.cat([input_ids, torch.tensor([self.tokenizer.sep_token_id]).repeat([input_ids.shape[0], 1]).to(self.device)], dim=-1)
        print("Input Ids after concat: ", input_ids.shape)
        current_target_len = 1
        total_target_len = labels.shape[1]
        print("Labels going in: ", labels[:, :current_target_len].shape)
        model_out = self.model(input_ids=input_ids, labels=labels[:, :current_target_len])
        print("Model ran.")
        loss = model_out.loss
        print("Loss: ", loss)
        # Add the next generated token to the input
        input_ids = torch.cat([input_ids, model_out.logits.argmax(dim=-1)[:, current_target_len].unsqueeze(-1)], dim=-1)
        print("Updated Input Ids: ", input_ids.shape)
        current_target_len += 1
        while current_target_len <= total_target_len:
            # Repeat until we have generated the label hopefully
            print("Labels going in: ", labels[:, :current_target_len].shape)
            model_out = self.model(input_ids=input_ids, labels=labels[:, :current_target_len])
            loss = torch.add(loss, model_out.loss)
            print("Loss: ", loss)
            input_ids = torch.cat([input_ids, model_out.logits.argmax(dim=-1)[:, current_target_len].unsqueeze(-1)], dim=-1)
            print("Updated Input Ids: ", input_ids.shape)
            current_target_len += 1
        return {'loss': loss, 'logits': model_out.logits}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    
    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(command=train)
    train_parser.add_argument("--train_dir", type=str, default=None)
    train_parser.add_argument("--val_dir", type=str, default=None)
    train_parser.add_argument("--test_dir", type=str, default=None)
    train_parser.add_argument("--T5_modelname", type=str, default="t5-base")
    train_parser.add_argument("--batch_size", type=int, default=8)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--learning_rate", "--lr", type=float, default=2e-4)
    train_parser.add_argument("--save_dir", type=str, default=None)
    train_parser.add_argument("--cuda", action="store_true")

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.set_defaults(command=evaluate_fn)
    eval_parser.add_argument("--test_dir", type=str, default=None)
    eval_parser.add_argument("--checkpoint_dir", type=str, default=None)
    eval_parser.add_argument("--predictions_dir", type=str, default=None)
    eval_parser.add_argument("--batch_size", type=int, default=8)
    eval_parser.add_argument("--cuda", action="store_true")

    args = parser.parse_args()
    args.command(args)

def train(args):
    print("Training with the following arguments: ", args)
    device = torch.device("cuda" if args.cuda else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(args.T5_modelname, model_max_length=512)
    model = Dangle_Model(args.T5_modelname, tokenizer, device)

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
        print("Epoch: ", epoch)
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
            pbar.set_postfix(dict(
                                Loss=loss.item(),
                                Avg_Loss=loss_sum / (step + 1)
                                ))
            optimizer.step()
            optimizer.zero_grad()
        print(f"Training Loss: {loss_sum / len(train_loader)}")

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
                model.save_checkpoint(args.save_dir)
                tokenizer.save_pretrained(args.save_dir)
                print(f"Best model saved to {args.save_dir}")

    # Test the model
    # Load the best model
    if args.save_dir is not None:
        tokenizer = T5Tokenizer.from_pretrained(args.save_dir)
        model = Dangle_Model(args.save_dir, tokenizer)
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
        tokenizer = T5Tokenizer.from_pretrained(args.checkpoint_dir)
        model = Dangle_Model(args.checkpoint_dir, tokenizer, device)
        model.to(device)

    # Use whatever metric was listed in the paper

    test_data = SemanticDataset(args.test_dir, tokenizer)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    
    pbar = tqdm(test_loader)
    pbar.set_description("Evaluation")

    exact_match_metric = evaluate.load("exact_match")
    

    predictions = []
    true_labels = []

    outputs = []
    # Probably should add use predictions directory arg here so we aren't writing this file in the src dir
    with open('predictions.json', 'w') as f:
        # save the predictions to a file
        for input_ids, label_ids in pbar:
            texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            input_ids = input_ids.to(device)
            label_ids = label_ids.to(device)
            generated_ids = model.generate(input_ids, max_length=512)

            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for text, label, generated_text in zip(texts, labels, generated_texts):
                outputs.append({
                    'text': text,
                    'label': label,
                    'generated': generated_text
                })

                predictions.append(generated_text)
                true_labels.append(label)

            json_object = json.dumps(outputs)
            f.write(json_object)
            f.write('\n')

    #prints exact match 
    results = exact_match_metric.compute(predictions=predictions, references=true_labels)
    print("Exact Match Accuracy: ", results['exact_match'])

if __name__ == "__main__":
    main()
