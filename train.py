import argparse
import torch
import pandas as pd
from agedb import *
from torch.utils.data import DataLoader
import torch.nn as nn
from utils import setup_seed
from network import ResNet_cross_entropy
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()

#model_parameter
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--age_groups', type=int, default=3)
parser.add_argument('--model_depth', type=str, default='50', help='model name')
#data_parameter
parser.add_argument('--dataset', type=str, default='agedb',
                    choices=['imdb_wiki', 'agedb'], help='dataset name')
parser.add_argument('--data_dir', type=str,
                    default='/root/autodl-tmp/data', help='data directory')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--img_size', type=int, default=224,
                    help='image size used in training')
parser.add_argument('--groups', type=int, default=10,
                    help='number of split bins to the awole datasets')
parser.add_argument('--workers', type=int, default=32,
                    help='number of workers used in data loading')
parser.add_argument('--reweight', type=str, default='inv',  choices=['inv', 'sqrt_inv'],
                    help='weight : inv or sqrt_inv')
parser.add_argument('--smooth', default='none', choices=['lds', 'none'], help='use LDS or not')

#training_parameter
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--optimizer', type=str, default='adam',
                    choices=['adam', 'sgd'], help='optimizer type')
parser.add_argument('--group_method', type=str, default='in_order',
                    choices=['in_order', 'by_count'])
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument('--epoch', type=int, default=90)
parser.add_argument('--ce_weight', type=float, default=1.5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data_loader(args):
    print('=====> Preparing data...')
    df = pd.read_csv(os.path.join(args.data_dir, "agedb.csv"))
    df_train, df_val, df_test = df[df['split'] ==
                                'train'], df[df['split'] == 'val'], df[df['split'] == 'test']
    train_labels = df_train['age']
    if args.group_method == 'in_order':
        train_shot_dict = get_shots_in_order(train_labels)
    elif args.group_method == 'by_count':
        train_shot_dict = get_shots_by_count(train_labels)
    #
    train_dataset = AgeDB(data_dir=args.data_dir, df=df_train, img_size=args.img_size, train_shot_dict = train_shot_dict,
                        split='train', reweight=args.reweight, group_num=args.groups, smooth=args.smooth)   
    #
    val_dataset = AgeDB(data_dir=args.data_dir, df=df_val, 
                        img_size=args.img_size, train_shot_dict = train_shot_dict, split='val', group_num=args.groups)
    test_dataset = AgeDB(data_dir=args.data_dir, df=df_test,
                        img_size=args.img_size, train_shot_dict = train_shot_dict, split='test', group_num=args.groups)
    #
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, pin_memory=pin_memory, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin_memory, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin_memory, drop_last=False)
    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")
    print(f"Test data size: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, train_labels

@torch.no_grad()
def evaluate_point_mae(model, val_loader):
    """Compute ordinary, unweighted point-prediction MAE."""
    model.eval()
    absolute_error_sum = 0.0
    sample_count = 0


    for x, y, _, _ in val_loader:
        x = x.to(device)
        y = y.to(device)

        y_hat, z, logits = model(x)
        absolute_error_sum += torch.abs(y_hat - y).sum().item()
        sample_count += y.numel()

    if sample_count == 0:
        raise ValueError('Cannot evaluate MAE on an empty loader.')

    val_mae = absolute_error_sum / sample_count
    return val_mae

def train(args, model, train_loader, val_loader, log_file):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[60, 80],gamma=0.1)

    reg_criterion = nn.L1Loss()
    ce_criterion = nn.CrossEntropyLoss()
    best_val_mae = float("inf")

    for epoch in tqdm(range(args.epoch), desc="Training"):
        model.train()
        epoch_l1_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_total_loss = 0.0

        for idx, (x, y, w, age_class) in enumerate(train_loader):
            x, y, w, age_class= x.to(device), y.to(device), w.to(device), age_class.to(device)

            y_hat, z, logits = model(x)
            l1_loss = reg_criterion(y_hat, y)
            cross_entropy_loss = ce_criterion(logits, age_class)

            loss = l1_loss + args.ce_weight * cross_entropy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_l1_loss += l1_loss.item()
            epoch_ce_loss += cross_entropy_loss.item()
            epoch_total_loss += loss.item()
            
        num_per_batch = len(train_loader)

        current_val_mae = evaluate_point_mae(model, val_loader)
        if current_val_mae < best_val_mae:
            best_val_mae = current_val_mae
            torch.save(model.state_dict(),f"best_model_{args.group_method}.pth")

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        message = (
        f"Epoch [{epoch + 1}/{args.epoch}] "
        f"lr: {current_lr:.6f},"
        f"l1: {epoch_l1_loss / num_per_batch:.6f}, "
        f"CE: {epoch_ce_loss / num_per_batch:.6f}, "
        f"Total: {epoch_total_loss / num_per_batch:.6f}, "
        f"Val MAE: {current_val_mae:.4f}, "
        f"Best Val MAE: {best_val_mae:.4f}"
        )

        print(message)
        print(message, file=log_file, flush=True)
                
    return model

@torch.no_grad()
def test(model, test_loader, args):
    model_path = f'best_model_{args.group_method}.pth'
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    absolute_error_sum = 0.0
    sample_count = 0

    for x, y, _, _ in test_loader:
        x = x.to(device)
        y = y.to(device)

        y_hat, z, logits = model(x)
        absolute_error_sum += torch.abs(y_hat - y).sum().item()
        sample_count += y.numel()
        
    if sample_count == 0:
        raise ValueError('Cannot evaluate MAE on an empty loader.')
        
    test_mae = absolute_error_sum / sample_count
    return test_mae


def main():
    args = parser.parse_args()
    setup_seed(args.seed)
    os.makedirs("logs", exist_ok=True)
    log_path = (
        f"logs/"
        f"{args.group_method}_"
        f"ce{args.ce_weight}_"
        f"wd{args.weight_decay}_"
        f"seed{args.seed}.txt"
    )

    train_loader, val_loader, test_loader, train_labels = get_data_loader(args)

    model = ResNet_cross_entropy(args)

    with open(log_path, mode="w", encoding="utf-8", buffering=1) as log_file:
        print(f"Group method: {args.group_method}", file=log_file)
        print(f"CE weight: {args.ce_weight}", file=log_file)
        print(f"Learning rate: {args.lr}", file=log_file)
        print(f"Batch size: {args.batch_size}", file=log_file)
        print(f"Epochs: {args.epoch}", file=log_file)
        print(f"Seed: {args.seed}", file=log_file)
        print("-" * 80, file=log_file, flush=True)

        model = train(args, model, train_loader, val_loader, log_file)

        test_mae = test(model, test_loader, args)
        test_message = (
            f"Test MAE of method {args.group_method}: "
            f"{test_mae:.4f}"
            )

        print(test_message)
        print(test_message, file=log_file, flush=True)

if __name__ == '__main__':
    main()
