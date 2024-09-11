import fire
import json
import numpy as np
import os
import pickle as pkl
import torch
import tqdm
from sklearn.model_selection import train_test_split
from torch import nn, optim


def main(
        layer_idx,
        data_dir = '../chunwei-data',
        output_dir='../chunwei-data/predictions',
        num_labels=10, 
        prompt=False,
        seed=0,
        use_cuda=True,
    ):

    with open(os.path.join(data_dir, 'all_layers_1k.json'), 'r') as fin:
        record_info = json.load(fin)['records'][1:]
    
    split_file = os.path.join(data_dir, 'split', f'seed{seed}.json')
    if not os.path.isfile(split_file):
        os.system('mkdir -p ' + os.path.join(data_dir, 'split'))
        record_ids = [x['record_id'] for x in record_info]
        train_ids, test_ids = train_test_split(record_ids, random_state=seed)
        split = {'train': train_ids, 'test': test_ids}
        with open(split_file, 'w') as fout:
            json.dump(split, fout, indent=2)
    else:
        with open(split_file, 'r') as fin:
            split = json.load(fin)
    
    os.system(f'mkdir -p {output_dir}')

    # load data from corresponding layer index
    X_train, Y_train, X_test, Y_test = list(), list(), list(), list()
    test_prompt_info = list()
    for record in tqdm.tqdm(record_info): 
        X, Y = list(), list()
        iters = record['iterations']
        iteration_count = record['iteration_count']

        if prompt:

            # prediction based on averaged prompt embedding
            if layer_idx < len(iters[0]['layers']):

                layer_info = iters[0]['layers'][layer_idx]
                layer_ts = layer_info['iter_ts']

                tensor_path = os.path.join(data_dir, 'llama_layer_tensor', f'L{layer_idx}_{layer_ts}+.pt')
                if os.path.isfile(tensor_path):
                    tensor = torch.load(tensor_path)[0]
                    avg_tensor = torch.mean(tensorm, dim=0)
                    X.append(avg_tensor)
                    Y.append(iteration_count - 1)

        else:

            for iter_idx, iteration in enumerate(iters[1:]):

                if layer_idx < len(iteration['layers']):
                
                    layer_info = iteration['layers'][layer_idx]
                    layer_ts = layer_info['iter_ts']

                    tensor_path = os.path.join(data_dir, 'llama_layer_tensor', f'L{layer_idx}_{layer_ts}+.pt')
                    if os.path.isfile(tensor_path):
                        tensor = torch.load(tensor_path)[0]
                        X.append(tensor)
                        Y.append(iteration_count - 1 - 1 - iter_idx)

        if len(X) > 0:
            if len(X) > 1:
                X = torch.cat(X, dim=0)
            else:
                X = X[0]
            if record['record_id'] in split['train']:
                X_train.append(X)
                Y_train.extend(Y)
            else:
                X_test.append(X)
                Y_test.extend(Y)

                for rs in Y:
                    test_prompt_info.append({
                        'id': record['record_id'],
                        'remaining_steps': rs
                    })
        
    X_train_torch = torch.cat(X_train, dim=0).to(torch.float32)
    X_test_torch = torch.cat(X_test, dim=0).to(torch.float32)

    print('min and max raw labels', np.amin(Y_train), np.amax(Y_train))
    Y_train = np.digitize(Y_train, np.linspace(0, 512, num_labels + 1)) - 1
    Y_train_torch = torch.tensor(Y_train, dtype=torch.long)
    print('min and max digitized labels', np.amin(Y_train), np.amax(Y_train))
    assert(np.amin(Y_train) >= 0)
    assert(np.amax(Y_train) < num_labels)
    Y_test_torch = torch.tensor(np.digitize(Y_test, np.linspace(0, 512, num_labels + 1)) - 1, dtype=torch.long)

    perm_idx = np.random.permutation(X_train_torch.size(0))
    X_train_torch = X_train_torch[perm_idx]
    Y_train_torch = Y_train_torch[perm_idx]

    print('Train dim', X_train_torch.size(), Y_train_torch.size())
    print('Test dim', X_test_torch.size(), Y_test_torch.size())

    model = nn.Sequential(
        nn.Linear(X_train_torch.shape[1], 512),
        nn.ReLU(),
        nn.Linear(512, num_labels)
    )

    if torch.cuda.is_available() and use_cuda:
        model.to('cuda')
        X_train_torch = X_train_torch.to('cuda')
        Y_train_torch = Y_train_torch.to('cuda')
        X_test_torch = X_test_torch.to('cuda')
        Y_test_torch = Y_test_torch.to('cuda')

    batch_size = 32
    num_epochs = 30
    initial_lr = 0.01
    final_lr = 0.

    # Define loss function (CE) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.01)

    # cosine decay of learning rate
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=final_lr)

    # Train the model
    for epoch in range(num_epochs):  # number of epochs
        model.train()
        with tqdm.tqdm(total=len(X_train_torch), desc=f"Epoch {epoch}") as pbar:
            for batch_idx in range(0, len(X_train_torch), batch_size):
                X_batch = X_train_torch[batch_idx:batch_idx+batch_size]
                Y_batch = Y_train_torch[batch_idx:batch_idx+batch_size]
                optimizer.zero_grad()
                Y_batch_pred = model(X_batch)
                loss = criterion(Y_batch_pred, Y_batch)
                loss.backward()
                optimizer.step()
                pbar.set_description(f"Epoch {epoch} Loss {loss.item()}")
                pbar.update(batch_size)

            lr_scheduler.step()
        
        model.eval()
        with torch.no_grad():
            Y_test_pred_raw = model(X_test_torch)
            Y_test_pred = torch.argmax(Y_test_pred_raw, dim=1)
            err = (Y_test_pred != Y_test_torch).sum().item() / len(Y_test_torch)
            print(f'Error: {err}, Epoch: {epoch}, lr: {optimizer.param_groups[0]["lr"]}')

            if epoch == num_epochs - 1:
                pred_np = Y_test_pred_raw.detach().cpu().numpy()
                assert(pred_np.shape[0] == len(test_prompt_info))

                for ridx in range(pred_np.shape[0]):
                    test_prompt_info[ridx]['pred'] = pred_np[ridx, :]
                
                if prompt:
                    pstring = '_prompt'
                else:
                    pstring = ""

                with open(os.path.join(output_dir, f'L{layer_idx}_class{num_labels}{pstring}_seed{seed}.pkl'), 'wb') as fout:
                    pkl.dump(test_prompt_info, fout)

if __name__ == '__main__':
    fire.Fire(main)