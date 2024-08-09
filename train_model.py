import torch
from torch import nn, optim


import numpy as np
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

def get_data_partition(by_prompt=True, train_size=0.8, max_files=None):

    fnames = sorted([x for x in os.listdir('/n/holylfs06/LABS/kempner_fellow_emalach/Lab/emalach/lllt') if x.endswith('.npz')])
    if max_files is not None:
        fnames = fnames[:max_files]
    Xs, Ys = list(), list()
    for fname in tqdm(fnames):
        dat = np.load(f'dataset/lllt/{fname}')
        Xs.append(dat['x'])
        Ys.append(dat['y'])
    
    if by_prompt:
        idxs = list(range(len(fnames)))
        train_p_idx, test_p_idx = train_test_split(idxs, train_size=train_size, random_state=10)

        X_train_test, Y_train_test = list(), list()
        prompts_train_test = list()
        for idx_list in [train_p_idx, test_p_idx]:
            X_train_test.append(np.concatenate([Xs[i] for i in idx_list], axis=0))
            Y_train_test.append(np.concatenate([Ys[i] for i in idx_list]))
            prompt_list = list()
            for i in idx_list:
                prompt_Ys = Ys[i]
                for y in prompt_Ys:
                    prompt_list.append((i, y))
            prompts_train_test.append(prompt_list)
        
        X_train, X_test = X_train_test
        Y_train, Y_test = Y_train_test
        prompts_train, prompts_test = prompts_train_test

        return X_train, Y_train, X_test, Y_test, prompts_train, prompts_test


print('loading data')
#data_file = '/n/holylfs06/LABS/kempner_fellow_emalach/Lab/emalach/lllt/all.npz'
data_file = '/dev/shm/all.npz'
data = np.load(data_file)

X_train, Y_train, X_test, Y_test = data['X_train'], data['Y_train'], data['X_test'], data['Y_test']
#X_train, Y_train, X_test, Y_test, prompts_train, prompts_test = get_data_partition(True, 0.8)

#breakpoint()


# shuffle train data
idx = np.random.permutation(len(X_train))
X_train = X_train[idx]
Y_train = Y_train[idx]

# separate Y to 10 bins
Y_train = np.digitize(Y_train, np.linspace(0, 512, 10))
Y_test = np.digitize(Y_test, np.linspace(0, 512, 10))

# # linear regression with sklearn
# reg = LinearRegression().fit(X_train, Y_train)

# # predict output
# Y_test_pred = reg.predict(X_test)

# normalize = True
# if normalize:
#     bc = np.bincount(Y_train)
#     probs = np.sum(bc)/bc
#     sample_probs = probs[Y_train]
#     sample_probs /= np.sum(sample_probs)
#     idx = np.random.choice(len(X_train), len(X_train), p=sample_probs)
#     X_train = X_train[idx]
#     Y_train = Y_train[idx]

# Convert data to PyTorch tensors
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
Y_train_torch = torch.tensor(Y_train, dtype=torch.long)
X_test_torch = torch.tensor(X_test, dtype=torch.float32)
Y_test_torch = torch.tensor(Y_test, dtype=torch.long)

# Define the model
# model = nn.Linear(X_train_torch.shape[1], 10)

# one hidden layer MLP
model = nn.Sequential(
    nn.Linear(X_train_torch.shape[1], 512),
    nn.ReLU(),
    nn.Linear(512, 10)
)

if torch.cuda.is_available():
    model.to('cuda')
    X_train_torch = X_train_torch.to('cuda')
    Y_train_torch = Y_train_torch.to('cuda')
    X_test_torch = X_test_torch.to('cuda')
    Y_test_torch = Y_test_torch.to('cuda')

batch_size = 10_000
num_epochs = 20
initial_lr = 0.01
final_lr = 0.

# Define loss function (CE) and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.01)

# cosine decay of learning rate
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=final_lr)


# Train the model
from tqdm import tqdm

for epoch in range(num_epochs):  # number of epochs
    model.train()
    with tqdm(total=len(X_train_torch), desc=f"Epoch {epoch}") as pbar:
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
        Y_test_pred = model(X_test_torch)
        Y_test_pred = torch.argmax(Y_test_pred, dim=1)
        err = (Y_test_pred != Y_test_torch).sum().item() / len(Y_test_torch)
        print(f'Error: {err}, Epoch: {epoch}, lr: {optimizer.param_groups[0]["lr"]}')

breakpoint()

W = model.weight.data.cpu().numpy()
b = model.bias.data.cpu().numpy()
Y_test_pred = model(X_test_torch).cpu().detach().numpy()

# np.save('linear_0.01_0.npz', W=W, b=b, Y_pred=Y_test_pred)
    


# Test the model

breakpoint()