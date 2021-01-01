'''
Cross - Entropy
D(Yp, Y) = -1/N * sum(Y * log(Yt))

In pytorch,
nn.CrossEntropyLoss applies : nn.LogSoftmax + nn.NLLLoss
(negative loss likelihood loss)

Do NOT use,
Softmax in last layer.
Y has class label, not One-Hot Encoded.
Yt has raw scores (logits), no Softmax

nn.BCELoss() is for binary classification.
We need to apply sigmoid at the end.

'''

import torch
import torch.nn as nn

loss = nn.CrossEntropyLoss()

y_true = torch.tensor([0])
y_pred = torch.tensor([[2.0, 1.0, 0.1]])

l1 = loss(y_pred, y_true)

print(l1.item())

_, predictions = torch.max(y_pred, 1)
print(predictions)
