import torch
import torch.nn as nn

# Model çıktıları (logits) - [batch_size, num_classes, height, width]
logits = torch.randn(16, 9, 64, 64)  # Örnek çıktı: 16x9x64x64

# One-hot encoded target - [batch_size, num_classes, height, width]
one_hot_target = torch.randint(0, 9, (16, 9, 64, 64))
one_hot_target = torch.nn.functional.one_hot(one_hot_target.argmax(dim=1), num_classes=9).permute(0, 3, 1, 2)
print(one_hot_target.shape)
# One-hot encoded hedefi sınıf indekslerine dönüştür
# argmax ile sınıf indekslerini elde edelim
targets = one_hot_target.argmax(dim=1)  # [batch_size, height, width]

# CrossEntropyLoss tanımlama
criterion = nn.CrossEntropyLoss()

# Loss hesaplama
loss = criterion(logits, targets)

# Sonucu yazdır
print(loss.item())
