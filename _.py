import torch
import torch.nn as nn


# 16 batch, 33 sınıf, 120x120 patch boyutunda model çıktısı (logits)
outputs = torch.randn(16, 33, 120, 120)  # Rastgele tahminler (logits)

# 16 batch, 120x120 patch boyutunda, sınıf etiketlerini içeren hedef tensor
target = torch.randint(0, 33, (16, 120, 120))  # 0-32 arasında rastgele sınıf etiketleri

# CrossEntropyLoss fonksiyonunu tanımlıyoruz
criterion = nn.CrossEntropyLoss()

# Loss hesaplama
loss = criterion(outputs, target)

print(loss)
