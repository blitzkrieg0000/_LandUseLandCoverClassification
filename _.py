import random

def generate_random_colors(num_colors):
    colors = []
    for _ in range(num_colors):
        color = [random.randint(0, 255) for _ in range(3)]
        colors.append(color)
    return colors

# 100 adet RGB renk listesi oluştur
random_colors = generate_random_colors(100)

# Renk listesini yazdır
print(random_colors)

