import numpy as np
from kohonen.KohonenMap import KohonenMap

inputs = np.array([[2.3, 6.7, 2.5, 3.6, 2, 1, 0.22, 2.9, 280, 1, 6, "Middle"],  # Грузовой автомобиль
                   [2.7, 13.5, 4.3, 5.5, 60, 1, 0.22, 4.2, 380, 2, 8, "Big"],  # Автобус
                   [2.1, 4.0, 1.3, 1.3, 5, 0, 0.16, 2.2, 180, 0, 4, "Little"],  # Легковой автомобиль
                   [1.0, 2.8, 1.1, 0.25, 2, 0, 0.18, 1.5, 90, 0, 2, "Micro"],  # Мотоцикл
                   [1.1, 2.6, 1.1, 0.22, 2, 0, 0.18, 1.5, 100, 0, 2, "Micro"],  # Мотоцикл
                   [2.0, 4.8, 1.3, 1.5, 5, 0, 0.2, 2.2, 190, 0, 4, "Little"],  # Легковой автомобиль
                   [2.5, 12.0, 4.1, 5.0, 50, 1, 0.22, 4.0, 350, 2, 8, "Big"],  # Автобус
                   [2.7, 14.0, 3.9, 5.4, 55, 1, 0.22, 4.2, 380, 2, 8, "Big"],  # Автобус
                   [1.1, 2.7, 1.2, 0.23, 2, 0, 0.17, 1.4, 95, 0, 2, "Micro"],  # Мотоцикл
                   [2.2, 6.4, 2.3, 3.6, 2, 1, 0.23, 2.8, 250, 1, 6, "Middle"],  # Грузовой автомобиль
                   [2.7, 14.0, 3.9, 5.4, 55, 1, 0.22, 4.2, 380, 2, 8, "Big"],  # Автобус
                   [2.1, 6.5, 2.2, 3.7, 2, 1, 0.22, 2.4, 270, 1, 6, "Middle"],  # Грузовой автомобиль
                   [1.0, 2.8, 1.2, 0.25, 2, 0, 0.18, 1.5, 80, 0, 2, "Micro"],  # Мотоцикл
                   [2.1, 6.5, 2.2, 3.7, 2, 1, 0.22, 2.4, 270, 1, 6, "Middle"],  # Грузовой автомобиль
                   [2.1, 4.7, 1.5, 1.4, 5, 0, 0.2, 2.2, 200, 0, 4, "Little"],  # Легковой автомобиль
                   [1.8, 4.5, 1.4, 1.2, 5, 0, 0.10, 1.9, 170, 0, 4, "Little"],  # Легковой автомобиль
                   [2.2, 6.3, 2.4, 3.4, 2, 1, 0.22, 2.6, 240, 1, 6, "Middle"],  # Грузовой автомобиль
                   [2.0, 4.8, 1.3, 1.5, 5, 0, 0.2, 2.2, 190, 0, 4, "Little"],  # Легковой автомобиль
                   [1.1, 2.7, 1.0, 0.25, 2, 0, 0.18, 1.2, 90, 0, 2, "Micro"],  # Мотоцикл
                   [2.1, 4.3, 1.4, 1.2, 5, 0, 0.2, 2.0, 120, 0, 4, "Little"],  # Легковой автомобиль
                   [2.0, 4.5, 1.5, 1.5, 5, 0, 0.18, 2.0, 150, 0, 4, "Little"],  # Легковой автомобиль
                   [2.9, 14.0, 4.2, 5.5, 40, 1, 0.22, 4.2, 380, 2, 8, "Big"],  # Автобус
                   [1.0, 2.9, 1.0, 0.23, 2, 0, 0.17, 1.4, 80, 0, 2, "Micro"],  # Мотоцикл
                   [2.7, 16.0, 4.1, 5.3, 60, 1, 0.22, 4.1, 370, 2, 8, "Big"],  # Автобус
                   [2.0, 6.6, 2.3, 3.5, 2, 1, 0.23, 2.8, 230, 1, 6, "Middle"],  # Грузовой автомобиль
                   [2.1, 6.5, 2.2, 3.7, 2, 1, 0.22, 2.4, 270, 1, 6, "Middle"],  # Грузовой автомобиль
                   [2.8, 15.0, 4.3, 5.4, 60, 1, 0.22, 4.0, 350, 2, 8, "Big"],  # Автобус
                   [1.1, 2.6, 1.1, 0.26, 2, 0, 0.18, 1.4, 90, 0, 2, "Micro"],  # Мотоцикл
                   [2.2, 6.3, 2.3, 3.5, 2, 1, 0.21, 2.7, 240, 1, 6, "Middle"],  # Грузовой автомобиль
                   [2.1, 4.3, 1.4, 1.2, 5, 0, 0.2, 2.0, 120, 0, 4, "Little"],  # Легковой автомобиль
                   [2.0, 6.0, 2.2, 3.5, 2, 1, 0.21, 2.7, 240, 1, 6, "Middle"],  # Грузовой автомобиль
                   [3.0, 15.0, 4.0, 5.2, 45, 1, 0.22, 4.1, 390, 2, 8, "Big"],  # Автобус
                   [1.0, 2.8, 1.2, 0.24, 2, 0, 0.17, 1.5, 95, 0, 2, "Micro"],  # Мотоцикл
                   [2.6, 13.0, 4.2, 5.3, 55, 1, 0.22, 4.1, 370, 2, 8, "Big"],  # Автобус
                   [1.2, 2.5, 1.1, 0.22, 2, 0, 0.17, 1.3, 85, 0, 2, "Micro"],  # Мотоцикл
                   [1.9, 4.5, 1.4, 1.2, 5, 0, 0.17, 2.3, 160, 0, 4, "Little"]])  # Легковой автомобиль

seed = 42
inputs = KohonenMap.conversionToFloat(inputs)
input_size = len(inputs[0])
map_size = (10, 10)
epochs = 500
learning_rate =0.1
kohonen_map = KohonenMap(input_size, map_size,seed,epochs,learning_rate)
kohonen_map.train(KohonenMap.normalize(inputs))