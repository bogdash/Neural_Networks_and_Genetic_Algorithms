import numpy as np
import scipy.special
import cv2
import time

# открываем csv-файл и читаем оттуда
training_data_file = open("sample_data/sign_mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()


class NeuralNetwork:
    def __init__(self,
                 eta=0.05,
                 niter=1,
                 inputnodes=784,
                 hiddennodes=50,
                 outputnodes=10):
        # количество нейронов во входном слое
        self.inodes = inputnodes
        # количество нейронов в скрытом слое
        self.hnodes = hiddennodes
        # количество нейронов в выходном слое
        self.onodes = outputnodes
        # весовые коэф. между входом и скрытым слоем
        self.wih = np.random.normal(0.0,
                                    pow(self.hnodes, -0.5),
                                    (self.hnodes, self.inodes))
        # Весовые коэф. между скрытым слоем  выходом
        self.who = np.random.normal(0.0,
                                    pow(self.onodes, -0.5),
                                    (self.onodes, self.hnodes))
        # скорость обучения
        self.eta = eta
        # сигмоида в качестве функции активации
        self.activation_function = lambda x: scipy.special.expit(x)
        # количество эпох обучения
        self.niter = niter
        pass

    def train(self, X, y):
        # преобразуем входные вектора признаков в двумерный массив
        X = np.array(X, ndmin=2).T
        # преобразуем вектор целевых значений в двумерный массив
        y = np.array(y, ndmin=2).T

        # рассчитываем входящие сигналы для скрытого слоя
        hidden_inputs = np.dot(self.wih, X)
        # Рассчитываем исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # рассчитываем входящие сигналы для выходного слоя
        final_inputs = np.dot(self.who, hidden_outputs)
        # рассчитываем выходные значения
        final_outputs = self.activation_function(final_inputs)

        # считаем ошибку выходного слоя (целевое - выходное)
        output_errors = y - final_outputs
        # ошибка скрытого слоя
        hidden_errors = np.dot(self.who.T, output_errors)

        # обновляем  весовые коэф. для связей выходного и скрытого слоя
        self.who += self.eta * np.dot(output_errors *
                                      final_outputs *
                                      (1.0 - final_outputs),
                                      np.transpose(hidden_outputs))
        # обновляем  весовые коэф. для связей скрытого и входного слоев
        self.wih += self.eta * np.dot(hidden_errors *
                                      hidden_outputs *
                                      (1.0 - hidden_outputs),
                                      np.transpose(X))
        pass

    def predict(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        # рассчитываем входящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)
        # рассчитываем исходящие сигналы для скрытого слоя
        final_inputs = np.dot(self.who, hidden_outputs)
        # рассчитываем исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


# количество входных, скрытых, выходных слоев
input_nodes = 784
hidden_nodes = 300
output_nodes = 25

# коэффициент обучения
eta = 0.03
n_iter = 1

print(f"Layers: {hidden_nodes}")
print(f"eta: {eta}")
print()

# Создаём нейронку
n = NeuralNetwork(eta, n_iter, input_nodes, hidden_nodes, output_nodes)

print("Training...")

# Обучаем сеть последовательно на каждом значении из датасета
for record in training_data_list:
    all_values = record.split(',')
    # масштабируем и смещаем входные значения
    inputs = np.asfarray(all_values[1:]) / 255.0
    # Подготавливаем целевой вектор
    targets = np.zeros(output_nodes) + 0.01

    value_index = int(all_values[0])

    targets[value_index] = 0.99
    n.train(inputs, targets)

print("Training Complete!")
print()

test_num = 138
test_data_file = open("sample_data/sign_mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
all_values = test_data_list[test_num].split(',')

# масштабируем и смещаем входные значения
r = n.predict(np.asfarray(all_values[1:]) / 255.0)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

# Мапим значения индекса на букву
# (J и Z никогда не будут активны, ибо они — жесты, а не статичные картинки)
alphabet = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "J",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "X",
    24: "Y",
    25: "Z"
}

print(f"Should be {alphabet[int(all_values[0])]}")
print(f"Predicted as {alphabet[r.argmax(axis=0)[0]]}")

# Проверяем эффективность
scorecard = []
for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    #   print(int(correct_label), "истинный маркер")
    # масштабируем и смещаем входные значения
    inputs = np.asfarray(all_values[1:]) / 255.0
    outputs = n.predict(inputs)
    label = np.argmax(outputs)
    #   print(label,"ответ сети")
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
pass

scorecard_array = np.asarray(scorecard)
print("Эффективность = ", scorecard_array.sum() / scorecard_array.size)

print()

# Захват камер

cam = cv2.VideoCapture(0)


def get_images():
    _, image = cam.read()

    (h, w, _) = image.shape

    # Обрезаем квадрат из середины экрана
    #
    #       Отнять    Прибавить
    #    <----------+---------->
    #               |
    #        (h / 2)|       +------ Экран (транслируется в 1280 x 720)
    #           |   |      \|/
    #   +-------|---+----------+
    # L |    : \|/  |     :    | R
    # E |    :<---->|     :    | I
    # F |    :      |     :    | G
    # T |    :      |     :    | H
    #   |    :      |     :    | T
    #   +----------------------+
    #              /|\
    #               |
    #            (w / 2)
    #
    left_bound = w / 2 - h / 2
    right_bound = w / 2 + h / 2

    # Без этого будет ругаться на не int значения (ибо они float)
    left_bound, right_bound = int(left_bound), int(right_bound)

    # Сама обрезка изображения
    image = image[:, left_bound:right_bound]

    # сохраняет оригинальное обрезанное изображение, чтобы не "вырви глаз"
    original_image = image

    # Переводим в ЧБ
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Переводим из 720 x 720 в 28 x 28, т.к. нейронка ест только такие мелкие фотки
    image = cv2.resize(image, (28, 28))

    # Задаём фиксированный размер для обоих окон
    # (иначе 28 x 28 будет в оригинальном мелком размере)
    cv2.namedWindow("Web-Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Web-Camera", 720, 720)

    cv2.namedWindow("what NNetwork See", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("what NNetwork See", 720, 720)

    # Готовим пиксели для нейронки в виде вектора (массива)
    pixels = []

    # используем отдельную переменную, чтобы не было проблем с несоответствием типов
    what_nn_see = np.array([], dtype=np.uint8)

    # Проходимся по каждому пикселю
    for row in image:
        what_nn_see = np.append(pixels, row)

        for pixel in row:
            pixels.append(pixel)

    # Переводим одномерный массив пикселей в двумерный
    what_nn_see = what_nn_see.reshape(len(image), len(image[0]))

    return original_image, pixels, what_nn_see


pTime = 0


# Захват видео (по кадрово)
while True:
    image, pixels, what_nn_see = get_images()

    # Скармливаем пиксели нейронке
    r = n.predict(np.asfarray(pixels) / 255.0)
    print(f"Predicted as {alphabet[r.argmax(axis=0)[0]]}")

    # Отображаем "кадры в секунду"
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(image,
                f'FPS:{int(fps)}',
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    # Показываем поток с камеры:
    # - оригинальный обрезанный
    cv2.imshow("Web-Camera", image)
    # - что подаётся на нейронку
    cv2.imshow("what NNetwork See", what_nn_see)

    # Делаем break при нажатии "q" (окно openCV должно быть в фокусе)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

# Завершаем работу окон openCV
cam.release()
cv2.destroyAllWindows()
