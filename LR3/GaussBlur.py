import cv2
import numpy as np

def main():

    # Задание №1
    img = cv2.imread(r'test.jpg', cv2.IMREAD_GRAYSCALE)

    standard_deviation = 9
    kernel_size = 11

    imgBlur_1 = AnotherGaussianBlur(img, kernel_size, standard_deviation)
    cv2.imshow('Original', img)
    cv2.imshow(str(kernel_size) + 'x' + str(kernel_size) + ' and deviation ' + str(standard_deviation), imgBlur_1)

    # Задание №4
    standard_deviation = 50
    kernel_size = 3
    imgBlur_2 = AnotherGaussianBlur(img, kernel_size, standard_deviation)
    cv2.imshow(str(kernel_size) + 'x' + str(kernel_size) + ' and deviation ' + str(standard_deviation), imgBlur_2)

    # Задание №5
    imgBlur_CV2 = cv2.GaussianBlur(img, (kernel_size, kernel_size), standard_deviation)
    cv2.imshow('Blur_by_CV2', imgBlur_CV2)
    cv2.waitKey(0)

# Задание №3
def AnotherGaussianBlur(img, kernel_size, standard_deviation):
    kernel = np.ones((kernel_size, kernel_size))  # Первоначальное ядро свёртки
    a = b = (kernel_size + 1) // 2  # Вычисление центрального элемента матрицы (определения пикселя в фокусе)

    # Построение матрицы свёртки
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] = gauss(i, j, standard_deviation, a, b)  # Вычисление функции Гаусса
    print(kernel)

    # Задание №2
    sum = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            sum += kernel[i, j]
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] /= sum
    print(kernel)

    # Применение операции свёртки к изображению
    imgBlur = Convolution(img, kernel)
    return imgBlur

# Реализация операции свёртки
def Convolution(img, kernel):
    kernel_size = len(kernel)
    imgBlur = img.copy()
    # Начальные координаты для итераций по пикселям
    x_start = kernel_size // 2
    y_start = kernel_size // 2
    for i in range(x_start, imgBlur.shape[0] - x_start):
        for j in range(y_start, imgBlur.shape[1] - y_start):
            # Операция свёртки - каждый пиксель умножается на соответствующий элемент ядра свертки и суммирование произведений
            val = 0
            for k in range(-(kernel_size // 2), kernel_size // 2 + 1):
                for l in range(-(kernel_size // 2), kernel_size // 2 + 1):
                    val += img[i + k, j + l] * kernel[k + (kernel_size // 2), l + (kernel_size // 2)]
            imgBlur[i, j] = val
    return imgBlur

def gauss(x, y, omega, a, b):

    # Вычисляем удвоенный квадрат параметра omega для упрощения дальнейших вычислений
    omegaIn2 = 2 * omega ** 2

    # m1 - первая часть формулы гауссовой функции, которая зависит от omegaIn2.
    # Она вычисляет нормализационный множитель так, чтобы интеграл от гауссовой функции по всей плоскости был = 1.
    # Это обеспечивает, что сумма всех значений гауссовой функции будет равна 1.
    m1 = 1 / (np.pi * omegaIn2)

    # m2 - вторая часть формулы гауссовой функции, которая вычисляет экспоненциальное значение с отрицательным аргументом.
    # (x-a) ** 2 и (y-b) ** 2 - квадраты расстояний от точки (x, y) до центра гауссовой функции (a, b).
    # Делим на omegaIn2 и берем экспоненту от отрицательного результата для определения того,
    # насколько быстро значение гауссовой функции убывает с расстоянием от центра (a, b).
    m2 = np.exp(-((x - a) ** 2 + (y - b) ** 2) / omegaIn2)

    # Возвращаем окончательное значение гауссовой функции для заданных координат (x, y).
    # Это значение будет наибольшим в центре (a, b) и уменьшается с удалением от центра согласно форме гауссовой кривой
    return m1 * m2

main()




