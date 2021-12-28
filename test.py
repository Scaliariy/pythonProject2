import numpy as np
import cv2
from matplotlib import pyplot as plt
from tabulate import tabulate

image_file = 'pictures/2/1.bmp'
image_file2 = 'pictures/2/2.bmp'
image_file3 = 'pictures/2/4.bmp'
image_exam = 'pictures/2/4.bmp'

delta = 50
binary_coefficient = 0.5
n = 100
d = np.arange(100)

# допуски для базового класса
image_base = cv2.imread(image_file)
image_base = cv2.cvtColor(image_base, cv2.COLOR_BGR2GRAY)
image_array = np.around(np.mean(image_base, axis=1), 0)
VD = image_array + delta
ND = image_array - delta


def binarize(image_file):
    image_src = cv2.imread(image_file)
    image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
    image_src = cv2.resize(image_src, (100, 100))
    image_b = np.zeros(shape=(100, 100))

    # бинаризация матрицы
    for i in range(len(image_src)):
        for j in range(len(image_src[i])):
            if ND[i] <= image_src[i][j] <= VD[i]:
                image_b[i][j] = 1
            else:
                image_b[i][j] = 0

    # нахождение эталонного вектора
    e_vector = np.mean(image_b, axis=0)
    for i in range(len(e_vector)):
        if e_vector[i] > binary_coefficient:
            e_vector[i] = 1
        else:
            e_vector[i] = 0

    # вывод картинок
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
    ax1.title.set_text('Original')
    ax2.title.set_text("Binarized")
    ax1.imshow(image_src, cmap='gray')
    ax2.imshow(image_b, cmap='binary')
    plt.tight_layout()
    plt.show()
    return e_vector, image_b


# вывод штрихкодов
def barcode(array):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 2))
    ax.title.set_text("Barcode")
    ax.imshow(array.reshape(1, -1), cmap='binary', aspect='auto', interpolation='nearest')
    plt.show()


# функция подсчета дистанции между эталонвекторами + подсчет sk1, sk2
def distance2(ev1_arr, ev2_arr, b_array1, b_array2):
    ev1 = np.array(ev1_arr)
    ev2 = np.array(ev2_arr)
    arr1 = np.array(b_array1)
    arr2 = np.array(b_array2)

    result_e = sum(abs(ev1 - ev2))
    print('distance between ev1 and ev2:', result_e)

    sk1 = np.empty([int(len(arr1))])
    sk2 = np.empty([int(len(arr2))])

    for i in range(len(arr1)):
        sk1[i] = sum(abs(ev1 - arr1[i]))
    for i in range(len(arr2)):
        sk2[i] = sum(abs(ev1 - arr2[i]))
    print('sk1: ', sk1)
    print('sk2: ', sk2)

    return result_e, sk1, sk2


# создание таблицы (таблица + вызов функции подсчета sk1, sk2)
def table(vector1, vector2, img_b1, img_b2):
    # вызов функции подсчета sk1, sk2
    result_e, sk1, sk2 = distance2(vector1, vector2, img_b1, img_b2)
    # d = np.arange(n)
    k1 = np.zeros(n)
    k2 = np.zeros(n)

    for j in range(len(d)):
        for i in range(len(sk1)):
            if d[j] >= sk1[i]:
                k1[j] += 1

    for j in range(len(d)):
        for i in range(len(sk2)):
            if d[j] >= sk2[i]:
                k2[j] += 1

    d1 = (np.array(k1 / n)) + 0.0001
    d2 = (np.array(1 - k2 / n)) + 0.0001
    alfa = (1 - k1 / n) + 0.0001
    beta = (k2 / n) + 0.0001
    e = (0.5 * np.log2((d1 + d2) / (alfa + beta))) * (d1 + d2 - alfa - beta)

    esh = 1 + 0.5 * ((alfa / (alfa + d2)) * (np.log2(alfa / (alfa + d2))) +
                     (beta / (beta + d1)) * (np.log2(beta / (beta + d1))) +
                     (d1 / (d1 + beta)) * (np.log2(d1 / (d1 + beta))) +
                     (d2 / (d2 + alfa)) * (np.log2(d2 / (d2 + alfa))))

    d1 = d1 - 0.0001
    d2 = d2 - 0.0001
    alfa = alfa - 0.0001
    beta = beta - 0.0001

    print(
        tabulate(np.array([d, k1, k2, d1, d2, alfa, beta, e, esh]).T,
                 headers=["d", "K1", "K2", "D1", "D2", "α", "β", "E", "Esh"]))

    e0 = np.ones(100)
    for i in range(100):
        if d1[i] < 0.5 or d2[i] < 0.5 or d[i] > result_e:
            e0[i] = 0
        else:
            e0[i] = e[i]

    esh0 = np.ones(100)
    for i in range(100):
        if d1[i] < 0.5 or d2[i] < 0.5 or d[i] > result_e:
            esh0[i] = 0
        else:
            esh0[i] = esh[i]

    return e, esh, e0, esh0, result_e


def searchMaxE(ek, esh):
    ekmax = np.amax(ek)
    eshmax = np.amax(esh)
    value1, maxIndK = max(zip(ek, range(len(ek))))
    value2, maxIndSh = max(zip(esh, range(len(esh))))
    return ekmax, eshmax, maxIndK, maxIndSh


def exam(image_be, ev1, ev2, ev3, maxIndK1, maxIndK2, maxIndK3, maxIndSh1, maxIndSh2, maxIndSh3):
    mu1avk = np.zeros(100)
    mu2avk = np.zeros(100)
    mu3avk = np.zeros(100)
    mu1avsh = np.zeros(100)
    mu2avsh = np.zeros(100)
    mu3avsh = np.zeros(100)
    for i in range(len(image_be)):
        x = image_be[i]
        d1 = sum(abs(ev1 - x))
        d2 = sum(abs(ev2 - x))
        d3 = sum(abs(ev3 - x))
        mu1k = 1 - d1 / maxIndK1
        mu2k = 1 - d2 / maxIndK2
        mu3k = 1 - d3 / maxIndK3
        mu1sh = 1 - d1 / maxIndSh1
        mu2sh = 1 - d2 / maxIndSh2
        mu3sh = 1 - d3 / maxIndSh3
        mu1avk[i] = mu1k
        mu2avk[i] = mu2k
        mu3avk[i] = mu3k
        mu1avsh[i] = mu1sh
        mu2avsh[i] = mu2sh
        mu3avsh[i] = mu3sh

    print("Для Кульбака: \t mu1: ", np.average(mu1avk), "mu2: ", np.average(mu2avk), "mu3: ", np.average(mu3avk))
    if np.average(mu1avk) > 0 or np.average(mu2avk) > 0 or np.average(mu3avk) > 0:
        print("Отже скоріше за все дані реалізації належать класу ",
              np.argmax([np.average(mu1avk), np.average(mu2avk), np.average(mu3avk)]) + 1)
    else:
        print("Отже скоріше за все дані реалізації не належать жодному з класів")
    print("Для Шеннона: \t mu1: ", np.average(mu1avsh), "mu2: ", np.average(mu2avsh), "mu3: ", np.average(mu3avsh))
    if np.average(mu1avsh) > 0 or np.average(mu2avsh) > 0 or np.average(mu3avsh) > 0:
        print("Отже скоріше за все дані реалізації належать класу ",
              np.argmax([np.average(mu1avsh), np.average(mu2avsh), np.average(mu3avsh)]) + 1)
    else:
        print("Отже скоріше за все дані реалізації не належать жодному з класів")


e_vector, image_b1 = binarize(image_file)
barcode(e_vector)
e_vector2, image_b2 = binarize(image_file2)
barcode(e_vector2)
e_vector3, image_b3 = binarize(image_file3)
barcode(e_vector3)
e_vector4, image_be = binarize(image_exam)
barcode(e_vector4)
e1, esh1, e01, esh01, result_e = table(e_vector, e_vector2, image_b1, image_b2)
e2, esh2, e02, esh02, result_e2 = table(e_vector, e_vector3, image_b1, image_b3)
e3, esh3, e03, esh03, result_e3 = table(e_vector2, e_vector3, image_b2, image_b3)

e1[0] = 0
e2[0] = 0
e3[0] = 0
e1[99] = 0
e2[99] = 0
e3[99] = 0
e01[0] = 0
e02[0] = 0
e03[0] = 0
e01[99] = 0
e02[99] = 0
e03[99] = 0

esh1[0] = 0
esh2[0] = 0
esh3[0] = 0
esh1[99] = 0
esh2[99] = 0
esh3[99] = 0
esh01[0] = 0
esh02[0] = 0
esh03[0] = 0
esh01[99] = 0
esh02[99] = 0
esh03[99] = 0

fig, ([ax1, ax2, ax5], [ax3, ax4, ax6]) = plt.subplots(nrows=2, ncols=3, gridspec_kw={'height_ratios': [4, 2]})

ax1.plot(e1, color="black")
ax1.plot(e01, color="black")
ax2.plot(e2, color="black")
ax2.plot(e02, color="black")
ax5.plot(e3, color="black")
ax5.plot(e03, color="black")
ax3.plot(esh1, color="black")
ax3.plot(esh01, color="black")
ax4.plot(esh2, color="black")
ax4.plot(esh02, color="black")
ax6.plot(esh3, color="black")
ax6.plot(esh03, color="black")

ax1.set_xlabel('d', fontsize=14)
ax1.set_ylabel('KFE', fontsize=14)
ax1.set_title('1 class (Кульбак)', fontsize=14)
ax2.set_xlabel('d', fontsize=14)
ax2.set_ylabel('KFE', fontsize=14)
ax2.set_title('2 class (Кульбак)', fontsize=14)
ax3.set_xlabel('d', fontsize=14)
ax3.set_ylabel('KFE', fontsize=14)
ax3.set_title('1 class (Шеннон)', fontsize=14)
ax4.set_xlabel('d', fontsize=14)
ax4.set_ylabel('KFE', fontsize=14)
ax4.set_title('2 class (Шеннон)', fontsize=14)
ax5.set_xlabel('d', fontsize=14)
ax5.set_ylabel('KFE', fontsize=14)
ax5.set_title('3 class (Кульбак)', fontsize=14)
ax6.set_xlabel('d', fontsize=14)
ax6.set_ylabel('KFE', fontsize=14)
ax6.set_title('3 class (Шеннон)', fontsize=14)

ax1.fill(e1, hatch="+++", facecolor="lightblue", edgecolor="red")
ax1.fill(e01, hatch="|", facecolor="lightgreen", edgecolor="red")
ax2.fill(e2, hatch="+++", facecolor="lightblue", edgecolor="red")
ax2.fill(e02, hatch="|", facecolor="lightgreen", edgecolor="red")
ax3.fill(esh1, hatch="+++", facecolor="lightblue", edgecolor="red")
ax3.fill(esh01, hatch="|", facecolor="lightgreen", edgecolor="red")
ax4.fill(esh2, hatch="+++", facecolor="lightblue", edgecolor="red")
ax4.fill(esh02, hatch="|", facecolor="lightgreen", edgecolor="red")
ax5.fill(e3, hatch="+++", facecolor="lightblue", edgecolor="red")
ax5.fill(e03, hatch="|", facecolor="lightgreen", edgecolor="red")
ax6.fill(esh3, hatch="+++", facecolor="lightblue", edgecolor="red")
ax6.fill(esh03, hatch="|", facecolor="lightgreen", edgecolor="red")
plt.tight_layout()
plt.show()

e01, esh01, maxIndK1, maxIndSh1 = searchMaxE(e1, esh1)
e02, esh02, maxIndK2, maxIndSh2 = searchMaxE(e2, esh2)
e03, esh03, maxIndK3, maxIndSh3 = searchMaxE(e3, esh3)

# --- таблица значений картинкой ---
table_values = np.column_stack(
    [["ev1", "ev2", "ev3"], [0, result_e, result_e2], [result_e, 0, result_e3], [result_e2, result_e3, 0]])

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
column_labels = [" ", "ev1", "ev2", "ev3"]
ax.axis('tight')
ax.axis('off')
ax.table(cellText=table_values, colLabels=column_labels, loc="center", cellLoc="center")
plt.tight_layout()
plt.show()

exam(image_be, e_vector, e_vector2, e_vector3, maxIndK1, maxIndK2, maxIndK3, maxIndSh1, maxIndSh2, maxIndSh3)

print("Значення КФЕ(Шеннон)= ", esh01, "; d= ", maxIndSh1, " КФЕ(Кульбак)= ", e01, "; d= ", maxIndK1,
      " для 1-го классу максимальне при delta = ", delta,
      "; d= ", maxIndK1)
print("Значення КФЕ(Шеннон)= ", esh02, "; d= ", maxIndSh2, " КФЕ(Кульбак)= ", e02, "; d= ", maxIndK2,
      " для 2-го классу максимальне при delta = ", delta,
      "; d= ", maxIndK2)
print("Значення КФЕ(Шеннон)= ", esh03, "; d= ", maxIndSh3, " КФЕ(Кульбак)= ", e03, "; d= ", maxIndK3,
      " для 3-го классу максимальне при delta = ", delta,
      "; d= ", maxIndK3)
print("Міжцентрова кодова відстань result_e: ", result_e, " result_e2: ", result_e2, " result_e3: ", result_e3)
