import numpy as np
import cv2
from matplotlib import pyplot as plt
from tabulate import tabulate
import time

start_time = time.time()
image_file = 'pictures/15/1.bmp'
image_file2 = 'pictures/15/2.bmp'
image_file3 = 'pictures/15/3.bmp'
delta1 = np.arange(100)
binary_coefficient1 = np.arange(0.0, 1.0, 0.1)  # 0.5
n = 100
delta01 = np.zeros(100)
delta02 = np.zeros(100)
delta03 = np.zeros(100)
Ekavg = np.zeros(100)
Eshavg = np.zeros(100)
Ekavg0 = np.zeros(100)
Eshavg0 = np.zeros(100)
Max_Ekavg0 = np.zeros(100)
Max_Eshavg0 = np.zeros(100)
bin_coefs = np.zeros(100)

for s in range(len(binary_coefficient1)):
    binary_coefficient = np.round(binary_coefficient1[s], 2)
    for i in range(100):
        delta = delta1[i]

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
            ones = 0
            zeros = 0

            # бинаризация матрицы
            for i in range(len(image_src)):
                for j in range(len(image_src[i])):
                    if ND[i] <= image_src[i][j] <= VD[i]:
                        image_b[i][j] = 1
                        ones += 1
                    else:
                        image_b[i][j] = 0
                        zeros += 1

            # нахождение эталонного вектора
            e_vector = np.mean(image_b, axis=0)
            for i in range(len(e_vector)):
                if e_vector[i] > binary_coefficient:
                    e_vector[i] = 1
                else:
                    e_vector[i] = 0

            return e_vector, image_b


        # вывод штрихкодов
        def barcode(array):
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 2))
            ax.title.set_text("Barcode")
            ax.imshow(array.reshape(1, -1), cmap='binary', aspect='auto', interpolation='nearest')
            # plt.show()


        # функция подсчета дистанции между эталонвекторами + подсчет sk1, sk2
        def distance2(ev1_arr, ev2_arr, b_array1, b_array2):
            ev1 = np.array(ev1_arr)
            ev2 = np.array(ev2_arr)
            arr1 = np.array(b_array1)
            arr2 = np.array(b_array2)

            result_e = sum(abs(ev1 - ev2))
            # print('distance between ev1 and ev2:', result_e)

            sk1 = np.empty([int(len(arr1))])
            sk2 = np.empty([int(len(arr2))])

            for i in range(len(arr1)):
                sk1[i] = sum(abs(ev1 - arr1[i]))
            for i in range(len(arr2)):
                sk2[i] = sum(abs(ev1 - arr2[i]))
            # print('sk1: ', sk1)
            # print('sk2: ', sk2)

            return result_e, sk1, sk2


        # создание таблицы (таблица + вызов функции подсчета sk1, sk2)
        def table(vector1, vector2, img_b1, img_b2):
            # вызов функции подсчета sk1, sk2
            result_e, sk1, sk2 = distance2(vector1, vector2, img_b1, img_b2)
            d = np.arange(n)
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

            # print(
            #     tabulate(np.array([d, k1, k2, d1, d2, alfa, beta, e, esh]).T,
            #              headers=["d", "K1", "K2", "D1", "D2", "α", "β", "E", "Esh"]))

            # ! механизм отсечения для графиков!
            e0 = np.ones(100)
            for i in range(100):
                if d1[i] < 0.5 or d2[i] < 0.5 or d[i] > result_e:
                    e0[i] = e[i]  # 0
                else:
                    e0[i] = e[i]

            esh0 = np.ones(100)
            for i in range(100):
                if d1[i] < 0.5 or d2[i] < 0.5 or d[i] > result_e:
                    esh0[i] = esh[i]  # 0
                else:
                    esh0[i] = esh[i]

            return e, esh, e0, esh0, result_e, d1, d2


        def searchMaxE(ek, esh, d1, d2):
            ekmax = np.amax(ek)
            eshmax = np.amax(esh)
            maxInd = np.argmax(ek)
            maxInd2 = np.argmax(esh)
            d1 = d1[maxInd]
            d2 = d2[maxInd]
            return ekmax, eshmax, maxInd, maxInd2, d1, d2


        e_vector, image_b1 = binarize(image_file)
        e_vector2, image_b2 = binarize(image_file2)
        e_vector3, image_b3 = binarize(image_file3)
        e1, esh1, e01, esh01, result_e, d11, d12 = table(e_vector, e_vector2, image_b1, image_b2)
        e2, esh2, e02, esh02, result_e2, d21, d22 = table(e_vector, e_vector3, image_b1, image_b3)
        e3, esh3, e03, esh03, result_e3, d31, d32 = table(e_vector2, e_vector3, image_b2, image_b3)

        E01, Esh01, maxInd11, maxInd12, D_11, D_12 = searchMaxE(e1, esh1, d11, d12)
        E02, Esh02, maxInd21, maxInd22, D_21, D_22 = searchMaxE(e2, esh2, d21, d22)
        E03, Esh03, maxInd31, maxInd32, D_31, D_32 = searchMaxE(e3, esh3, d31, d32)

        # ----------------------------------------------------------------
        Ekavg[i] = (E01 + E02 + E03) / 3
        Eshavg[i] = (Esh01 + Esh02 + Esh03) / 3
        if D_11 < 0.5 or D_12 < 0.5 or maxInd11 > result_e or maxInd12 > result_e2 or D_21 < 0.5 or D_22 < 0.5 \
                or maxInd21 > result_e or maxInd22 > result_e2 or D_31 < 0.5 or D_32 < 0.5 or maxInd32 > result_e3:
            Ekavg0[i] = 0
            Eshavg0[i] = 0
        else:
            Ekavg0[i] = (E01 + E02 + E03) / 3
            Eshavg0[i] = (Esh01 + Esh02 + Esh03) / 3

        Ekavg[0] = 0
        Eshavg[0] = 0
        Ekavg[99] = 0
        Eshavg[99] = 0

    Max_Ekavg0[s] = np.amax(Ekavg0)
    delta01[s] = np.argmax(Ekavg0)
    Max_Eshavg0[s] = np.amax(Eshavg0)
    delta02[s] = np.argmax(Eshavg0)
    bin_coefs[s] = binary_coefficient

# табличка максимальных КФЕ при всех дельтах

# ! ГРАФИК !
fig, ([ax1, ax2]) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

ax1.plot(Ekavg, color="black")
ax1.plot(Ekavg0, color="black")
ax2.plot(Eshavg, color="black")
ax2.plot(Eshavg0, color="black")

ax1.set_xlabel('delta', fontsize=14)
ax1.set_ylabel('KFE', fontsize=14)
ax1.set_title(['Кульбак', ' b_coef: ', bin_coefs[np.argmax(Max_Ekavg0)]], fontsize=14)
ax2.set_xlabel('delta', fontsize=14)
ax2.set_ylabel('KFE', fontsize=14)
ax2.set_title(['Шеннон', ' b_coef: ', bin_coefs[np.argmax(Max_Eshavg0)]], fontsize=14)

ax1.fill(Ekavg, hatch="+++", facecolor="lightblue", edgecolor="red")
ax1.fill(Ekavg0, hatch="|", facecolor="lightgreen", edgecolor="red")
ax2.fill(Eshavg, hatch="+++", facecolor="lightblue", edgecolor="red")
ax2.fill(Eshavg0, hatch="|", facecolor="lightgreen", edgecolor="red")
plt.tight_layout()
plt.show()

print("Значення Ekavg= ", np.amax(Max_Ekavg0), " при delta = ", delta01[np.argmax(Max_Ekavg0)], " при b_coef:",
      bin_coefs[np.argmax(Max_Ekavg0)])
print("Значення Eshavg= ", np.amax(Eshavg0), " при delta = ", delta02[np.argmax(Max_Eshavg0)], " при b_coef: ",
      bin_coefs[np.argmax(Max_Eshavg0)])
print("--- %s seconds ---" % (time.time() - start_time))
