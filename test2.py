import numpy as np
import cv2
from matplotlib import pyplot as plt
from tabulate import tabulate

#
# image_file = 'pictures/1.bmp'
# image_file2 = 'pictures/2.bmp'
# delta1 = np.arange(100)
# binary_coefficient1 = np.arange(0.1, 1.0, 0.1)  # (0.1, 1.0, 0.1) (0.5, 0.6, 0.1)
# n = 100
# Ek_1_Max = np.zeros(100)
# Ek_2_Max = np.zeros(100)
# Esh_1_Max = np.zeros(100)
# Esh_2_Max = np.zeros(100)
# Ek_1_Max0 = np.zeros(100)
# Ek_2_Max0 = np.zeros(100)
# Esh_1_Max0 = np.zeros(100)
# Esh_2_Max0 = np.zeros(100)
# DM_11 = np.zeros(100)
# DM_12 = np.zeros(100)
# DM_21 = np.zeros(100)
# DM_22 = np.zeros(100)
# gresult_e = np.zeros(100)
# gresult_e2 = np.zeros(100)
# gmaxInd1 = np.zeros(100)
# gmaxInd2 = np.zeros(100)
# Ek_1_MAX = np.zeros(100)
# Ek_2_MAX = np.zeros(100)
# Esh_1_MAX = np.zeros(100)
# Esh_2_MAX = np.zeros(100)
# delta01 = np.zeros(100)
# delta02 = np.zeros(100)
# delta03 = np.zeros(100)
# delta04 = np.zeros(100)
#
# for s in range(len(binary_coefficient1)):
#     binary_coefficient = np.round(binary_coefficient1[s], 2)
#     for i in range(100):
#         delta = delta1[i]
#
#         # допуски для базового класса
#         image_base = cv2.imread(image_file)
#         image_base = cv2.cvtColor(image_base, cv2.COLOR_BGR2GRAY)
#         image_array = np.around(np.mean(image_base, axis=1), 0)
#         VD = image_array + delta
#         ND = image_array - delta
#
#
#         # def optimization():
#         #     global ekmax2, ekmax1, eshmax1, eshmax2
#         #     for i in range(100):
#         #         delta = i
#         #         Ek_1_Max = np.zeros(100)
#         #         Ek_2_Max = np.zeros(100)
#         #         Esh_1_Max = np.zeros(100)
#         #         Esh_2_Max = np.zeros(100)
#         #
#         #         # e_vector, image_b1 = binarize(image_file)
#         #         # e_vector2, image_b2 = binarize(image_file2)
#         #         # e01, esh01, maxInd1 = searchMaxE(e1, esh1)
#         #         # e02, esh02, maxInd2 = searchMaxE(e2, esh2)
#         #
#         #         Ek_1_Max[i] = e01
#         #         Ek_2_Max[i] = e02
#         #         Esh_1_Max[i] = esh01
#         #         Esh_2_Max[i] = esh02
#         #         ekmax1 = np.Ek_1_Max
#         #         eshmax1 = np.Esh_1_Max
#         #         ekmax2 = np.Ek_1_Max
#         #         eshmax2 = np.Esh_1_Max
#         #
#         #     return ekmax1, ekmax2, eshmax1, eshmax2
#
#         def binarize(image_file):
#             image_src = cv2.imread(image_file)
#             image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
#             image_src = cv2.resize(image_src, (100, 100))
#             image_b = np.zeros(shape=(100, 100))
#             ones = 0
#             zeros = 0
#
#             # бинаризация матрицы
#             for i in range(len(image_src)):
#                 for j in range(len(image_src[i])):
#                     if ND[i] <= image_src[i][j] <= VD[i]:
#                         image_b[i][j] = 1
#                         ones += 1
#                     else:
#                         image_b[i][j] = 0
#                         zeros += 1
#
#             # нахождение эталонного вектора
#             e_vector = np.mean(image_b, axis=0)
#             for i in range(len(e_vector)):
#                 if e_vector[i] > binary_coefficient:
#                     e_vector[i] = 1
#                 else:
#                     e_vector[i] = 0
#
#             # print('binaries_matrix: ', image_b)
#             # print('et vector: ', e_vector)
#             # print('matrix: ', image_src)
#
#             # соотошение 1/0 в матрице
#             # print('\n', '1: ', round((ones / 10000) * 100, 2),
#             #       '\n', '0: ', round((zeros / 10000) * 100), 2)
#
#             # вывод картинок
#             # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
#             #
#             # # ax1.axis("off")
#             # ax1.title.set_text('Original')
#             #
#             # # ax2.axis("off")
#             # ax2.title.set_text("Binarized")
#             #
#             # ax1.imshow(image_src, cmap='gray')
#             # ax2.imshow(image_b, cmap='binary')
#             # plt.tight_layout()
#             # plt.show()
#             return e_vector, image_b
#
#
#         # вывод штрихкодов
#         def barcode(array):
#             fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 2))
#             ax.title.set_text("Barcode")
#             ax.imshow(array.reshape(1, -1), cmap='binary', aspect='auto', interpolation='nearest')
#             # plt.show()
#
#
#         # def distance(ev1_arr, ev2_arr):
#         #     ev1 = np.array(ev1_arr)
#         #     ev2 = np.array(ev2_arr)
#         #
#         #     result = sum(abs(ev1 - ev2))
#         #     print('distance between ev1 and ev2:', result)
#         #     return result
#
#         # функция подсчета дистанции между эталонвекторами + подсчет sk1, sk2
#         def distance2(ev1_arr, ev2_arr, b_array1, b_array2):
#             ev1 = np.array(ev1_arr)
#             ev2 = np.array(ev2_arr)
#             arr1 = np.array(b_array1)
#             arr2 = np.array(b_array2)
#
#             result_e = sum(abs(ev1 - ev2))
#             # print('distance between ev1 and ev2:', result_e)
#
#             sk1 = np.empty([int(len(arr1))])
#             sk2 = np.empty([int(len(arr2))])
#
#             for i in range(len(arr1)):
#                 sk1[i] = sum(abs(ev1 - arr1[i]))
#             for i in range(len(arr2)):
#                 sk2[i] = sum(abs(ev1 - arr2[i]))
#             # print('sk1: ', sk1)
#             # print('sk2: ', sk2)
#
#             return result_e, sk1, sk2
#
#
#         # создание таблицы (таблица + вызов функции подсчета sk1, sk2)
#         def table(vector1, vector2, img_b1, img_b2):
#             # вызов функции подсчета sk1, sk2
#             result_e, sk1, sk2 = distance2(vector1, vector2, img_b1, img_b2)
#             d = np.arange(n)
#             k1 = np.zeros(n)
#             k2 = np.zeros(n)
#
#             for j in range(len(d)):
#                 for i in range(len(sk1)):
#                     if d[j] >= sk1[i]:
#                         k1[j] += 1
#
#             for j in range(len(d)):
#                 for i in range(len(sk2)):
#                     if d[j] >= sk2[i]:
#                         k2[j] += 1
#
#             d1 = (np.array(k1 / n)) + 0.0001
#             d2 = (np.array(1 - k2 / n)) + 0.0001
#             alfa = (1 - k1 / n) + 0.0001
#             beta = (k2 / n) + 0.0001
#             e = (0.5 * np.log2((d1 + d2) / (alfa + beta))) * (d1 + d2 - alfa - beta)
#
#             esh = 1 + 0.5 * ((alfa / (alfa + d2)) * (np.log2(alfa / (alfa + d2))) +
#                              (beta / (beta + d1)) * (np.log2(beta / (beta + d1))) +
#                              (d1 / (d1 + beta)) * (np.log2(d1 / (d1 + beta))) +
#                              (d2 / (d2 + alfa)) * (np.log2(d2 / (d2 + alfa))))
#
#             # print(
#             #     tabulate(np.array([d, k1, k2, d1, d2, alfa, beta, e, esh]).T,
#             #              headers=["d", "K1", "K2", "D1", "D2", "α", "β", "E", "Esh"]))
#
#             # ! механизм отсечения для графиков!
#             e0 = np.ones(100)
#             for i in range(100):
#                 if d1[i] < 0.5 or d2[i] < 0.5 or d[i] > result_e:
#                     e0[i] = e[i]  # 0
#                 else:
#                     e0[i] = e[i]
#
#             esh0 = np.ones(100)
#             for i in range(100):
#                 if d1[i] < 0.5 or d2[i] < 0.5 or d[i] > result_e:
#                     esh0[i] = esh[i]  # 0
#                 else:
#                     esh0[i] = esh[i]
#
#             return e, esh, e0, esh0, result_e, d1, d2
#
#
#         def searchMaxE(ek, esh, d1, d2):
#             ekmax = np.amax(ek)
#             eshmax = np.amax(esh)
#             maxInd = np.argmax(ek)
#             d1 = d1[maxInd]
#             d2 = d2[maxInd]
#             return ekmax, eshmax, maxInd, d1, d2
#
#
#         e_vector, image_b1 = binarize(image_file)
#         e_vector2, image_b2 = binarize(image_file2)
#         e1, esh1, e01, esh01, result_e, d11, d12 = table(e_vector, e_vector2, image_b1, image_b2)
#         e2, esh2, e02, esh02, result_e2, d21, d22 = table(e_vector2, e_vector, image_b2, image_b1)
#
#         E01, Esh01, maxInd1, D_11, D_12 = searchMaxE(e1, esh1, d11, d12)
#         E02, Esh02, maxInd2, D_21, D_22 = searchMaxE(e2, esh2, d21, d22)
#
#         #----------------------------------------------------------------
#         # Ekavg = np.zeros(100)
#         # Eshavg = np.zeros(100)
#         # Ekavg[i] = (E01 + E02)/2
#         # Eshavg[i] = (Esh01 + Esh02) / 2
#
#         gresult_e[i] = result_e
#         gresult_e2[i] = result_e2
#         gmaxInd1[i] = maxInd1
#         gmaxInd2[i] = maxInd2
#         if D_11 < 0.5 or D_12 < 0.5 or maxInd1 > result_e:
#             Ek_1_Max[i] = 0
#         else:
#             Ek_1_Max[i] = E01
#         if D_11 < 0.5 or D_12 < 0.5 or maxInd1 > result_e:
#             Ek_2_Max[i] = 0
#         else:
#             Ek_2_Max[i] = E02
#         if D_21 < 0.5 or D_22 < 0.5 or maxInd2 > result_e2:
#             Esh_1_Max[i] = 0
#         else:
#             Esh_1_Max[i] = Esh01
#         if D_21 < 0.5 or D_22 < 0.5 or maxInd2 > result_e2:
#             Esh_2_Max[i] = 0
#         else:
#             Esh_2_Max[i] = Esh02
#         Ek_1_Max0[i] = E01
#         Ek_2_Max0[i] = E02
#         Esh_1_Max0[i] = Esh01
#         Esh_2_Max0[i] = Esh02
#         DM_11[i] = D_11
#         DM_12[i] = D_12
#         DM_21[i] = D_21
#         DM_22[i] = D_22
#
#     # табличка максимальных КФЕ при всех дельтах
#     print(tabulate(
#         np.array(
#             [delta1, DM_11, DM_12, Ek_1_Max, Esh_1_Max, gresult_e, DM_21, DM_22, Ek_2_Max, Esh_2_Max, gresult_e2]).T,
#         headers=["delta1", "DM_11", "DM_12", "Ek_1_Max", "Esh_1_Max", "gresult_e", "DM_21", "DM_22", "Ek_2_Max",
#                  "Esh_2_Max",
#                  "gresult_e2"]))
#     # отсечение нулевых растояний между веткорами
#     if gresult_e[np.argmax(Ek_1_Max)] != 0:
#         Ek_1_MAX[s] = np.amax(Ek_1_Max)
#     else:
#         Ek_1_MAX[s] = 0
#     if gresult_e[np.argmax(Ek_2_Max)] != 0:
#         Ek_2_MAX[s] = np.amax(Ek_2_Max)
#     else:
#         Ek_2_MAX[s] = 0
#     if gresult_e2[np.argmax(Esh_1_Max)] != 0:
#         Esh_1_MAX[s] = np.amax(Esh_1_Max)
#     else:
#         Esh_1_MAX[s] = 0
#     if gresult_e2[np.argmax(Esh_1_Max)] != 0:
#         Esh_2_MAX[s] = np.amax(Esh_2_Max)
#     else:
#         Esh_2_MAX[s] = 0
#
#     delta01[s] = np.argmax(Ek_1_Max)
#     delta02[s] = np.argmax(Ek_2_Max)
#     delta03[s] = np.argmax(Esh_1_Max)
#     delta04[s] = np.argmax(Esh_2_Max)
#
#     # ! ГРАФИК !
#     # fig1, (ax1) = plt.subplots()
#     # ax1.title.set_text(binary_coefficient)
#     # ax1.plot(Ek_1_Max, color="red")
#     # ax1.plot(Ek_2_Max, color="green")
#     # ax1.plot(Esh_1_Max, color="blue")
#     # ax1.plot(Esh_2_Max, color="orange")
#     # plt.tight_layout()
#     # plt.show()
#     Ek_1_Max0[0] = 0
#     Ek_2_Max0[0] = 0
#     Esh_1_Max0[0] = 0
#     Esh_2_Max0[0] = 0
#     Ek_1_Max0[99] = 0
#     Ek_2_Max0[99] = 0
#     Esh_1_Max0[99] = 0
#     Esh_2_Max0[99] = 0
#
#
#     fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(nrows=2, ncols=2, gridspec_kw={'height_ratios': [4, 2]})
#
#     ax1.plot(Ek_1_Max, color="black")
#     ax1.plot(Ek_1_Max0, color="black")
#     ax2.plot(Ek_2_Max, color="black")
#     ax2.plot(Ek_2_Max0, color="black")
#     ax3.plot(Esh_1_Max, color="black")
#     ax3.plot(Esh_1_Max0, color="black")
#     ax4.plot(Esh_2_Max, color="black")
#     ax4.plot(Esh_2_Max0, color="black")
#
#     ax1.set_xlabel('delta', fontsize=14)
#     ax1.set_ylabel('KFE', fontsize=14)
#     ax1.set_title('1 class (Кульбак)', fontsize=14)
#     ax2.set_xlabel('delta', fontsize=14)
#     ax2.set_ylabel('KFE', fontsize=14)
#     ax2.set_title('2 class (Кульбак)', fontsize=14)
#     ax3.set_xlabel('delta', fontsize=14)
#     ax3.set_ylabel('KFE', fontsize=14)
#     ax3.set_title('1 class (Шеннон)', fontsize=14)
#     ax4.set_xlabel('delta', fontsize=14)
#     ax4.set_ylabel('KFE', fontsize=14)
#     ax4.set_title('2 class (Шеннон)', fontsize=14)
#
#     ax1.fill(Ek_1_Max0, hatch="+++", facecolor="lightblue", edgecolor="red")
#     ax1.fill(Ek_1_Max, hatch="|", facecolor="lightgreen", edgecolor="red")
#     ax2.fill(Ek_2_Max0, hatch="+++", facecolor="lightblue", edgecolor="red")
#     ax2.fill(Ek_2_Max, hatch="|", facecolor="lightgreen", edgecolor="red")
#     ax3.fill(Esh_1_Max0, hatch="+++", facecolor="lightblue", edgecolor="red")
#     ax3.fill(Esh_1_Max, hatch="|", facecolor="lightgreen", edgecolor="red")
#     ax4.fill(Esh_2_Max0, hatch="+++", facecolor="lightblue", edgecolor="red")
#     ax4.fill(Esh_2_Max, hatch="|", facecolor="lightgreen", edgecolor="red")
#     plt.tight_layout()
#     plt.show()
#
#     print("binary_coefficient: ", binary_coefficient)
#     print("Значення КФЕ(Шеннон)= ", np.amax(Esh_1_Max), " КФЕ(Кульбак)= ", np.amax(Ek_1_Max),
#           " для 1-го классу максимальне при delta = ", delta1[np.argmax(Esh_1_Max)], " ; ", delta1[np.argmax(Ek_1_Max)])
#     print("Значення КФЕ(Шеннон)= ", np.amax(Esh_2_Max), " КФЕ(Кульбак)= ", np.amax(Ek_2_Max),
#           " для 2-го классу максимальне при delta = ", delta1[np.argmax(Esh_2_Max)], " ; ", delta1[np.argmax(Ek_2_Max)])
#     print("Міжцентрова кодова відстань ", gresult_e[np.argmax(Ek_1_Max)], " ", gresult_e2[np.argmax(Ek_2_Max)])
#
# print("Ek_1_MAX: ", np.amax(Ek_1_MAX), "delta: ", delta01[np.argmax(Ek_1_MAX)], "binary_coefficient: ",
#       binary_coefficient1[np.argmax(Ek_1_MAX)])
# print("Ek_2_MAX: ", np.amax(Ek_2_MAX), "delta: ", delta02[np.argmax(Ek_2_MAX)], "binary_coefficient: ",
#       binary_coefficient1[np.argmax(Ek_2_MAX)])
# print("Esh_1_MAX: ", np.amax(Esh_1_MAX), "delta: ", delta03[np.argmax(Esh_1_MAX)], "binary_coefficient: ",
#       binary_coefficient1[np.argmax(Esh_1_MAX)])
# print("Esh_2_MAX: ", np.amax(Esh_2_MAX), "delta: ", delta04[np.argmax(Esh_2_MAX)], "binary_coefficient: ",
#       binary_coefficient1[np.argmax(Esh_2_MAX)])
# import time
# start_time = time.time()
# for i in range(1000000):
#     print(i)
# # d = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# # e = [0, 0, 0, 10, 10, 10, 0, 0, 0, 0]  # 6; 10
#
# # print(np.amax(d), np.amax(e), np.argmax(d), np.argmax(e))
# # p, z = max(zip(e, range(len(e))))
# # print(p, z)
# # if np.amax(d) and np.amax(e):
# #     x, y = np.amax(d), np.amax(e)
# #     print(x, y)
# print("--- %s seconds ---" % (time.time() - start_time))
