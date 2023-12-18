import csv
import time

import numpy as np
from matplotlib import pyplot as plt
from numba import cuda, uint8


def mass_search_CPU(R, N, H):
    for j in range(R.shape[1]):
        for i in range(R.shape[0]):
            n = N[i]
            for k in range(len(n)):
                if n[k] in H[j]:
                    R[i, j - k] -= 1
    return R


@cuda.jit
def mass_search_GPU(R, N, H):
    for j in range(cuda.grid(2)[1], R.shape[1],
                   cuda.blockDim.y * cuda.gridDim.y):
        for i in range(cuda.grid(2)[0], R.shape[0],
                       cuda.blockDim.x * cuda.gridDim.x):
            if i < R.shape[0] and j < R.shape[1]:
                n = N[i]
                for k in range(len(n)):
                    for p in range(len(H[j])):
                        if n[k] == H[j][p]:
                            R[i, j - k] -= 1


def save_array_to_csv(my_array, filename):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(my_array)


def start_calculation(sizes: np.ndarray):
    table_values_CPU = np.zeros((0, sizes.shape[0]))
    table_values_GPU = np.zeros((0, sizes.shape[0]))

    for n in range(12):
        print(f"\nn = {n}")
        array_time_CPU = []
        array_time_GPU = []
        for size in sizes:
            N = np.random.randint(len(ABC), size=(size, 3), dtype=np.uint8)
            H = np.random.randint(len(ABC), size=(size, 3), dtype=np.uint8)
            R = np.zeros((size, 3), dtype=int)

            start_CPU = time.time()
            result_CPU = mass_search_CPU(R.copy(), N, H)
            end_CPU = time.time()
            execution_CPU = end_CPU - start_CPU
            print(
                f"Размерность --> {(size, 3)}  \nРезультат на CPU: \n{result_CPU} \nВремя выполнения --> {execution_CPU} секунд\n")
            array_time_CPU.append(execution_CPU)

            result_GPU = np.zeros((size, 3), dtype=int)

            start_GPU = time.time()

            result_GPU = cuda.to_device(result_GPU)
            N_GPU = cuda.to_device(N)
            H_GPU = cuda.to_device(H)

            mass_search_GPU[blocks_per_grid, threads_per_block](result_GPU, N_GPU, H_GPU)

            result_GPU = result_GPU.copy_to_host()
            end_GPU = time.time()
            execution_GPU = end_GPU - start_GPU
            array_time_GPU.append(execution_GPU)
            save_array_to_csv(result_GPU, f"R_GPU_{size}")
            print(
                f"Размерность --> {(size, 3)}  \nРезультат на GPU: \n{result_GPU} \nВремя выполнения --> {execution_GPU} секунд")
            print(f"Проверка равенства матриц {np.array_equal(result_GPU, result_CPU)}\n")
        table_values_CPU = np.vstack((table_values_CPU, np.array(array_time_CPU).reshape((1, 10))))
        table_values_GPU = np.vstack((table_values_GPU, np.array(array_time_GPU).reshape((1, 10))))
    table_values_CPU = np.squeeze(table_values_CPU)
    table_values_GPU = np.squeeze(table_values_GPU)
    print(f"Таблица времени CPU : \n{table_values_CPU}\n")
    print(f"Таблица времени GPU : \n{table_values_GPU}\n")

    mas_CPU_time = [np.mean(table_values_CPU[:, i]) for i in range(table_values_CPU.shape[1])]
    mas_GPU_time = [np.mean(table_values_GPU[:, i]) for i in range(table_values_GPU.shape[1])]
    return mas_CPU_time, mas_GPU_time


if __name__ == "__main__":
    print("LR_3")
    threads_per_block = (4, 4)
    blocks_per_grid = (8, 8)

    sizes = np.linspace(200, 2000, 10, dtype=int)

    ABC = np.arange(256)

    mas_CPU_time, mas_GPU_time = start_calculation(sizes)
    mas_CPU_time = np.array(mas_CPU_time)
    mas_GPU_time = np.array(mas_GPU_time)
    print(f"Время CPU : \n{mas_CPU_time}\n")
    print(f"Время GPU : \n{mas_GPU_time}\n")

    print(mas_CPU_time / mas_GPU_time)
    plt.ticklabel_format(axis='x', style='plain')
    plt.plot(sizes, mas_CPU_time / mas_GPU_time, label='Ускорение', color='green', linestyle='-', linewidth=2)

    plt.title('Графики Ускорения')
    plt.xlabel('размерность')
    plt.ylabel('ускорение')
    plt.xticks(np.linspace(200, 2000, 10), sizes)
    plt.legend()
    plt.show()
