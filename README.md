# High-Performance-Computing
**Лабораторные работы по HPC на CUDA с помощью pycuda.**

В репозитории находится три файла .py с лабами по HPC: Matrix Multiplication (hpc_lab0.py) / PI Calc (hpc_lab1.py) / Bilateral (hpc_lab2.py).
Графики ускорения cpu/gpu сделать хотелось, но не органично смотрится в репозитории. Простите.
Время указано в миллисекундах с округлением.

**Система:** 
CPU - Intel Core i5 5200U
GPU - NVIDIA GeForce 820M
  ___

### 1 Matrix Multiplication

Размерность матрицы | Время на GPU (мс)| Время на CPU (мс)| Ускорение
--- | --- | --- | --- 
256 |2  | 14 500 | 7250 
512 |12  | 123 100|10 258
1024 | 70 | 900 000 | 12 857
2048 | 470 | 7 650 000| 16 276

___
### 2 PI Calc 

Число точек  | Время на GPU (мс) | Время на CPU(мс) | Ускорение
--- | --- | --- | --- 
256 х 256 |100 | 400 | 4 
512 х 512 | 240 | 1300 | 5.4
1024 х 1024  | 300 | 1830 | 6.1
2048 х 2048 | 410 | 6850  | 16.7

___

## 3 Bilateral
 Время на GPU (мс) | Время на CPU(мс) | Ускорение 
 --- | ---  | --- 
420 | 17 500 | 41.7

Было:
![img](https://raw.githubusercontent.com/onlyDesuNOW/hpc_labs/master/input.bmp)

Стало:
![img](https://raw.githubusercontent.com/onlyDesuNOW/hpc_labs/master/output.bmp)
