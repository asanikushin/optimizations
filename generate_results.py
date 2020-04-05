from oracles import BaseSmoothOracle, ChebyshevOscillator, TrigonometricOscillator, BigFuncOracle, LittleFuncOracle
from methods import BaseOptimizationMethod, PureGradientMethod
from methods import MAX_ITER, TOLERANCE
from momentum import Extrapolation, ArmijoRule

import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
import base64
from io import BytesIO


def run_test(oracle: BaseSmoothOracle, method: BaseOptimizationMethod, x_0, l_0, save=None):
    start = timer()
    last_x, last_value = method(x_0, l_0)
    end = timer()
    computational_time = end - start
    oracle_call = oracle.counters()
    iterations, objective_values, gradient_norm = method.counters()

    oracle.reset()
    method.reset()

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(objective_values)
    plt.title(f"Objective: {oracle}, {method}")
    plt.subplot(2, 1, 2)
    plt.plot(gradient_norm)
    plt.title(f"Gradient: {oracle}, {method}")

    plot_file = BytesIO()
    plt.savefig(plot_file, format='png')
    plot_file.seek(0)  # rewind to beginning of file

    data_png = str(base64.b64encode(plot_file.getvalue()))[2:-1]

    # save results in html
    # else show plots inline
    if save:
        # method counter
        print(f"<p>Суммарное время работы: {round(computational_time, 3)} секунд</p>", file=save)
        print(f"<p>Лучшее значение целевой функции: {np.min(objective_values)}<br>", file=save)
        print(f"Номер шага с лучшим значением: {np.argmin(objective_values)}</p>", file=save)

        print(f"<p>Число итераций метода: {iterations}</p>", file=save)

        # oracle counter
        print(f"<table border=1><caption>Количество вызовов оракула</caption>", file=save)
        print(f"<tr><td></td><th>{f_hat_svg}</th><th>F</th><th>{phi_svg}</th></tr>", file=save)
        print(f"<tr><th>Значение</th><td>{oracle_call['value']}</td><td>{oracle_call['F oracle']['value']}</td>"
              f"<td>{oracle_call['F oracle']['oscillator oracle']['value']}</td></tr>", file=save)
        print(f"<tr><th>Градиент</th><td>{oracle_call['gradient']}</td><td>{oracle_call['F oracle']['gradient']}</td>"
              f"<td>{oracle_call['F oracle']['oscillator oracle']['gradient']}</td></tr>", file=save)
        print(f"</table>", file=save)

        # plots
        print(f"<br><details><summary>Графики зависимости значения целевой функции и нормы градиента от шага</summary>",
              file=save)
        print(f'<img src="data:image/png;base64,{data_png}"/>', file=save)
        print(f"</details>", file=save, flush=True)
    else:
        print("\t\tlast x:", last_x.tolist())
        plt.show()

    return {"iterations": iterations, "oracle calls": oracle_call, "computational time": computational_time,
            "best_value": np.min(objective_values),
            "best_value_pos": np.argmin(objective_values),
            }


DIMENSION = 20
start_value = np.ones(DIMENSION)
start_value[0] = -1
start_l = 0.001

file_name = f"result {DIMENSION}.html"
file = open(file_name, "w")

if file:
    print(f"Save at '{file_name}'")
    print('<!DOCTYPE HTML><html><head>'
          '<meta charset="utf-8">'
          '<title>FlexGNM results</title>'
          '<script type="text/javascript" src="http://latex.codecogs.com/latexit.js"></script></head><body>', file=file)
    print(f"<h1>Результаты проекта по курсу непрерывных оптимизаций</h1>", file=file)
    print(f"<p>Проект выполнен студентами 3 курса Аникушиным Антоном и Борзенковой Софией</p>", file=file)
    print(f"<div><b>Размерность функции: {DIMENSION}</b><br>", file=file)
    print(f"<b>Максимальное число итераций метода: {MAX_ITER}</b><br>", file=file)
    print(f'<b>Точность (<img src="http://latex.codecogs.com/svg.latex?\\epsilon" border="0"/>) вычислений методов: '
          f'{TOLERANCE}</b><br>', file=file)

    f_hat_svg = '<img src="http://latex.codecogs.com/svg.latex?f_1" border="0"/>'
    phi_svg = '<img src="http://latex.codecogs.com/svg.latex?\\phi" border="0"/>'

    print(f'<b>Начальная точка для оптимизации (<img src="http://latex.codecogs.com/svg.latex?x_0" border="0"/>): '
          f'{start_value.astype(int)}</b><br>', file=file)
    print(f'<b>Начальная длина шага для оптимизации (<img src="http://latex.codecogs.com/svg.latex?L_0" border="0"/>): '
          f'{start_l}</b>', file=file)
    print(f"</div>", file=file)
else:
    print(f"Dimension: {DIMENSION}")
    print(f"Maximum iteration: {MAX_ITER}")
    print(f"Epsilon: {TOLERANCE}")
    print(f"Start point: {start_value.astype(int)}")
    print(f"L_0: {start_l}")
    print()

for oscillator in [ChebyshevOscillator(), TrigonometricOscillator()]:
    if file:
        print(f"<h2>{oscillator}</h2>", file=file)
    else:
        print(f"{oscillator}")
    big_func = BigFuncOracle(DIMENSION, oscillator)
    f_hat = LittleFuncOracle(DIMENSION, big_func)

    extrapolation = Extrapolation(f_hat)
    armijo = ArmijoRule(f_hat)
    for optimization in [PureGradientMethod(f_hat), PureGradientMethod(f_hat, momentum=armijo),
                         PureGradientMethod(f_hat, momentum=extrapolation)]:
        if file:
            print(f"<h3>{optimization}</h3>", file=file)
        else:
            print("\t", optimization, sep="")
        results = run_test(f_hat, optimization, start_value, start_l, file)

        if not file:
            print("\t\t", results, sep="", end="\n\n")
    if not file:
        print("=========\n\n", end="")

if file:
    print("</body></html>", file=file)
    file.close()
