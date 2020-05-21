from oracles import BaseSmoothOracle, ChebyshevOscillator, TrigonometricOscillator, BigFuncOracle, LittleFuncOracle
from methods import BaseOptimizationMethod, PureGradientMethod
from methods import MAX_ITER, TOLERANCE
from momentum import Extrapolation, ArmijoRule

import numpy as np
from timeit import default_timer as timer

START_DIM = 3
END_DIM = 13


def run_test(oracle: BaseSmoothOracle, method: BaseOptimizationMethod, x_0, l_0):
    start = timer()
    last_x, last_value = method(x_0, l_0)
    end = timer()
    computational_time = end - start
    iterations, objective_values, _ = method.counters()

    oracle.reset()
    method.reset()

    return {"iterations": iterations, "computational_time": computational_time,
            "best_value": np.min(objective_values), "best_value_pos": np.argmin(objective_values),
            "last_value": last_value,
            }


def create_table(test_results, caption=None):
    table = "<table border=1>\n"
    if caption:
        table += f"<caption>{caption}</caption>\n"

    table += "<tr><td>Dimension</td><td>Iterations</td><td>Time</td>"
    table += "<td>Best value</td><td>Value index</td><td>Last value</td></tr>\n"

    for index, dim in enumerate(range(START_DIM, END_DIM + 1)):
        test = test_results[index]
        table += "<tr>\n"
        table += f"<td>{dim}</td>"
        table += f"<td>{test['iterations']}</td>"
        table += f"<td>{test['computational_time']}</td>"
        table += f"<td>{test['best_value']}</td>"
        table += f"<td>{test['best_value_pos']}</td>"
        table += f"<td>{test['last_value']}</td>"
        table += "</tr>"
    table += "</table>"
    return table


def main():
    runs = dict()
    functions = set()
    methods = set()
    start = timer()
    for dim in range(START_DIM, END_DIM + 1):
        print(f"Start dimension {dim}")
        start_value = np.ones(dim)
        start_value[0] = -1
        start_l = 1

        for oscillator in [ChebyshevOscillator(), TrigonometricOscillator()]:
            functions.add(str(oscillator))
            runs[str(oscillator)] = runs.get(str(oscillator), dict())
            print(f"\t{oscillator}")
            big_func = BigFuncOracle(dim, oscillator)
            f_hat = LittleFuncOracle(dim, big_func)

            extrapolation = Extrapolation(f_hat)
            armijo = ArmijoRule(f_hat)

            for optimization in [PureGradientMethod(f_hat), PureGradientMethod(f_hat, momentum=armijo),
                                 PureGradientMethod(f_hat, momentum=extrapolation)]:
                methods.add(str(optimization))
                runs[str(oscillator)][str(optimization)] = runs[str(oscillator)].get(str(optimization), list())
                print(f"\t\t{optimization}")
                test = run_test(f_hat, optimization, x_0=start_value, l_0=start_l)
                runs[str(oscillator)][str(optimization)].append(test)

        print("=========\n\n", end="")
    end = timer()
    computational_time = end - start

    functions = sorted(functions)
    methods = sorted(methods)
    tables = ""
    for func in functions:
        tables += "<table>\n"
        tables += f"<caption>{func}</caption>\n"
        tables += "<tr>"
        for method in methods:
            tables += f"<td align=center>{method}</td>"
        tables += "</tr>\n"
        tables += "<tr>"
        for method in methods:
            tables += f"\n<td>{create_table(runs[func][method])}</td>"
        tables += "</tr>\n"
        tables += "</table>\n<br><br>\n"

    with open("experiment.html", "w") as file:
        print('<!DOCTYPE HTML><html><head>\n'
              '\t<meta charset="utf-8">\n'
              '\t<title>FlexGNM results</title>\n'
              '\t<style>\n'
              '\t\ttable { border-collapse: collapse; }\n'
              '\t\ttable, th, td { border: 2px solid grey; }\n\t</style>\n'
              '</head><body>',
              file=file)
        print('<h1>Результаты проекта по курсу непрерывных оптимизаций</h1>\n'
              '<p>Проект выполнен студентами 3 курса Аникушиным Антоном и Борзенковой Софией</p>\n'
              f'<b>Максимальное число итераций метода: {MAX_ITER}</b><br>\n'
              '<b>Точность (<img src="http://latex.codecogs.com/svg.latex?\\epsilon" border="0"/>) вычислений методов: '
              f'{TOLERANCE}</b><br>'
              f'Полное время работы: {computational_time}', file=file)
        print(tables, file=file)
        print("</body></html>", file=file)


if __name__ == '__main__':
    main()
