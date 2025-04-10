import sys
from subprocess import Popen, PIPE
import matplotlib.pyplot as plt
from prettytable import PrettyTable

pythonpath1 = str(sys.executable) + " branchnbound.py "
pythonpath2 = str(sys.executable) + " perebor.py "
pythonpath4m5_1 = pythonpath1 + "m5n4.json"
pythonpath5m5_1 = pythonpath1 + "m5n5.json"
pythonpath6m5_1 = pythonpath1 + "m5n6.json"
pythonpath7m5_1 = pythonpath1 + "m5n7.json"
pythonpath8m5_1 = pythonpath1 + "m5n8.json"
pythonpath9m5_1 = pythonpath1 + "m5n9.json"

pythonpath4m5_2 = pythonpath2 + "m5n4.json"
pythonpath5m5_2 = pythonpath2 + "m5n5.json"
pythonpath6m5_2 = pythonpath2 + "m5n6.json"
pythonpath7m5_2 = pythonpath2 + "m5n7.json"
pythonpath8m5_2 = pythonpath2 + "m5n8.json"
pythonpath9m5_2 = pythonpath2 + "m5n9.json"

outn4m5_1, err = Popen(pythonpath4m5_1, shell=True, stdout=PIPE).communicate()
outn5m5_1, err = Popen(pythonpath5m5_1, shell=True, stdout=PIPE).communicate()
outn6m5_1, err = Popen(pythonpath6m5_1, shell=True, stdout=PIPE).communicate()
outn7m5_1, err = Popen(pythonpath7m5_1, shell=True, stdout=PIPE).communicate()
outn8m5_1, err = Popen(pythonpath8m5_1, shell=True, stdout=PIPE).communicate()
outn9m5_1, err = Popen(pythonpath9m5_1, shell=True, stdout=PIPE).communicate()

outn4m5_2, err = Popen(pythonpath4m5_2, shell=True, stdout=PIPE).communicate()
outn5m5_2, err = Popen(pythonpath5m5_2, shell=True, stdout=PIPE).communicate()
outn6m5_2, err = Popen(pythonpath6m5_2, shell=True, stdout=PIPE).communicate()
outn7m5_2, err = Popen(pythonpath7m5_2, shell=True, stdout=PIPE).communicate()
outn8m5_2, err = Popen(pythonpath8m5_2, shell=True, stdout=PIPE).communicate()
outn9m5_2, err = Popen(pythonpath9m5_2, shell=True, stdout=PIPE).communicate()
out1 = [
str(outn4m5_1, 'utf-8').splitlines()[-1],
str(outn5m5_1, 'utf-8').splitlines()[-1],
str(outn6m5_1, 'utf-8').splitlines()[-1],
str(outn7m5_1, 'utf-8').splitlines()[-1],
str(outn8m5_1, 'utf-8').splitlines()[-1],
str(outn9m5_1, 'utf-8').splitlines()[-1]
]
out2 = [
str(outn4m5_2, 'utf-8').splitlines()[-1],
str(outn5m5_2, 'utf-8').splitlines()[-1],
str(outn6m5_2, 'utf-8').splitlines()[-1],
str(outn7m5_2, 'utf-8').splitlines()[-1],
str(outn8m5_2, 'utf-8').splitlines()[-1],
str(outn9m5_2, 'utf-8').splitlines()[-1]
]

th5 = ['n', 'время выполнения методом ветвей и границ, сек', 'время выполнения методом простого перебора, сек']
td5 = list()
plotx = list()
plot1 = list()
plot2 = list()
i = 4
for el1, el2 in zip(out1, out2):
    it1, rec1, time1, Da1, Drec1 = el1.split()
    it2, rec2, time2, Da1, Drec2 = el2.split()
    td5.append(i)
    td5.append(float("{0:.3f}".format(float(time1))))
    td5.append(float("{0:.3f}".format(float(time2))))
    plotx.append(i)
    plot1.append(float("{0:.3f}".format(float(time1))))
    plot2.append(float("{0:.3f}".format(float(time2))))
    i = i + 1

columns = len(th5)
table5 = PrettyTable(th5)
td5_data = td5[:]
while td5_data:
    table5.add_row(td5_data[:columns])
    td5_data = td5_data[columns:]
print(table5)
plt.plot(plotx, plot1, '-', plotx, plot2, '--')
plt.show()