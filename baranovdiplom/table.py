import sys
from subprocess import Popen, PIPE
from prettytable import PrettyTable

pythonpath = str(sys.executable) + " branchnbound.py "
pythonpath10m5 = pythonpath + "m5n10.json"
pythonpath11m5 = pythonpath + "m5n11.json"
pythonpath12m5 = pythonpath + "m5n12.json"
pythonpath13m5 = pythonpath + "m5n13.json"
pythonpath14m5 = pythonpath + "m5n14.json"
pythonpath15m5 = pythonpath + "m5n15.json"
pythonpath10m7 = pythonpath + "m7n10.json"
pythonpath11m7 = pythonpath + "m7n11.json"
pythonpath12m7 = pythonpath + "m7n12.json"
pythonpath13m7 = pythonpath + "m7n13.json"
pythonpath14m7 = pythonpath + "m7n14.json"
pythonpath15m7 = pythonpath + "m7n15.json"

print(pythonpath)
outn10m5, err = Popen(pythonpath10m5, shell=True, stdout=PIPE).communicate()
outn11m5, err = Popen(pythonpath11m5, shell=True, stdout=PIPE).communicate()
outn12m5, err = Popen(pythonpath12m5, shell=True, stdout=PIPE).communicate()
outn13m5, err = Popen(pythonpath13m5, shell=True, stdout=PIPE).communicate()
outn14m5, err = Popen(pythonpath14m5, shell=True, stdout=PIPE).communicate()
outn15m5, err = Popen(pythonpath15m5, shell=True, stdout=PIPE).communicate()

outn10m7, err = Popen(pythonpath10m7, shell=True, stdout=PIPE).communicate()
outn11m7, err = Popen(pythonpath11m7, shell=True, stdout=PIPE).communicate()
outn12m7, err = Popen(pythonpath12m7, shell=True, stdout=PIPE).communicate()
outn13m7, err = Popen(pythonpath13m7, shell=True, stdout=PIPE).communicate()
outn14m7, err = Popen(pythonpath14m7, shell=True, stdout=PIPE).communicate()
outn15m7, err = Popen(pythonpath15m7, shell=True, stdout=PIPE).communicate()
print("aa")
out5 = [
str(outn10m5, 'utf-8').splitlines()[-1],
str(outn11m5, 'utf-8').splitlines()[-1],
str(outn12m5, 'utf-8').splitlines()[-1],
str(outn13m5, 'utf-8').splitlines()[-1],
str(outn14m5, 'utf-8').splitlines()[-1],
str(outn15m5, 'utf-8').splitlines()[-1]
]
out7 = [
str(outn10m7, 'utf-8').splitlines()[-1],
str(outn11m7, 'utf-8').splitlines()[-1],
str(outn12m7, 'utf-8').splitlines()[-1],
str(outn13m7, 'utf-8').splitlines()[-1],
str(outn14m7, 'utf-8').splitlines()[-1],
str(outn15m7, 'utf-8').splitlines()[-1]
]

th5 = ['n', 'количество рекордных точек','время выполнения, сек','абсолютная погрешность','относительная погрешность, %']
th7 = ['n', 'количество рекордных точек','время выполнения, сек','абсолютная погрешность','относительная погрешность, %']
td5 = list()
td7 = list()

for index, el in enumerate(out5):
    it, rec, time, Da, Drec = el.split()
    td5.append(index+10)
    td5.append(rec)
    td5.append(float("{0:.3f}".format(float(time))))
    td5.append(int(Drec)-int(Da))
    td5.append(float("{0:.3f}".format(float(((int(Drec)-int(Da))/int(Da))*100))))

for index, el in enumerate(out7):
    it, rec, time, Da, Drec = el.split()
    td7.append(index+10)
    td7.append(rec)
    td7.append(float("{0:.3f}".format(float(time))))
    td7.append(int(Drec)-int(Da))
    td7.append(float("{0:.3f}".format(float(((int(Drec)-int(Da))/int(Da))*100))))

columns = len(th5)
table5 = PrettyTable(th5)
td5_data = td5[:]
while td5_data:
    table5.add_row(td5_data[:columns])
    td5_data = td5_data[columns:]
print("\nm=5")
print(table5)

table7 = PrettyTable(th7)
td7_data = td7[:]
while td7_data:
    table7.add_row(td7_data[:columns])
    td7_data = td7_data[columns:]
print("\nm=7")
print(table7)
