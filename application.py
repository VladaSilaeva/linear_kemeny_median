import tkinter as tk
from tkinter import messagebox, filedialog

from PIL import Image, ImageTk

from matrix_solve import main, special_case


class Application:
    def __init__(self, master):
        self.master = master # окно, от которого наследуется приложение
        self.var = tk.StringVar() # ввод "линейный" или "матричный"
        self.alternatives = 0 # n - количество альтернатив
        self.experts = 0 # m - количество экспертов
        self.data = None # r или R в зависимости от ввода - предпочтения экспертов
        self.c = None # веса экспертов
        self.photo = None # хранилище для картинки графа
        self.solver = None # решатель

        self.master.geometry("400x300")  # размер окна
        self.center_window(400, 400) # центрирование
        self.format_choice() # запуск первого окна с выбором формата ввода

    def center_window(self, width, height):
        """Центрирование окна на экране"""
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        x_coordinate = (screen_width / 2) - (width / 2)
        y_coordinate = (screen_height / 2) - (height / 2)
        self.master.geometry('%dx%d+%d+%d' % (width, height, x_coordinate, y_coordinate))

    def format_choice(self):
        """Окно1: Выбор формата ввода: линейный/матричный"""
        self.experts = self.alternatives = 0
        self.data = self.c = self.solver = None
        for widget in self.master.winfo_children():
            widget.destroy()
        self.master.title("Выбор формата")

        # Первый экран выбора формата
        label_format = tk.Label(self.master, text="Формат ввода:")
        label_format.pack(pady=10)

        self.var = tk.StringVar(value="линейный")  # По умолчанию выбран ввод линейных порядков
        linear_radio = tk.Radiobutton(self.master, text="Ввод линейных порядков", variable=self.var, value="линейный")
        matrix_radio = tk.Radiobutton(self.master, text="Ввод матриц бинарных отношений", variable=self.var,
                                      value="матричный")
        linear_radio.pack()
        matrix_radio.pack()

        frame_buttons = tk.Frame(self.master)
        frame_buttons.pack(side="bottom", fill="x")
        text_hint = "Выбор типа ввода: линейные порядки или матрицы (более общий случай)"
        button_hint = tk.Button(frame_buttons, text="Справка",
                                command=lambda: messagebox.showinfo("Справка", text_hint))
        button_next = tk.Button(frame_buttons, text="Далее", command=lambda: self.size_choice())
        button_next.pack(side="right", padx=10)
        button_hint.pack(side="right", padx=10)

    def size_choice(self):
        """Окно2: Ввод количества экспертов и количества альтернатив
            Возможно сразу открыть входной файл с предпочтениями"""
        self.experts = self.alternatives = 0
        self.data = self.c = self.solver = None
        for widget in self.master.winfo_children():
            widget.destroy()
        self.master.title(
            f"Ввод количества экспертов и альтернатив")

        label_selected_format = tk.Label(self.master, text=f"Ваш выбор: {self.var.get()}")
        label_selected_format.pack(pady=10)
        frame_alternatives = tk.Frame(self.master)
        frame_experts = tk.Frame(self.master)
        frame_buttons = tk.Frame(self.master)

        label_alternatives = tk.Label(frame_alternatives, text="Количество альтернатив:")
        entry_alternatives = tk.Entry(frame_alternatives)
        label_experts = tk.Label(frame_experts, text="Количество экспертов:")
        entry_experts = tk.Entry(frame_experts)

        frame_alternatives.pack(fill="x")
        frame_experts.pack(fill="x")
        frame_buttons.pack(side="bottom", fill="x")
        label_alternatives.pack(side="left", padx=10)
        entry_alternatives.pack(side="left")
        label_experts.pack(side="left", padx=10)
        entry_experts.pack(side="left", padx=12)

        load_button = tk.Button(self.master, text="Загрузить данные из файла", command=self.load_from_file)
        load_button.pack(padx=10)

        button_back = tk.Button(frame_buttons, text="Назад", command=lambda: self.format_choice())
        text_hint = "Количество экспертов и количество альтернатив - натуральные числа\n"
        text_hint+="Возможно сразу ввести предпочтения из текстового файла.\n"
        text_hint += """В случае линейного ввода, данные в файле имеют вид:
    3 4                 # n и m через пробел или пустая строка (можно вычислить n,m по следующему вводу))
    1.0 0.5 1.5 1.0     # m весов (дробные числа) через пробел или пустая строка если они равны c_1=...=c_m=1.0
    1 2 3               # мнение 1-го эксперта (перестановка индексов i=1,2,...,n записанная через пробел)
    2 3 1               # мнение 2-го эксперта
    1 3 2
    3 2 1               # мнение m-го эксперта
                        # после пустой строки могут идти комментарии"""
        text_hint += """\n\nВ случае матричного ввода, данные в файле имеют вид:
    2 2                 # n и m через пробел или пустая строка (можно вычислить n,m по следующему вводу))
    1.5 0.5             # m весов (дробные числа) через пробел или пустая строка если они равны c_1=...=c_m=1.0
    1 1                 # мнение 1-го эксперта 1 строка
    0 0                 # мнение 1-го эксперта n строка
    1 0.5               # мнение m-го эксперта
    0.5 1    
                        # после пустой строки могут идти комментарии"""
        button_hint = tk.Button(frame_buttons, text="Справка",
                                command=lambda: messagebox.showinfo("Справка", text_hint))
        button_next = tk.Button(frame_buttons, text="Далее", command=lambda: next_to_preferences_input())

        button_next.pack(side="right", padx=10)
        button_hint.pack(side="right", padx=10)
        button_back.pack(side="right", padx=10)

        def next_to_preferences_input():
            print("next")
            try:
                self.experts = int(entry_experts.get())
                self.alternatives = int(entry_alternatives.get())

                if not (self.validate_input_n(entry_experts) and self.validate_input_n(entry_alternatives)):
                    raise ValueError

                if self.var.get() == "линейный":
                    self.linear_input()
                elif self.var.get() == "матричный":
                    self.matrix_input()
            except ValueError:
                self.experts = self.alternatives = 0
                messagebox.showerror("Ошибка", "Неверное значение! Введите целые числа больше нуля.")

    def linear_input(self, data=None):
        """Окно3.1: Линейный ввод предпочтений экспертов.
            Возможно ввести данные как с клавиатуры, так и из файла"""
        self.solver = None
        for widget in self.master.winfo_children():
            widget.destroy()
        self.master.title(
            f"Ввод линейных порядков для {self.experts} экспертов и {self.alternatives} альтернатив")

        canvas = tk.Canvas(self.master)
        scrollbar = tk.Scrollbar(self.master, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        entries = []

        frame = tk.Frame(scrollable_frame)
        frame.pack(fill='x', padx=10, pady=5)

        label = tk.Label(frame, text=f"Количество альтернатив: {self.alternatives}")
        label.pack(side='top')

        label = tk.Label(frame, text=f"Веса экспертов:")
        label.pack(side='left')

        experts_coef = tk.StringVar(value="1 " * self.experts)
        entry_c = tk.Entry(frame, textvariable=experts_coef)
        entry_c.pack(side='right', fill='x', expand=True)
        entry_c.bind("<FocusOut>", lambda event, e=entry_c: self.validate_input_c(e))

        lin = []
        for i in range(self.experts):
            frame = tk.Frame(scrollable_frame)
            frame.pack(fill='x', padx=10, pady=5)

            label = tk.Label(frame, text=f"Эксперт {i + 1}:")
            label.pack(side='left')

            lin.append(tk.StringVar(value=""))
            entry = tk.Entry(frame, textvariable=lin[i])
            entry.pack(side='right', fill='x', expand=True)
            entry.bind("<FocusOut>", lambda event, e=entry: self.validate_input_lin(e))

            entries.append(entry)

        def next_step():
            try:
                nonlocal entries, entry_c

                for entry in entries:
                    if not self.validate_input_lin(entry):
                        raise ValueError
                r = [[int(e) for e in entry.get().split()] for entry in entries]
                if not (self.validate_input_c(entry_c)):
                    messagebox.showwarning("Предупреждение", "Неверные веса заменены на 1")

                    c = [1.0] * self.experts
                else:
                    c = [float(ci) for ci in entry_c.get().split()]
                print(r)
                print(c)
                self.data = r
                self.c = c
                self.solve_choice()
            except ValueError:
                messagebox.showerror("Ошибка", "Неверное значение! Введите заново.")

        def set_linear(r, c, n, m):
            print("------------c=", c)
            experts_coef.set(' '.join(map(str, c)))
            for k in range(self.experts):
                if k < m:
                    lin[k].set(" ".join(map(str, r[k])))
                else:
                    lin[k].set("")

        if data is not None:
            print('data', data)
            set_linear(*data)

        load_button = tk.Button(scrollable_frame, text="Загрузить линейные порядки из файла",
                                command=lambda: set_linear(*self.load_matrix_from_file()))
        load_button.pack(padx=10)

        frame_buttons = tk.Frame(scrollable_frame)
        frame_buttons.pack(side="bottom", fill="x")
        button_next = tk.Button(frame_buttons, text="Далее", command=next_step)
        text_hint = f'Вводятся веса экспертов: \n\tm={self.experts} дробных чисел написанных через пробел '
        text_hint+=f"(при m=3 '1.0 0.5 1.5'), в сумме равных m={self.experts}.\n А также {self.experts} мнений "
        text_hint+=f"экспертов в виде линейных порядков: \n\tперестановок чисел от 1 до {self.alternatives} - "
        text_hint+="натуральные числа записанные через пробел (при m=3,n=4 '1 2 3 4', '4 2 3 1', '1 3 4 2')"
        text_hint += """\nВозможен ввод из файла, данные в файле имеют вид:
    3 4                 # n и m через пробел или пустая строка (можно вычислить n,m по следующему вводу))
    1.0 0.5 1.5 1.0     # m весов (дробные числа) через пробел или пустая строка если они равны c_1=...=c_m=1.0
    1 2 3               # мнение 1-го эксперта (перестановка индексов i=1,2,...,n записанная через пробел)
    2 3 1               # мнение 2-го эксперта
    1 3 2
    3 2 1               # мнение m-го эксперта
                        # после пустой строки могут идти комментарии"""
        button_hint = tk.Button(frame_buttons, text="Справка",
                                command=lambda: messagebox.showinfo("Справка", text_hint))
        button_back = tk.Button(frame_buttons, text="Назад", command=self.size_choice)

        button_next.pack(side="right", padx=10)
        button_hint.pack(side="right", padx=10)
        button_back.pack(side="right", padx=10)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def matrix_input(self, data=None):

        """Окно3.1: Матричный ввод предпочтений экспертов.
            Возможно ввести данные как с клавиатуры, так и из файла"""
        self.data = self.c = self.solver = None
        print("matrix")
        print(data)
        for widget in self.master.winfo_children():
            widget.destroy()
        self.master.title(
            f"Ввод матриц бинарных отношений для {self.experts} экспертов и {self.alternatives} альтернатив")

        canvas = tk.Canvas(self.master)
        scrollbar = tk.Scrollbar(self.master, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        matrices_entries = []
        matrices = []

        frame = tk.Frame(scrollable_frame)
        frame.pack(fill='x', padx=10, pady=5)

        label = tk.Label(frame, text=f"Количество альтернатив: {self.alternatives}")
        label.pack(side='top')

        label = tk.Label(frame, text=f"Веса экспертов:")
        label.pack(side='left')

        experts_coef = tk.StringVar(value="1 " * self.experts)
        entry_c = tk.Entry(frame, textvariable=experts_coef)
        entry_c.pack(side='right', fill='x', expand=True)
        entry_c.bind("<FocusOut>", lambda event, e=entry_c: self.validate_input_c(e))

        for k in range(self.experts):
            frame = tk.Frame(scrollable_frame)
            frame.pack(fill='x', padx=10, pady=5)

            label = tk.Label(frame, text=f"Матрица эксперта {k + 1}:")
            label.pack(side='top')

            matrix_entries = []
            matrix = [[tk.StringVar(value="0") for _ in range(self.alternatives)] for _ in range(self.alternatives)]
            for i in range(self.alternatives):
                row_frame = tk.Frame(frame)
                row_frame.pack(fill='x', side='top')

                row_entries = []
                for j in range(self.alternatives):
                    entry = tk.Entry(row_frame, textvariable=matrix[i][j], width=4)
                    entry.pack(side='left', padx=2)
                    entry.bind('<Right>', lambda event, x=i, y=j, num=k: move_right(x, y, num))
                    entry.bind('<Left>', lambda event, x=i, y=j, num=k: move_left(x, y, num))
                    entry.bind('<Up>', lambda event, x=i, y=j, num=k: move_up(x, y, num))
                    entry.bind('<Down>', lambda event, x=i, y=j, num=k: move_down(x, y, num))
                    entry.bind("<FocusOut>", lambda event, e=entry: self.validate_input_elem(e))
                    row_entries.append(entry)
                matrix_entries.append(row_entries)
            matrices_entries.append(matrix_entries)
            matrices.append(matrix)

        def move_right(row, col, num):
            new_col = col + 1
            if new_col >= self.alternatives:
                return move_down(row, 0, num)
            matrices_entries[num][row][new_col].focus_set()

        def move_left(row, col, num):
            new_col = col - 1
            if new_col < 0:
                return move_up(row, self.alternatives - 1, num)
            matrices_entries[num][row][new_col].focus_set()

        def move_up(row, col, num):
            new_row = row - 1
            new_num = num
            if new_row < 0:
                new_row = self.alternatives - 1
                new_num = num - 1
            if new_num < 0:
                return
            matrices_entries[new_num][new_row][col].focus_set()

        def move_down(row, col, num):
            new_row = row + 1
            new_num = num
            if new_row >= self.alternatives:
                new_row = 0
                new_num = num + 1
            if new_num >= self.experts:
                return
            matrices_entries[new_num][new_row][col].focus_set()

        def next_step():
            try:
                nonlocal entry_c, matrices_entries

                for matrix_enteries in matrices_entries:
                    for i in range(self.alternatives):
                        for j in range(self.alternatives):
                            if not self.validate_input_elem(matrix_enteries[i][j]):
                                raise ValueError
                R = [[[int(cell.get()) for cell in row] for row in matrix_enteries] for matrix_enteries in
                     matrices_entries]
                if not (self.validate_input_c(entry_c)):
                    messagebox.showerror("Ошибка", "Неверные веса заменены на 1")
                    c = [1.0] * self.experts
                else:
                    c = [float(ci) for ci in entry_c.get().split()]
                print(R)
                print(c)
                self.data = R
                self.c = c
                self.solve_choice()
            except ValueError:
                messagebox.showerror("Ошибка", "Неверное значение! Введите заново.")

        def set_matrix(R, c, n, m):
            experts_coef.set(' '.join(map(str, c)))
            for k in range(self.experts):
                for i in range(n):
                    for j in range(n):
                        if k < m:
                            matrices[k][i][j].set(R[n * k + i][j])
                        else:
                            matrices[k][i][j].set("0")

        if data is not None:
            set_matrix(*data)

        load_button = tk.Button(scrollable_frame, text="Загрузить матрицу из файла",
                                command=lambda: set_matrix(*self.load_matrix_from_file()))
        load_button.pack(padx=10)

        frame_buttons = tk.Frame(scrollable_frame)
        frame_buttons.pack(side="bottom", fill="x")
        button_next = tk.Button(frame_buttons, text="Далее", command=next_step)
        text_hint = f'Вводятся веса экспертов: {self.experts} дробных чисел написанных через пробел ("1.0 0.5 1.5"), '
        text_hint+= f'в сумме равных {self.experts}.\n А также {self.experts} мнений экспертов в виде матриц размерности '
        text_hint+= f'{self.alternatives}x{self.alternatives}.'
        text_hint+="\nКнопки стрелок на клавиатуре помогают переместиться между ячейками матриц"
        text_hint += """\nВозможен ввод из файла, данные в файле имеют вид:
    2 2                 # n и m через пробел или пустая строка (можно вычислить n,m по следующему вводу))
    1.5 0.5             # m весов (дробные числа) через пробел или пустая строка если они равны c_1=...=c_m=1.0
    1 1                 # мнение 1-го эксперта 1 строка
    0 0                 # мнение 1-го эксперта n строка
    1 0.5               # мнение m-го эксперта
    0.5 1    
                        # после пустой строки могут идти комментарии"""

        button_hint = tk.Button(frame_buttons, text="Справка",
                                command=lambda: messagebox.showinfo("Справка", text_hint))
        button_back = tk.Button(frame_buttons, text="Назад", command=self.size_choice)

        button_next.pack(side="right", padx=10)
        button_hint.pack(side="right", padx=10)
        button_back.pack(side="right", padx=10)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def validate_input_n(self, entry, l=0, r=None):
        try:
            value = int(entry.get())
            if value <= l:
                raise ValueError
            elif r is not None:
                if value > r:
                    raise ValueError
            entry.config(bg="white")
            return True
        except ValueError:
            entry.config(bg="pink")
            return False

    def validate_input_elem(self, entry):
        try:
            value = int(entry.get())
            if value >= 0:
                entry.config(bg="white")
                return True
            else:
                raise ValueError
        except ValueError:
            entry.config(bg="pink")
            return False

    def validate_input_lin(self, entry):
        try:
            wrong_input = False
            values = [int(e) for e in entry.get().split()]
            values.sort()
            if values != list(range(1, self.alternatives + 1)):
                wrong_input = True
            if wrong_input:
                raise ValueError
            entry.config(bg="white")
            return True
        except ValueError:
            entry.config(bg="pink")
            return False

    def validate_input_c(self, entry):
        try:
            wrong_input = False
            values = [float(e) for e in entry.get().split()]
            if len(values) != self.experts:
                wrong_input = True
            if sum(values) != self.experts:  # float==int
                wrong_input = True
            if wrong_input:
                raise ValueError
            entry.config(bg="white")
            return True
        except ValueError:
            entry.config(bg="pink")
            return False

    def load_from_file(self):
        if self.var.get() == "линейный":
            r, c, n, m = self.load_lin_from_file()
            self.alternatives = n
            self.experts = m
            self.linear_input([r, c, n, m])
        elif self.var.get() == "матричный":
            R, c, n, m = self.load_matrix_from_file()
            self.alternatives = n
            self.experts = m
            self.matrix_input([R, c, n, m])

    def load_matrix_from_file(self):
        file_path = filedialog.askopenfilename(filetypes=[('Text files', '*.txt')])
        if file_path:
            R = []
            c = []
            n, m = 0, 0
            with open(file_path, 'r') as f:
                nm_line = f.readline()
                c_line = f.readline()
                for line in f.readlines():
                    if len(line.strip()):
                        R.append(line.split())
                    else:
                        break
                f.close()
                if len(nm_line.strip()) == 2:
                    n, m = [int(i) for i in nm_line.split()]
                else:
                    n = len(R[0])
                    m = len(R) // n
                if self.alternatives and n != self.alternatives:
                    messagebox.showerror("Ошибка", "Размер матрицы в файле не соответствует введенному ранее!")
                    return
                if len(c_line.strip()):
                    c = [float(i) for i in c_line.split()]
                    if len(c) != m or sum(c) != m:
                        print('wrong c')
                        c = [1.0] * m
                else:
                    c = [1.0] * m
                print("load matr from file", R, c, n, m)
                return R, c, n, m

    def load_lin_from_file(self):
        file_path = filedialog.askopenfilename(filetypes=[('Text files', '*.txt')])
        if file_path:
            r = []
            c = []
            n, m = 0, 0
            with open(file_path, 'r') as f:
                nm_line = f.readline()
                c_line = f.readline()
                for line in f.readlines():
                    if len(line.strip()):
                        r.append(line.split())
                    else:
                        break
                f.close()
                if len(nm_line.strip()) == 2:
                    n, m = [int(i) for i in nm_line.split()]
                else:
                    n = len(r[0])
                    m = len(r)
                if self.alternatives and n != self.alternatives:
                    messagebox.showerror("Ошибка",
                                         "Количество альтернатив в файле не соответствует введенному ранее!")
                    return
                if len(c_line.strip()):
                    c = [float(i) for i in c_line.split()]
                    if len(c) != m or sum(c) != m:
                        messagebox.showerror("Внимание",
                                             "Введены некорректные веса для экспертов. Они заменены на веса по умолчанию (c_i=1.0, i=1,m)")

                        print('wrong c')
                        c = [1.0] * m
                        print('new c', c)
                else:
                    print('c=', c)
                    c = [1.0] * m
                    # messagebox.showerror("Внимание","Не были введены веса для экспертов. Веса заменены на веса по умолчанию (c_i=1.0, i=1,m)")
                print("load lin from file", r, c, n, m)
                return r, c, n, m

    def solve_choice(self):
        """Окно4: выбор метода решения:
        решение - обычное решение методом ветвей и границ
        решение с ускорением - решение с предварительным вычислением рекорда
        решение в частном случае - решение случая нечетного m, линейных предпочтений с одинаковыми весами
        закрепление первого
        закрепление последнего"""
        self.solver = None
        for widget in self.master.winfo_children():
            widget.destroy()
        self.master.title("Выбор решения")
        lin = self.var.get() == "линейный"

        def solve():
            ans_str, ans, self.solver, tree_str = main(self.data, self.c, self.alternatives, self.experts,
                                                       solver=self.solver, lin=lin)
            self.display_results("solve\n" + ans_str + "\nДерево решений:\n" + tree_str, "обычное")

        def fast_solve():
            ans_str, ans, self.solver, tree_str = main(self.data, self.c, self.alternatives, self.experts,
                                                       solver=self.solver, lin=lin, forced_down=True)
            self.display_results("fast solve\n" + ans_str + "\nДерево решений:\n" + tree_str, "с ускорением")

        def special_solve():
            ans_str, ans = special_case(self.data, self.c, self.alternatives, self.experts)
            self.display_results("special solve\n" + ans_str, "в частном случае")

        def first():
            ans_strs = ""
            ans = []
            for i in range(self.alternatives):
                ans_strs += f'first={i + 1}\n'
                ans_str, a, self.solver, tree_str = main(self.data, self.c, self.alternatives, self.experts,
                                                         solver=self.solver, first=i, lin=True)
                ans_strs += ans_str + '\n'
                ans.append(a)
            ans_strs += '\n'.join([f'r_first({i + 1})={ans[i]}' for i in range(self.alternatives)])
            r=dict()
            for i in range(self.alternatives):
                if ans[i][0] not in r.keys():
                    r[ans[i][0]]=[]
                r[ans[i][0]].append(i+1)
            r_i=sorted(r.items())
            print(r_i)
            ans_strs +='\n'+'\n'.join([f'r{i}={d}' for d,i in r_i])
            self.display_results(ans_strs, "первый закреплен")

        def last():
            ans_strs = ""
            ans = []
            for i in range(self.alternatives):
                ans_strs += f'last={i + 1}\n'
                ans_str, a, self.solver, tree_str = main(self.data, self.c, self.alternatives, self.experts,
                                                         solver=self.solver, last=i, lin=True)
                ans_strs += ans_str + '\n'
                ans.append(a)
            ans_strs += '\n'.join([f'r_last({i + 1})={ans[i]}' for i in range(self.alternatives)])
            self.display_results(ans_strs, "последний закреплен")

        def first_last():

            try:
                nonlocal entry_first, entry_last
                if not (self.validate_input_n(entry_first, 0, self.alternatives) and self.validate_input_n(entry_last,
                                                                                                           0,
                                                                                                           self.alternatives)):
                    raise ValueError
                f = int(entry_first.get())
                l = int(entry_last.get())
                ans_str1, ans1, self.solver, tree_str = main(self.data, self.c, self.alternatives, self.experts,
                                                             solver=self.solver, first=f - 1, last=l - 1,
                                                             lin=lin)
                ans_str2, ans2, self.solver, tree_str = main(self.data, self.c, self.alternatives, self.experts,
                                                             solver=self.solver, first=l - 1, last=f - 1,
                                                             lin=lin)
                ans_str = f'first={f}, last={l}\n{ans_str1} first={l}, last={f}\n{ans_str2}\n'
                if ans1[0] < ans2[0]:
                    ans_str += f'{f} лучше {l} в {ans2[0] / ans1[0]} раз'
                else:
                    ans_str += f'{l} лучше {f} в {ans1[0] / ans2[0]} раз'
                self.display_results(ans_str, f"сравнение {f} и {l}")
            except ValueError:
                messagebox.showerror("Ошибка", "Неверное значение! Введите заново.")

        def to_preferences_input():
            if lin:
                self.linear_input([self.data, self.c, self.alternatives, self.experts])
            else:
                self.matrix_input([self.data, self.c, self.alternatives, self.experts])

        self.master.title("Параметры")
        solve_frame = tk.Frame(self.master)
        # Обычное решение
        button_solve = tk.Button(solve_frame, text="Решить", command=solve)
        # Решение с ускорением
        button_fast_solve = tk.Button(solve_frame, text="Решить с ускорением", command=fast_solve)
        # Решение в частном случае
        button_special_solve = tk.Button(solve_frame, text="Решить в частном случае", command=special_solve)
        solve_frame.pack()
        button_solve.pack(side="left", padx=10)
        button_fast_solve.pack(side="left", padx=10)
        button_special_solve.pack(side="left", padx=10)

        button_first = tk.Button(self.master, text="Закрепить первую альтернативу", command=first)
        button_last = tk.Button(self.master, text="Закрепить последнюю альтернативу", command=last)
        button_first.pack()
        button_last.pack()

        compare_frame = tk.Frame(self.master)
        entry_first = tk.Entry(compare_frame)
        entry_first.bind("<FocusOut>", lambda event, e=entry_first: self.validate_input_n(e, 0, self.alternatives))
        entry_last = tk.Entry(compare_frame)
        entry_last.bind("<FocusOut>", lambda event, e=entry_last: self.validate_input_n(e, 0, self.alternatives))
        button_first_last = tk.Button(compare_frame, text="Сравнить две альтернативы", command=first_last)
        compare_frame.pack()
        button_first_last.pack(side="left", padx=10)
        entry_first.pack(side="left", padx=10)
        entry_last.pack(side="left", padx=10)

        # Назад к введению данных
        button_back = tk.Button(self.master, text="Назад", command=to_preferences_input)
        button_back.pack(padx=10, pady=10)

        is_special_case = False
        if lin:
            if self.experts % 2:
                is_special_case = True
                for i in self.c:
                    if i != 1:
                        is_special_case = False
                        break
        if not is_special_case:
            button_special_solve.config(state="disabled")

    def display_results(self, ans_str, solution_type):
        def save():
            file_path = filedialog.asksaveasfilename(filetypes=[('Text files', '*.txt')])
            if file_path:
                f = open(file_path, 'w', encoding='UTF-8')
                f.write(f'{self.alternatives} {self.experts}\n')
                f.write(' '.join(map(str, self.c)))
                f.write('\n')
                for row in self.data:
                    f.write(' '.join(map(str, row)) + '\n')
                f.write(f'\nРешение {solution_type}\n')
                f.write(ans_str)
                f.close()

        # Окно для отображения результатов
        result_window = tk.Toplevel(self.master)
        result_window.title(f"Решение {solution_type}")
        l = 6
        if solution_type == "в частном случае":
            l = 1
        save_button = tk.Button(result_window, text="Сохранить ответ", command=save)
        save_button.pack()
        text_area = tk.Text(result_window, height=10 * l, width=400)
        text_area.pack(padx=10, pady=10)
        if solution_type == "в частном случае":
            canvas = tk.Canvas(result_window, height=600, width=800, bg="white")
            image = Image.open("doctest-output/main.gv.png")
            width, height = image.size
            width, height = 600 * width // height, 600
            image = image.resize((width, height))
            self.photo = ImageTk.PhotoImage(image)
            canvas.create_image((400, 0), anchor='n', image=self.photo)
            canvas.pack(anchor="n")

        text_area.insert(tk.END, ans_str)
        text_area.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = Application(root)
    root.mainloop()
