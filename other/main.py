import tkinter as tk
import time
from tkinter import messagebox, filedialog
from matrix_solve import main,special_case
from PIL import Image, ImageTk


def center_window(window, width, height):
    """Центрирование окна на экране"""
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x_coordinate = (screen_width / 2) - (width / 2)
    y_coordinate = (screen_height / 2) - (height / 2)
    window.geometry('%dx%d+%d+%d' % (width, height, x_coordinate, y_coordinate))

def validate_input_n(entry, l=0, r=None):
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
def validate_input_elem(entry):
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
def validate_input_lin(entry, n):
    try:
        wrong_input = False
        values = [int(e) for e in entry.get().split()]
        values.sort()
        if values != list(range(1,n+1)):
            wrong_input = True
        if wrong_input:
            raise ValueError
        entry.config(bg="white")
        return True
    except ValueError:
        entry.config(bg="pink")
        return False
def validate_input_c(entry, m):
    try:
        wrong_input = False
        values = [float(e) for e in entry.get().split()]
        if len(values)!=m:
            wrong_input=True
        if sum(values)!=m:#float==int
            wrong_input=True
        if wrong_input:
            raise ValueError
        entry.config(bg="white")
        return True
    except ValueError:
        entry.config(bg="pink")
        return False
def create_initial_window():
    def next_step():
        try:
            experts_count = int(expert_entry.get())
            alternatives_count = int(alt_entry.get())

            if not (validate_input_n(expert_entry) and validate_input_n(alt_entry)):
                raise ValueError

            selected_option = var.get()
            window.destroy()

            if selected_option == "linear":
                create_linear_order_input_window(experts_count, alternatives_count)
            elif selected_option == "matrix":
                create_matrix_input_window(experts_count, alternatives_count)
        except ValueError:
            messagebox.showerror("Ошибка", "Неверное значение! Введите целые числа больше нуля.")
    # Окно для ввода количества экспертов и альтернатив
    window = tk.Tk()
    window.title("Параметры")

    # Центрирование окна
    center_window(window, 400, 200)

    # Поле для ввода количества экспертов
    expert_label = tk.Label(window, text="Введите количество экспертов:")
    expert_label.grid(row=0, column=0, padx=10, pady=5, sticky='sw')

    expert_entry = tk.Entry(window)
    expert_entry.grid(row=0, column=1)
    expert_entry.bind("<FocusOut>", lambda event, e=expert_entry: validate_input_n(e))

    # Поле для ввода количества альтернатив
    alt_label = tk.Label(window, text="Введите количество альтернатив:")
    alt_label.grid(row=1, column=0, padx=10, pady=5, sticky='sw')

    alt_entry = tk.Entry(window)
    alt_entry.grid(row=1, column=1)
    alt_entry.bind("<FocusOut>", lambda event, e=alt_entry: validate_input_n(e))

    # Выбор типа ввода данных
    var = tk.StringVar(value="linear")  # По умолчанию выбрано "линейный порядок"
    linear_radio = tk.Radiobutton(window, text="Ввод линейных порядков", variable=var, value="linear")
    matrix_radio = tk.Radiobutton(window, text="Ввод матриц бинарных отношений", variable=var, value="matrix")
    linear_radio.grid(row=2, column=0, padx=10, sticky='sw')
    matrix_radio.grid(row=3, column=0, padx=10, sticky='sw')

    # Следующий шаг
    button_next = tk.Button(window, text="Далее", command=next_step)
    button_next.grid(row=4, columnspan=2, pady=10)

    window.mainloop()


def create_linear_order_input_window(experts_count, alternatives_count):
    # Окно для ввода линейных порядков
    linear_window = tk.Toplevel()
    linear_window.title(f"Ввод линейных порядков для {experts_count} экспертов и {alternatives_count} альтернатив")

    # Центрирование окна
    center_window(linear_window, 400, 500)

    canvas = tk.Canvas(linear_window)
    scrollbar = tk.Scrollbar(linear_window, orient="vertical", command=canvas.yview)
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

    label = tk.Label(frame, text=f"Количество альтернатив: {alternatives_count}")
    label.pack(side='top')

    label = tk.Label(frame, text=f"Веса экспертов:")
    label.pack(side='left')

    experts_coef = tk.StringVar(value="1 " * experts_count)
    entry_c = tk.Entry(frame, textvariable=experts_coef)
    entry_c.pack(side='right', fill='x', expand=True)
    entry_c.bind("<FocusOut>", lambda event, e=entry_c: validate_input_c(e,experts_count))

    lin=[]
    for i in range(experts_count):
        frame = tk.Frame(scrollable_frame)
        frame.pack(fill='x', padx=10, pady=5)

        label = tk.Label(frame, text=f"Эксперт {i + 1}:")
        label.pack(side='left')

        lin.append(tk.StringVar(value=""))
        entry = tk.Entry(frame,textvariable=lin[i])
        entry.pack(side='right', fill='x', expand=True)
        entry.bind("<FocusOut>", lambda event, e=entry: validate_input_lin(e,alternatives_count))

        entries.append(entry)

    def solve():
        try:
            nonlocal entries,entry_c
            #entry_c.config(state="readonly")
            if not (validate_input_c(entry_c,experts_count)):
                raise ValueError
            for entry in entries:
                if not validate_input_lin(entry,alternatives_count):
                    raise ValueError
                #entry.config(state="readonly")
            r = [[int(e) for e in entry.get().split()] for entry in entries]
            c = [float(ci) for ci in entry_c.get().split()]
            print(r)
            print(c)
            #linear_window.destroy()
            solve_window(r,c,experts_count,alternatives_count, "линейные порядки")
            """entry_c.config(state="normal")
            for entry in entries:
                entry.config(state="normal")"""
        except ValueError:
            messagebox.showerror("Ошибка", "Неверное значение! Введите заново.")

    def load_lin_from_file():
        file_path = filedialog.askopenfilename(filetypes=[('Text files', '*.txt')])
        if file_path:
            r=[]
            c=[]
            n,m = 0, 0
            with open(file_path, 'r') as f:
                nm_line=f.readline()
                c_line=f.readline()
                for line in f.readlines():
                    if len(line.strip()):
                        r.append(line.split())
                    else:
                        break
                f.close()
                if len(nm_line.strip())==2:
                    n, m = [int(i) for i in nm_line.split()]
                else:
                    n = len(r[0])
                    m = len(r)
                if n != alternatives_count:
                    messagebox.showerror("Ошибка", "Количество альтернатив в файле не соответствует введенному ранее!")
                    return
                if len(c_line.strip()):
                    c = [float(i) for i in c_line.split()]
                    if len(c) != m or sum(c) != m:
                        messagebox.showerror("Внимание",
                                             "Введены некорректные веса для экспертов. Они заменены на веса по умолчанию (c_i=1.0, i=1,m)")

                        print('wrong c')
                        c = [1.0] * experts_count
                else:
                    c = [1.0] * experts_count
                    #messagebox.showerror("Внимание","Не были введены веса для экспертов. Веса заменены на веса по умолчанию (c_i=1.0, i=1,m)")

                experts_coef.set(' '.join(map(str, c)))
                for k in range(experts_count):
                    if k<m:
                        lin[k].set(" ".join(map(str,r[k])))
                    else:
                        lin[k].set("")
                return r, c, n, m
    def go_back():
        linear_window.destroy()
        create_initial_window()


    button_solve = tk.Button(scrollable_frame, text="Решить", command=solve)
    button_solve.pack(padx=10, pady=10)

    load_button = tk.Button(scrollable_frame, text="Загрузить данные из файла", command=load_lin_from_file)
    load_button.pack(padx=10, pady=10)

    button_back = tk.Button(scrollable_frame, text="Назад", command=go_back)
    button_back.pack(padx=10, pady=10)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    linear_window.mainloop()


def create_matrix_input_window(experts_count, alternatives_count):
    # Окно для ввода матриц бинарных отношений
    matrix_window = tk.Toplevel()
    matrix_window.title(f"Ввод матриц бинарных отношений для {experts_count} экспертов и {alternatives_count} альтернатив")

    # Центрирование окна
    center_window(matrix_window, 400, 600)

    canvas = tk.Canvas(matrix_window)
    scrollbar = tk.Scrollbar(matrix_window, orient="vertical", command=canvas.yview)
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

    label = tk.Label(frame, text=f"Количество альтернатив: {alternatives_count}")
    label.pack(side='top')

    label = tk.Label(frame, text=f"Веса экспертов:")
    label.pack(side='left')

    experts_coef=tk.StringVar(value="1 "*experts_count)
    entry_c = tk.Entry(frame,textvariable=experts_coef)
    entry_c.pack(side='right', fill='x', expand=True)
    entry_c.bind("<FocusOut>", lambda event, e=entry_c: validate_input_c(e,experts_count))

    for k in range(experts_count):
        frame = tk.Frame(scrollable_frame)
        frame.pack(fill='x', padx=10, pady=5)

        label = tk.Label(frame, text=f"Матрица эксперта {k + 1}:")
        label.pack(side='top')

        #matrix_frame = tk.Frame(frame)
        #matrix_frame.pack(side='bottom')

        matrix_entries = []
        matrix = [[tk.StringVar(value="0") for _ in range(alternatives_count)] for _ in range(alternatives_count)]
        for i in range(alternatives_count):
            row_frame = tk.Frame(frame)
            row_frame.pack(fill='x', side='top')

            row_entries = []
            for j in range(alternatives_count):
                entry = tk.Entry(row_frame, textvariable=matrix[i][j], width=4)
                entry.pack(side='left', padx=2)
                entry.bind('<Right>', lambda event, x=i, y=j, num=k: move_right(x, y, num))
                entry.bind('<Left>', lambda event, x=i, y=j, num=k: move_left(x, y, num))
                entry.bind('<Up>', lambda event, x=i, y=j, num=k: move_up(x, y, num))
                entry.bind('<Down>', lambda event, x=i, y=j, num=k: move_down(x, y, num))
                entry.bind("<FocusOut>", lambda event, e=entry: validate_input_elem(e))
                row_entries.append(entry)
            matrix_entries.append(row_entries)
        matrices_entries.append(matrix_entries)
        matrices.append(matrix)

    def move_right(row, col, num):
        new_col = col+1
        if new_col >= alternatives_count:
            return move_down(row,0,num)
        matrices_entries[num][row][new_col].focus_set()

    def move_left(row, col, num):
        new_col = col - 1
        if new_col < 0:
            return move_up(row, alternatives_count-1, num)
        matrices_entries[num][row][new_col].focus_set()

    def move_up(row, col, num):
        new_row=row-1
        new_num = num
        if new_row < 0:
            new_row=alternatives_count-1
            new_num=num-1
        if new_num < 0:
            return
        matrices_entries[new_num][new_row][col].focus_set()

    def move_down(row, col, num):
        new_row = row+1
        new_num = num
        if new_row >= alternatives_count:
            new_row=0
            new_num=num+1
        if new_num >= experts_count:
            return
        matrices_entries[new_num][new_row][col].focus_set()
    def solve():
        try:
            nonlocal entry_c, matrices_entries
            if not (validate_input_c(entry_c, experts_count)):
                raise ValueError
            entry_c.config(state="readonly")
            for matrix_enteries in matrices_entries:
                for i in range(alternatives_count):
                    for j in range(alternatives_count):
                        if not validate_input_elem(matrix_enteries[i][j]):
                            raise ValueError
                        matrix_enteries[i][j].config(state="readonly")
            R = [[[int(cell.get()) for cell in row] for row in matrix] for matrix_enteries in matrices_entries]
            c = [float(ci) for ci in entry_c.get().split()]
            print(R)
            print(c)
            #matrix_window.destroy()
            solve_window(R,c,experts_count,alternatives_count, "матрицы бинарных отношений")
            entry_c.config(state="normal")
            for matrix_enteries in matrices_entries:
                for i in range(alternatives_count):
                    for j in range(alternatives_count):
                        matrix_enteries[i][j].config(state="normal")
        except ValueError:
            messagebox.showerror("Ошибка", "Неверное значение! Введите заново.")

    def go_back():
        matrix_window.destroy()
        create_initial_window()
    def load_matrix_from_file():
        file_path = filedialog.askopenfilename(filetypes=[('Text files', '*.txt')])
        if file_path:
            R=[]
            c=[]
            n,m = 0, 0
            with open(file_path, 'r') as f:
                nm_line=f.readline()
                c_line=f.readline()
                for line in f.readlines():
                    if len(line.strip()):
                        R.append(line.split())
                    else:
                        break
                f.close()
                if len(nm_line.strip())==2:
                    n, m = [int(i) for i in nm_line.split()]
                else:
                    n = len(R[0])
                    m = len(R)//n
                if n != alternatives_count:
                    messagebox.showerror("Ошибка", "Размер матрицы в файле не соответствует введенному ранее!")
                    return
                if len(c_line.strip()):
                    c = [float(i) for i in c_line.split()]
                    if len(c) != m or sum(c) != m:
                        print('wrong c')
                        c = [1.0] * m
                else:
                    c = [1.0] * m
                experts_coef.set(' '.join(map(str, c)))
                for k in range(experts_count):
                    for i in range(n):
                        for j in range(n):
                            if k<m:
                                matrices[k][i][j].set(R[n*k+i][j])
                            else:
                                matrices[k][i][j].set("0")
                return R, c, n, m

    button_solve = tk.Button(scrollable_frame, text="Решить", command=solve)
    button_solve.pack(padx=10, pady=10)

    load_button = tk.Button(scrollable_frame, text="Загрузить матрицу из файла", command=load_matrix_from_file)
    load_button.pack(padx=10, pady=10)

    button_back = tk.Button(scrollable_frame, text="Назад", command=go_back)
    button_back.pack(padx=10, pady=10)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    matrix_window.mainloop()

def solve_window(data,c,m,n, type_of_data):
    def solve():
        start = time.time()
        lin = type_of_data=="линейные порядки"
        ans_str, ans, solver = main(data, c, n, m, lin=lin)
        finish = time.time()
        #t=finish-start

        #text_area.insert(tk.END, "solve"+ans_str)
        display_results("solve\n"+ans_str)

    def fast_solve():
        lin = type_of_data == "линейные порядки"
        ans_str, ans, solver = main(data, c, n, m, lin=lin, forced_down=True)
        display_results("fast solve\n"+ans_str)
    def special_solve():
        ans_str, ans = special_case(data,c,n,m)
        display_results("special solve\n"+ans_str,0)
    def first():
        lin = type_of_data == "линейные порядки"
        ans_strs=""
        ans=[]
        for i in range(n):
            ans_strs += f'first={i + 1}\n'
            ans_str, a, solver = main(data, c, n, m, first=i, lin=True)
            ans_strs+=ans_str+'\n'
            ans.append(a)
        ans_str+='\n'.join([f'r_first({i + 1})={ans[i]}' for i in range(n)])
        display_results(ans_strs)
    def last():
        lin = type_of_data == "линейные порядки"
        ans_strs = ""
        ans = []
        for i in range(n):
            ans_strs += f'last={i + 1}\n'
            ans_str, a, solver = main(data, c, n, m, last=i, lin=True)
            ans_strs += ans_str+'\n'
            ans.append(a)
        ans_str += '\n'.join([f'r_last({i + 1})={ans[i]}' for i in range(n)])
        display_results(ans_strs)
    def first_last():
        lin = type_of_data == "линейные порядки"

        try:
            nonlocal entry_first, entry_last
            if not (validate_input_n(entry_first, 0,n) and validate_input_n(entry_last, 0,n)):
                raise ValueError
            f = int(entry_first.get())
            l = int(entry_last.get())
            ans_str1, ans1,solver = main(data, c, n, m, first=f-1, last=l-1, lin=True)
            ans_str2, ans2,solver = main(data, c, n, m, first=l-1, last=f-1, lin=True)
            ans_str=f'first={f}, last={l}\n{ans_str1} first={l}, last={f}\n{ans_str2}\n'
            if ans1[0]<ans2[0]:
                ans_str+=f'{f} лучше {l} в {ans2[0]/ans1[0]} раз'
            else:
                ans_str += f'{l} лучше {f} в {ans1[0]/ans2[0]} раз'
            display_results(ans_str)
        except ValueError:
            messagebox.showerror("Ошибка", "Неверное значение! Введите заново.")
    def go_back():
        window.destroy()
    # Окно для ввода количества экспертов и альтернатив
    window = tk.Toplevel()
    window.title("Параметры")

    # Центрирование окна
    center_window(window, 600, 250)

    # Обычное решение
    button_solve = tk.Button(window, text="Решить", command=solve)
    button_solve.grid(row=0,column=0, padx=10, pady=10)
    # Решение с ускорением
    button_fast_solve = tk.Button(window, text="Решить с ускорением", command=fast_solve)
    button_fast_solve.grid(row=0,column=1,padx=10, pady=10)
    # Решение в частном случае
    button_special_solve = tk.Button(window, text="Решить в частном случае", command=special_solve)
    button_special_solve.grid(row=0,column=2,padx=10, pady=10)

    #
    button_first = tk.Button(window, text="Закрепить первую альтернативу", command=first)
    button_first.grid(row=1,column=0,padx=10, pady=10)
    button_last = tk.Button(window, text="Закрепить последнюю альтернативу", command=last)
    button_last.grid(row=2,column=0,padx=10, pady=10)

    label_first = tk.Label(window, text="first:")
    label_first.grid(row=3, column=0, padx=10, pady=5, sticky='sw')
    entry_first = tk.Entry(window)
    entry_first.grid(row=3, column=1)
    entry_first.bind("<FocusOut>", lambda event, e=entry_first: validate_input_n(e, 0, n))

    last_label = tk.Label(window, text="last:")
    last_label.grid(row=4, column=0, padx=10, pady=5, sticky='sw')
    entry_last = tk.Entry(window)
    entry_last.grid(row=4, column=1)
    entry_last.bind("<FocusOut>", lambda event, e=entry_last: validate_input_n(e, 0, n))

    button_first_last = tk.Button(window, text="Сравнить две альтернативы", command=first_last)
    button_first_last.grid(row=3,rowspan=2, column=2, padx=10, pady=10)

    # Назад к введению данных
    button_back = tk.Button(window, text="Назад", command=go_back)
    button_back.grid(row=5, column=0,padx=10, pady=10)

    """canvas = tk.Canvas(window)
    scrollbar = tk.Scrollbar(window, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    frame = tk.Frame(scrollable_frame)
    frame.pack(fill='x', padx=10, pady=5)
    text_area = tk.Text(frame, height=20, width=50)
    text_area.grid(padx=10, pady=10, columnspan=3)
    canvas.grid(columnspan=3)
    scrollbar.grid(column=4, sticky="")"""


    is_special_case = False
    if type_of_data == "линейные порядки":
        if m % 2:
            is_special_case = True
            for i in c:
                if i!=1:
                    is_special_case = False
                    break
    if not is_special_case:
        button_special_solve.config(state = ["disabled"])

    window.mainloop()
def display_results(ans_str, type=None):
    # Окно для отображения результатов
    result_window = tk.Toplevel()
    result_window.title(f"Результаты")
    l=6
    if type is not None:
        l=1

    text_area = tk.Text(result_window, height=10*l, width=400)
    text_area.pack(padx=10, pady=10)
    if type is not None:
        canvas = tk.Canvas(result_window,height=600,width=800,bg="white")
        """img = tk.PhotoImage(master=canvas,file='doctest-output/main.gv.png')
        w = img.width()
        h = img.height()
        print(h)
        image = canvas.create_image(w//2, h//2, image=img)"""
        image = Image.open("doctest-output/main.gv.png")
        width, height=image.size
        width, height = 600*width//height, 600
        image=image.resize((width,height))
        photo = ImageTk.PhotoImage(image)
        image = canvas.create_image((400, 0), anchor='n', image=photo)
        canvas.pack(anchor="n")
    # Центрирование окна
    center_window(result_window, 800, 600)
    text_area.insert(tk.END,ans_str)
    text_area.config(state="disabled")
    result_window.mainloop()


if __name__=="__main__":
    # Запускаем начальное окно
    create_initial_window()
