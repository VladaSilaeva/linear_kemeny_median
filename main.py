
import tkinter as tk
from tkinter import messagebox


def center_window(window, width, height):
    """Центрирование окна на экране"""
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x_coordinate = (screen_width / 2) - (width / 2)
    y_coordinate = (screen_height / 2) - (height / 2)
    window.geometry('%dx%d+%d+%d' % (width, height, x_coordinate, y_coordinate))


def create_initial_window():
    def next_step():
        try:
            experts_count = int(expert_entry.get())
            alternatives_count = int(alt_entry.get())

            if experts_count <= 0 or alternatives_count <= 0:
                raise ValueError

            selected_option = var.get()
            window.destroy()

            if selected_option == "linear":
                create_linear_order_input_window(experts_count, alternatives_count)
            elif selected_option == "matrix":
                create_matrix_input_window(experts_count, alternatives_count)
            else:
                messagebox.showerror("Ошибка", "Произошла ошибка при выборе типа ввода данных.")
        except ValueError:
            messagebox.showerror("Ошибка", "Неверное значение! Введите целые числа больше нуля.")
    # Окно для ввода количества экспертов и альтернатив
    window = tk.Tk()
    window.title("Параметры")

    # Центрирование окна
    center_window(window, 300, 250)

    # Поле для ввода количества экспертов
    expert_label = tk.Label(window, text="Введите количество экспертов:")
    expert_label.pack(padx=10, pady=(10, 0))

    expert_entry = tk.Entry(window)
    expert_entry.pack(padx=10, pady=(0, 10))

    # Поле для ввода количества альтернатив
    alt_label = tk.Label(window, text="Введите количество альтернатив:")
    alt_label.pack(padx=10, pady=(10, 0))

    alt_entry = tk.Entry(window)
    alt_entry.pack(padx=10, pady=(0, 10))

    # Выбор типа ввода данных
    var = tk.StringVar(value="linear")  # По умолчанию выбрано "линейный порядок"
    linear_radio = tk.Radiobutton(window, text="Ввод линейных порядков", variable=var, value="linear")
    matrix_radio = tk.Radiobutton(window, text="Ввод матриц бинарных отношений", variable=var, value="matrix")
    linear_radio.pack(padx=10, pady=(10, 0))
    matrix_radio.pack(padx=10, pady=(0, 10))

    # Следующий шаг
    button_next = tk.Button(window, text="Далее", command=next_step)
    button_next.pack(padx=10, pady=10)

    window.mainloop()


def create_linear_order_input_window(experts_count, alternatives_count):
    # Окно для ввода линейных порядков
    linear_window = tk.Tk()
    linear_window.title(f"Ввод линейных порядков для {experts_count} экспертов")

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

    for i in range(experts_count):
        frame = tk.Frame(scrollable_frame)
        frame.pack(fill='x', padx=10, pady=5)

        label = tk.Label(frame, text=f"Эксперт {i + 1}:")
        label.pack(side='left')

        entry = tk.Entry(frame)
        entry.pack(side='right', fill='x', expand=True)
        entries.append(entry)

    def solve():
        try:
            nonlocal entries
            results = [[int(e) for e in entry.get().split(',')] for entry in entries]
            print(results)
            linear_window.destroy()
            display_results(results, "линейные порядки")
        except ValueError:
            messagebox.showerror("Ошибка", "Неверное значение! Введите заново.")


    button_solve = tk.Button(scrollable_frame, text="Решить", command=solve)
    button_solve.pack(padx=10, pady=10)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    linear_window.mainloop()


def create_matrix_input_window(experts_count, alternatives_count):
    # Окно для ввода матриц бинарных отношений
    matrix_window = tk.Tk()
    matrix_window.title(f"Ввод матриц бинарных отношений для {experts_count} экспертов")

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

    for i in range(experts_count):
        frame = tk.Frame(scrollable_frame)
        frame.pack(fill='x', padx=10, pady=5)

        label = tk.Label(frame, text=f"Матрица эксперта {i + 1}:")
        label.pack(side='top')

        #matrix_frame = tk.Frame(frame)
        #matrix_frame.pack(side='bottom')

        matrix_entries = []
        for j in range(alternatives_count):
            row_frame = tk.Frame(frame)
            row_frame.pack(fill='x', side='top')

            row_entries = []
            for k in range(alternatives_count):
                entry = tk.Entry(row_frame, width=4)
                entry.pack(side='left', padx=2)
                row_entries.append(entry)
            matrix_entries.append(row_entries)
        matrices_entries.append(matrix_entries)

    def solve():
        try:
            nonlocal matrices_entries
            results = [[[int(cell.get()) for cell in row] for row in matrix] for matrix in matrices_entries]
            print(results)
            matrix_window.destroy()
            display_results(results, "матрицы бинарных отношений")
        except ValueError:
            messagebox.showerror("Ошибка", "Неверное значение! Введите заново.")

    button_solve = tk.Button(scrollable_frame, text="Решить", command=solve)
    button_solve.pack(padx=10, pady=10)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    matrix_window.mainloop()


def display_results(data, type_of_data):
    # Окно для отображения результатов
    result_window = tk.Tk()
    result_window.title(f"Результаты ({type_of_data})")

    # Центрирование окна
    center_window(result_window, 400, 300)

    text_area = tk.Text(result_window, height=20, width=50)
    text_area.pack(padx=10, pady=10)

    if type_of_data == "линейные порядки":
        for idx, order in enumerate(data):
            text_area.insert(tk.END, f"Эксперт {idx + 1}:\n")
            text_area.insert(tk.END, ', '.join(map(str, order)) + "\n\n")
    elif type_of_data == "матрицы бинарных отношений":
        for idx, matrix in enumerate(data):
            text_area.insert(tk.END, f"Матрица эксперта {idx + 1}:\n")
            for row in matrix:
                text_area.insert(tk.END, ' '.join(map(str, row)) + "\n")
            text_area.insert(tk.END, "\n")

    result_window.mainloop()


if __name__=="__main__":
    # Запускаем начальное окно
    create_initial_window()
