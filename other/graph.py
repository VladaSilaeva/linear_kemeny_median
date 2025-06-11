import tkinter as tk


def create_experts_window():
    # Окно для ввода количества экспертов
    window = tk.Tk()
    window.title("Количество экспертов")

    label = tk.Label(window, text="Введите количество экспертов:")
    label.pack(padx=10, pady=10)

    entry = tk.Entry(window)
    entry.pack(padx=10, pady=10)

    def next_step():
        global experts_count
        try:
            experts_count = int(entry.get())
            if experts_count <= 0:
                raise ValueError
            window.destroy()
            create_data_entry_window(experts_count)
        except ValueError:
            tk.messagebox.showerror("Ошибка", "Неверное значение! Введите целое число больше нуля.")

    button = tk.Button(window, text="Далее", command=next_step)
    button.pack(padx=10, pady=10)

    window.mainloop()


def create_data_entry_window(count):
    # Окно для ввода данных от каждого эксперта
    data_window = tk.Tk()
    data_window.title(f"Ввод данных {count} экспертов")

    entries = []

    for i in range(count):
        frame = tk.Frame(data_window)
        frame.pack(fill='x', padx=10, pady=5)

        label = tk.Label(frame, text=f"Эксперт {i + 1}:")
        label.pack(side='left')

        entry = tk.Entry(frame)
        entry.pack(side='right', fill='x', expand=True)
        entries.append(entry)

    def solve():
        nonlocal entries
        results = [entry.get() for entry in entries]
        data_window.destroy()
        display_results(results)

    button = tk.Button(data_window, text="Решить", command=solve)
    button.pack(padx=10, pady=10)

    data_window.mainloop()


def display_results(data):
    # Окно для отображения введенных данных
    result_window = tk.Tk()
    result_window.title("Результаты")

    text_area = tk.Text(result_window, height=20, width=50)
    text_area.pack(padx=10, pady=10)

    for idx, value in enumerate(data):
        text_area.insert(tk.END, f"Эксперт {idx + 1}: {value}\n")

    result_window.mainloop()


# Запускаем первую форму
create_experts_window()