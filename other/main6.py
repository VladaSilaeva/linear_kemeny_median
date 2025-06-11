import tkinter as tk


class Application(tk.Tk):
    def __init__(self):
        super().__init__()

        self.start()
    def start(self):
        self.title("Выбор формата")
        self.geometry("400x300")  # Размер окна

        # Переменная для хранения выбранного формата
        self.format_var = tk.StringVar()
        self.format_var.set("")

        # Первый экран выбора формата
        label_format = tk.Label(self, text="Формат ввода:")
        label_format.pack(pady=10)

        radio_linear = tk.Radiobutton(self, text="Линейный", variable=self.format_var, value="линейный")
        radio_matrix = tk.Radiobutton(self, text="Матричный", variable=self.format_var, value="матричный")
        radio_linear.pack()
        radio_matrix.pack()

        button_help = tk.Button(self, text="Справка", command=lambda: self.show_help())
        button_next = tk.Button(self, text="Далее", command=lambda: self.next_step_1())
        button_help.place(relx=0.5, rely=0.8, anchor='center')
        button_next.place(relx=0.9, rely=0.9, anchor='se')
    def show_help(self):
        help_window = tk.Toplevel(self)
        help_label = tk.Label(help_window, text="Это справочное окно.")
        help_label.pack()

    def next_step_1(self):
        format_choice = self.format_var.get()
        if not format_choice:
            return

        # Скрываем первый экран и показываем второй экран
        for widget in self.winfo_children():
            widget.destroy()

        label_selected_format = tk.Label(self, text=f"Ваш выбор: {format_choice}")
        label_selected_format.pack(pady=10)

        label_alternatives = tk.Label(self, text="Количество альтернатив:")
        entry_alternatives = tk.Entry(self)
        label_experts = tk.Label(self, text="Количество экспертов:")
        entry_experts = tk.Entry(self)

        label_alternatives.pack()
        entry_alternatives.pack()
        label_experts.pack()
        entry_experts.pack()

        button_back = tk.Button(self, text="Назад", command=lambda: self.back_to_start())
        button_help = tk.Button(self, text="Справка", command=lambda: self.show_help())
        button_next = tk.Button(self, text="Далее", command=lambda: self.next_step_2(entry_alternatives, entry_experts))

        button_back.place(relx=0.1, rely=0.9, anchor='sw')
        button_help.place(relx=0.5, rely=0.8, anchor='center')
        button_next.place(relx=0.9, rely=0.9, anchor='se')

    def back_to_start(self):
        for widget in self.winfo_children():
            widget.destroy()
        self.start()

    def next_step_2(self, alternatives_entry, experts_entry):
        def _next_step():
            try:
                num_alternatives = int(alternatives_entry.get())
                num_experts = int(experts_entry.get())

                # Скрываем предыдущий экран и показываем третий экран
                for widget in self.winfo_children():
                    widget.destroy()

                label_info = tk.Label(self, text=f"Введено:\nАльтернатив: {num_alternatives}\nЭксперты: {num_experts}")
                label_info.pack(pady=10)

                button_back = tk.Button(self, text="Назад",
                                        command=lambda: self.back_to_previous(num_alternatives, num_experts))
                button_help = tk.Button(self, text="Справка", command=lambda: self.show_help())
                button_next = tk.Button(self, text="Далее",
                                        command=lambda: self.collect_data(num_alternatives, num_experts))

                button_back.place(relx=0.1, rely=0.9, anchor='sw')
                button_help.place(relx=0.5, rely=0.8, anchor='center')
                button_next.place(relx=0.9, rely=0.9, anchor='se')
            except ValueError:
                pass

        return _next_step()

    def back_to_previous(self, num_alternatives, num_experts):
        for widget in self.winfo_children():
            widget.destroy()
        self.next_step_1()

    def collect_data(self, num_alternatives, num_experts):
        for widget in self.winfo_children():
            widget.destroy()

        data_entries = []
        for i in range(num_experts):
            expert_frame = tk.Frame(self)
            expert_frame.pack(fill=tk.X, pady=5)

            label = tk.Label(expert_frame, text=f"Показатели эксперта №{i + 1}:")
            label.pack(side=tk.LEFT)

            entries_row = []
            for j in range(num_alternatives):
                entry = tk.Entry(expert_frame, width=5)
                entry.pack(side=tk.LEFT, padx=5)
                entries_row.append(entry)
            data_entries.append(entries_row)

        submit_button = tk.Button(self, text="Отправить",
                                  command=lambda: print([entry.get() for row in data_entries for entry in row]))
        submit_button.pack(pady=10)


if __name__ == "__main__":
    app = Application()
    app.mainloop()

