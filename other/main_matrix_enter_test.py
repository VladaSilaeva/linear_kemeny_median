import tkinter as tk


class MatrixApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Матрица 5x5")
        self.geometry("400x300")
        self.matrix = [[tk.StringVar(value="0") for _ in range(5)] for _ in range(5)]
        self.entries = []
        self.create_widgets()

    def create_widgets(self):
        # Создаем сетку из Entry для ввода элементов матрицы
        for i in range(5):
            row_entries = []
            for j in range(5):
                entry = tk.Entry(self, textvariable=self.matrix[i][j], width=4)
                entry.grid(row=i, column=j)
                entry.bind("<FocusOut>", lambda event, i=i, j=j: self.validate_entry(i, j))
                row_entries.append(entry)
            self.entries.append(row_entries)

        # Кнопка для проверки всех значений
        check_button = tk.Button(self, text="Проверить всю матрицу", command=self.check_all_values)
        check_button.grid(row=6, columnspan=5)

    def validate_entry(self, i, j):
        value = self.matrix[i][j].get().strip()
        if not value.isdigit() or int(value) < 0:
            self.entries[i][j].configure(bg="red")
        else:
            self.entries[i][j].configure(bg="white")

    def check_all_values(self):
        for i in range(5):
            for j in range(5):
                self.validate_entry(i, j)


if __name__ == "__main__":
    app = MatrixApp()
    app.mainloop()