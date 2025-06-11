import tkinter as tk
from tkinter import filedialog


class MatrixApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Матрица 5x5")
        self.geometry("400x300")
        self.matrix = [[tk.StringVar(value="0") for _ in range(5)] for _ in range(5)]
        self.entries = []
        self.create_widgets()

    def create_widgets(self):
        # Создаем сетку входов для матрицы
        for i in range(5):
            row_entries = []
            for j in range(5):
                entry = tk.Entry(self, textvariable=self.matrix[i][j], width=4)
                entry.grid(row=i, column=j)
                entry.bind("<FocusOut>", lambda event, row=i, col=j: self.validate_entry(event, row, col))
                row_entries.append(entry)
            self.entries.append(row_entries)

        # Кнопка для загрузки матрицы из файла
        load_button = tk.Button(self, text="Загрузить матрицу из файла", command=self.load_matrix_from_file)
        load_button.grid(row=6, columnspan=5, pady=(10, 0))

    def validate_entry(self, event, row, col):
        value = self.matrix[row][col].get().strip()
        try:
            if not value.isdigit() or int(value) <= 0:
                raise ValueError
            self.entries[row][col].configure(bg="white")
        except ValueError:
            self.entries[row][col].configure(bg="red")

    def load_matrix_from_file(self):
        file_path = filedialog.askopenfilename(filetypes=[('Text files', '*.txt')])
        if file_path:
            with open(file_path, 'r') as f:
                lines = f.read().splitlines()
                for i, line in enumerate(lines):
                    values = line.split()
                    for j, val in enumerate(values):
                        self.matrix[i][j].set(val)


if __name__ == "__main__":
    app = MatrixApp()
    app.mainloop()