from perceptron import *
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog as fd
from matplotlib.figure import Figure

class Application(tk.Tk):
	def __init__(self):
		super().__init__()
		self.title("Perceptron")
		self.geometry("850x600")
		self.createWidgets()

	def createWidgets(self):
		# select file
		tk.Label(text="File name: ", font=('Comic Sans MS', 12)).grid(row=0, column=0)
		self.filename = tk.StringVar()
		print_filename = tk.Label(self, textvariable=self.filename, font=('Comic Sans MS', 12))
		print_filename.grid(row=0, column=1)
		tk.Button(self, text='Select File', font=('Comic Sans MS', 12), command=self.select_file).grid(row=0, column=2)

		# setting epoch n lr
		tk.Label(text="Epoch: ", font=('Comic Sans MS', 12)).grid(row=1, column=0)
		self.epoch_box = tk.Spinbox(self, from_=0, to=200, font=('Comic Sans MS', 12))
		self.epoch_box.grid(row=1, column=1)
		tk.Label(text="Learning rate: ", font=('Comic Sans MS', 12)).grid(row=2, column=0)
		self.lr_box = tk.Spinbox(self, from_=0, to=1, increment=0.01, font=('Comic Sans MS', 12))
		self.lr_box.grid(row=2, column=1)

		tk.Button(master=self, text='Start', font=('Comic Sans MS', 12), command=self.draw_picutre).grid(row=0, column=5)
		tk.Button(master=self, text='Exit', font=('Comic Sans MS', 12), command=self._quit).grid(row=1, column=5)
		
		# accuracy
		tk.Label(text="Training Accuracy: ", font=('Comic Sans MS', 12)).grid(row=0, column=3)
		self.train_acc_text = tk.StringVar()
		print_train_acc = tk.Label(self, textvariable=self.train_acc_text, font=('Comic Sans MS', 12))
		print_train_acc.grid(row=0, column=4)
		tk.Label(text="Testing Accuracy: ", font=('Comic Sans MS', 12)).grid(row=1, column=3)
		self.test_acc_text = tk.StringVar()
		print_test_acc = tk.Label(self, textvariable=self.test_acc_text, font=('Comic Sans MS', 12))
		print_test_acc.grid(row=1, column=4)
		tk.Label(text="Weights: ", font=('Comic Sans MS', 12)).grid(row=8, column=0)
		self.weight_text = tk.StringVar()
		self.print_weight = tk.Label(self, textvariable=self.weight_text, font=('Comic Sans MS', 12))
		self.print_weight.grid(row=8, column=1)
		

		# figures
		self.figure_train = Figure(figsize=(4,4), dpi=100)
		self.train_ax = self.figure_train.add_subplot(111, projection='3d')
		self.train_ax.set_title('Training result')
		self.train_plt = FigureCanvasTkAgg(self.figure_train, self)
		self.train_plt.get_tk_widget().grid(row=6, column=0, columnspan=3)

		self.figure_test = Figure(figsize=(4,4), dpi=100)
		self.test_ax = self.figure_test.add_subplot(111, projection='3d')
		self.test_ax.set_title('Testing result')
		self.test_plt = FigureCanvasTkAgg(self.figure_test, self)
		self.test_plt.get_tk_widget().grid(row=6, column=3, columnspan=3)
	
	def draw_picutre(self):
		epochs = int(self.epoch_box.get())
		learning_rate = float(self.lr_box.get())
		Train_acc, x_train, y_train, Test_acc, x_test, y_test, weights, train_predict, test_predict = process(self.File, epochs, learning_rate)
		
		# print accuracy and weights
		self.train_acc_text.set(str(round(Train_acc * 100, 2)) + "%")
		self.test_acc_text.set(str(round(Test_acc * 100, 2)) + "%")
		for i in range(len(weights)):
			weights[i] = round(weights[i], 2)
		self.weight_text.set(str(weights))
		
		self.train_ax.clear()
		# draw points (training)
		self.train_ax.set_title('Training result')
		#self.train_ax.set_xlim(np.min(x_train[:, 0]), np.max(x_train[:, 0]))
		self.train_ax.set_ylim(np.min(x_train[:, 1]), np.max(x_train[:, 1]))
		self.train_ax.scatter(x_train[:, 0], x_train[:, 1], train_predict, c=y_train)
		# draw line
		x = np.linspace(np.min(x_train[:, 0]), np.max(x_train[:, 0]), 100)
		y = (-weights[1] * x - weights[0])/weights[2]
		self.train_ax.plot(x, y)
		self.train_plt.draw()

		self.test_ax.clear()
		# draw points (testing)
		self.test_ax.set_title('Testing result')
		#self.test_ax.set_xlim(np.min(x_test[:, 0]), np.max(x_test[:, 0]))
		self.test_ax.set_ylim(np.min(x_test[:, 1]), np.max(x_test[:, 1]))
		self.test_ax.scatter(x_test[:, 0], x_test[:, 1], test_predict, c=y_test)
		# draw line
		x = np.linspace(np.min(x_test[:, 0]), np.max(x_test[:, 0]), 100)
		y = (-weights[1] * x - weights[0])/weights[2]
		self.test_ax.plot(x, y)
		self.test_plt.draw()

		
	def select_file(self):
		filetypes = (('text files', '*.txt'), ('All files', '*.*'))
		self.File = fd.askopenfilename(title='Open a file', initialdir='/', filetypes=filetypes)
		file = ""
		for i in range(len(self.File) - 1, 0, -1):
			if self.File[i] == '/':
				file = self.File[i+1:]
				break
		self.filename.set(file)

	def _quit(self):
		self.quit()
		self.destroy()


if __name__ == "__main__":
	app = Application()
	app.mainloop()