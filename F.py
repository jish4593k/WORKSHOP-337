import sys
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Generate random data
np.random.seed(42)
X = np.random.rand(100, 1)
y = 3 * X.squeeze() + 2 + 0.1 * np.random.randn(100)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)

# Train the neural network
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    y_pred = model(X_tensor)
    loss = criterion(y_pred, y_tensor)
    loss.backward()
    optimizer.step()


sns.set(style="whitegrid")
sns.scatterplot(x=X.squeeze(), y=y, label="Original Data")
sns.lineplot(x=X.squeeze(), y=model(X_tensor).detach().numpy().squeeze(), color="red", label="Model Prediction")
plt.title("Simple Linear Regression")
plt.legend()
class PlotWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.canvas = MplCanvas(self.central_widget, width=5, height=4, dpi=100)
        self.layout.addWidget(self.canvas)

        self.plot_data()

    def plot_data(self):
        self.canvas.axes.clear()
        sns.scatterplot(x=X.squeeze(), y=y, label="Original Data", ax=self.canvas.axes)
        sns.lineplot(x=X.squeeze(), y=model(X_tensor).detach().numpy().squeeze(), color="red", label="Model Prediction", ax=self.canvas.axes)
        self.canvas.axes.set_title('Simple Linear Regression')
        self.canvas.axes.legend()
        self.canvas.draw()


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlotWindow()
    window.show()
    sys.exit(app.exec_())
