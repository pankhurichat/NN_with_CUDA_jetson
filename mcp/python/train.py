import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from network import Network
from losses import CrossEntropyLoss


# ----------------------------
# Load Iris Dataset
# ----------------------------
iris = load_iris()

X = iris.data                    # (150,4)
y = iris.target.reshape(-1, 1)   # (150,1)


# ----------------------------
# One-Hot Encode Labels
# ----------------------------
encoder = OneHotEncoder(sparse_output=False)
Y = encoder.fit_transform(y)      # (150,3)


# ----------------------------
# Train/Test Split
# ----------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)


# ----------------------------
# Create Network
# ----------------------------
network = Network(
    input_size=4,
    hidden_size=8,
    output_size=3
)

loss_fn = CrossEntropyLoss()

learning_rate = 0.05
epochs = 500


# ----------------------------
# Training Loop
# ----------------------------
for epoch in range(epochs):

    total_loss = 0

    for x, target in zip(X_train, Y_train):

        # Convert to column vectors
        x = x.reshape(-1,1)
        target = target.reshape(-1,1)

        # Forward
        prediction = network.forward(x)

        # Loss
        loss = loss_fn.forward(prediction, target)
        total_loss += loss

        # Backward
        network.backward(loss_fn.backward())

        # Update weights
        network.update(learning_rate)

    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d}   Loss = {total_loss:.4f}")


x = np.array([5.1,3.5,1.4,0.2]).reshape(-1,1)

prediction = network.forward(x)

print(prediction)
print(np.argmax(prediction))

# ----------------------------
# Evaluate
# ----------------------------
correct = 0

for x, target in zip(X_test, Y_test):

    x = x.reshape(-1,1)

    prediction = network.forward(x)

    pred_class = np.argmax(prediction)
    true_class = np.argmax(target)

    if pred_class == true_class:
        correct += 1


accuracy = correct / len(X_test)

print(f"\nTest Accuracy : {accuracy*100:.2f}%")

np.savetxt("W1.txt", network.dense1.w)
np.savetxt("b1.txt", network.dense1.b)

np.savetxt("W2.txt", network.dense2.w)
np.savetxt("b2.txt", network.dense2.b)