# Deep Learning Basics README

## 1. Activation Functions

Activation functions help neural networks learn complex patterns.

Without activation functions:
- Neural network becomes only a linear model
- Cannot solve complex problems

### Common Activation Functions

| Function | Purpose |
|---|---|
| ReLU | Most common for hidden layers |
| Sigmoid | Binary classification output |
| Softmax | Multi-class classification |
| Leaky ReLU | Solves dying ReLU problem |
| ELU | Smooth negative values |
| SELU | Self-normalizing networks |

---

# ReLU

\[
f(x)=max(0,x)
\]

- Positive values stay same
- Negative values become 0

### Advantage
- Fast
- Simple
- Most used

### Problem
Dying ReLU:
Some neurons stop learning.

---

# Leaky ReLU

\[
f(x)=max(0.01x,x)
\]

Negative values are not fully zero.

### Benefit
- Helps neurons continue learning

---

# PReLU

\[
f(x)=max(ax,x)
\]

- Similar to Leaky ReLU
- Parameter `a` is learned automatically

---

# ELU

\[
f(x)=
\begin{cases}
x,&x>0 \\
\alpha(e^x-1),&x\le0
\end{cases}
\]

- Smooth negative values
- Better gradient flow

---

# SELU

\[
f(x)=\lambda
\begin{cases}
x,&x>0 \\
\alpha(e^x-1),&x\le0
\end{cases}
\]

- Self-normalizing activation
- Good for deep networks

---

# 2. Regularization

Regularization prevents overfitting.

Overfitting means:
- Model memorizes training data
- Performs badly on new data

---

# L2 Regularization

\[
Loss = Original\ Loss + \lambda\sum w^2
\]

### Purpose
- Keeps weights small
- Makes model stable

### Keras Example

```python
Dense(
    64,
    activation='relu',
    kernel_regularizer=l2(0.001)
)
```

---

# 3. Dropout

Dropout randomly turns off neurons during training.

### Purpose
- Prevents overfitting
- Makes network independent

### Example

```python
Dropout(0.5)
```

Means:
- 50% neurons randomly off during training

---

# Dropout Placement

```text
Dense Layer
↓
Activation
↓
Dropout
```

---

# 4. Early Stopping

Early stopping stops training automatically when validation performance stops improving.

### Example

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
```

---

# Apply in model.fit()

```python
model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=50,
    callbacks=[early_stop]
)
```

---

# 5. Validation Concepts

## Validation Data

Used to check model performance during training.

Model learns from:
- Training data

Model is checked on:
- Validation data

---

# validation_split

```python
validation_split=0.2
```

Means:
- 20% training data becomes validation data

---

# validation_data

```python
validation_data=(X_val, y_val)
```

You manually provide validation data.

---

# Validation Loss

Error on validation data.

If:
- training loss decreases
- validation loss increases

Then:
- overfitting is happening

---

# Validation Accuracy

Accuracy on validation data.

Used to check generalization.

---

# 6. Accuracy

Measures correct predictions.

\[
Accuracy =
\frac{Correct\ Predictions}{Total\ Predictions}
\]

---

# 7. Precision

Question:
> Out of predicted positives, how many were actually positive?

\[
Precision =
\frac{TP}{TP+FP}
\]

### Important When
False alarms are dangerous.

Example:
- Spam detection
- Fraud detection

---

# 8. Recall

Question:
> Out of actual positives, how many were detected?

\[
Recall =
\frac{TP}{TP+FN}
\]

### Important When
Missing positive cases is dangerous.

Example:
- Cancer detection
- Disease prediction

---

# 9. F1 Score

Balance between precision and recall.

\[
F1 =
2\times
\frac{Precision\times Recall}
{Precision+Recall}
\]

---

# 10. Complete Keras Example

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()

model.add(
    Dense(
        20,
        activation='relu',
        kernel_regularizer=l2(0.001),
        input_dim=2
    )
)

model.add(Dropout(0.5))

model.add(
    Dense(
        15,
        activation='relu',
        kernel_regularizer=l2(0.001)
    )
)

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=50,
    callbacks=[early_stop]
)
```

---

# Quick Summary Table

| Concept | Main Purpose |
|---|---|
| ReLU | Hidden layer activation |
| Sigmoid | Binary output |
| L2 Regularization | Reduce large weights |
| Dropout | Prevent overfitting |
| Early Stopping | Stop overtraining |
| Validation Loss | Check generalization |
| Precision | Correct positive predictions |
| Recall | Detect actual positives |
| F1 Score | Balance precision & recall |
