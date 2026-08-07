# Chapter 5: Categorical Cross-Entropy loss — measures how far a predicted
# probability distribution (Softmax output) is from the true one-hot class.
# Since target_output is one-hot, this collapses to -log(predicted
# probability of the correct class): confident + correct -> loss near 0,
# confident + wrong -> loss shoots toward infinity.
import math
import numpy as np

softmax_output = [0.7, 0.1, 0.2]

target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2])

print(loss)

# Demonstrates why -log(confidence) makes a good loss: it grows sharply as
# confidence in the correct class drops toward 0, and is 0 at confidence 1.
print (math.log( 1. ))
print (math.log( 0.95 ))
print (math.log( 0.9 ))
print (math.log( 0.8 ))
print ( '...' )
print (math.log( 0.2 ))
print (math.log( 0.1 ))
print (math.log( 0.05 ))
print (math.log( 0.01 ))

b = 5.2

print(np.log(b))


print(math.e ** 1.6486586255873816)



# Extending to a batch of 3 samples, each with sparse class labels
# (class_targets holds the correct class index per sample, not one-hot).
softmax_output1 = [[0.7, 0.1, 0.2],
                   [0.1, 0.5, 0.4],
                   [0.02, 0.9, 0.08]]

class_targets = [0, 1, 1]

# Pull out just the predicted probability of the correct class per sample.
for targ_idx, distribution in zip(class_targets, softmax_output1):
    # loss = -math.log(distribution[targ_idx])
    print(distribution[targ_idx])


print('---')


softmax_output2 = np.array([[0.7, 0.1, 0.2],
                   [0.1, 0.5, 0.4],
                   [0.02, 0.9, 0.08]])

# Vectorized version of the loop above: fancy-indexing with
# ([row indices], [class index per row]) pulls the correct-class
# probability from every row in one shot.
print(softmax_output2[[0, 1, 2], class_targets])


print('---')


# -log() of each correct-class probability, then averaged across the
# batch to get a single scalar loss for the whole batch.
neg_log = -np.log(softmax_output2[range(len(softmax_output2)), class_targets])

average_loss = np.mean(neg_log)

print(average_loss)


print('------')


# Same calculation, but supporting one-hot encoded targets too: multiply
# elementwise by the one-hot mask and sum each row, which zeroes out every
# probability except the correct class's (equivalent to the fancy-index
# version above, just written to also handle 2D targets).
class_targets1 = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 1, 0]])

if len(class_targets1.shape) == 1:
    correct_confidences = softmax_output2[range(len(softmax_output2)), class_targets1]
elif len(class_targets1.shape) == 2:
    correct_confidences = np.sum(softmax_output2 * class_targets1, axis=1)

nsg_log2 = -np.log(correct_confidences)

average_loss2 = np.mean(nsg_log2)

print(average_loss2)


# Considering the Log of 0, which is undefined, we can use the following to prevent it from happening:
# -log(0) is infinity, which would break training (nan losses/gradients),
# so predictions get clipped away from exactly 0 and exactly 1 below.
print(-np.log(0))

print(np.e**(-np.inf))


print([1, 2, 3, -np.log(0)])

print(-np.log(1e-7))
print(-np.log(1-1e-7))


y_pred = np.array([
    [0.7, 0.1, 0.2],
    [0.1, 0.5, 0.4],
    [0.02, 0.9, 0.08]
])

# Clip predictions to [1e-7, 1-1e-7] — small enough not to affect the loss
# meaningfully, but avoids -log(0) (undefined) and -log(1) edge cases from
# ever occurring. Clipping both sides keeps the mean from being dragged
# toward one extreme.
y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

print('---------------')


# Reusable Loss base class: any subclass just needs to implement forward()
# (per-sample losses), and calculate() handles averaging into one scalar.
class Loss:
    def calculate(self, output, y):

        sample_losses = self.forward(output, y)

        data_loss = np.mean(sample_losses)

        return data_loss


# Categorical Cross-Entropy loss, packaged as a reusable class supporting
# both sparse (1D) and one-hot (2D) targets.
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)

        return negative_log_likelihoods


    
class_targets = np.array([0, 1, 1])
loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(softmax_output2, class_targets)

print(loss)