import math 
import numpy as np

softmax_output = [0.7, 0.1, 0.2]

target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2])

print(loss)


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



softmax_output1 = [[0.7, 0.1, 0.2],
                   [0.1, 0.5, 0.4],
                   [0.02, 0.9, 0.08]]

class_targets = [0, 1, 1]

for targ_idx, distribution in zip(class_targets, softmax_output1):
    # loss = -math.log(distribution[targ_idx])
    print(distribution[targ_idx])
    
    
print('---')


softmax_output2 = np.array([[0.7, 0.1, 0.2],
                   [0.1, 0.5, 0.4],
                   [0.02, 0.9, 0.08]])


print(softmax_output2[[0, 1, 2], class_targets])


print('---')


neg_log = -np.log(softmax_output2[range(len(softmax_output2)), class_targets])

average_loss = np.mean(neg_log)

print(average_loss)


print('------')


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

y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

print('---------------')

class Loss:
    def calculate(self, output, y):
        
        sample_losses = self.forward(output, y)
        
        data_loss = np.mean(sample_losses)
        
        return data_loss
    
    
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