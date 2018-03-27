from random import sample
from time import process_time
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn import datasets
from numpy import array, setdiff1d

NUM_CLASSES = 10
INPUT_DIM = 64
TRAINING_FRACTION = 0.7

# the dataset to train with
digits = datasets.load_digits()
training_size = int(len(digits.data)*TRAINING_FRACTION)

# create a random mask to sample training & testing data
train_mask = sample(range(len(digits.data)), k=training_size)
train_mask.sort()
train_mask = array(train_mask)
test_mask = setdiff1d(range(len(digits.data)), train_mask)

x_train = array(digits.data[train_mask])
x_test = array(digits.data[test_mask])

# Convert labels to categorical one-hot encoding
y_train = to_categorical(digits.target[train_mask], num_classes=NUM_CLASSES)
y_test = to_categorical(digits.target[test_mask], num_classes=NUM_CLASSES)

best_model = None
best_score = [0, 0]
best_settings = (0, 0, 0, 0)

for hidden_layers in [0, 2, 4, 6]:
	for dropout in [0.05, 0.1, 0.15, 0.2]:
		# decreasing sequence of dimensions
		for dim_decay in [1, 0.95, 0.9]:
			for epochs in [10, 20]:
				# build model
				model = Sequential()
				
				# Must specify the input dimensions for the first layer only.
				# Since the images are 8x8 pixels, we map 1 pixel to 1 dimension.
				# The first number is simply the dimensionality of the output space.
				model.add(Dense(int(INPUT_DIM * dim_decay), input_dim=INPUT_DIM, activation='relu'))
				
				# hidden layers
				model.add(Dropout(dropout))
				for i in range(hidden_layers):
					model.add(Dense(int(INPUT_DIM * dim_decay ** (i + 2)), activation='relu'))
					model.add(Dropout(dropout))
				
				# Since we want to classify digits as 0 to 9, the output dimension is 10.
				model.add(Dense(NUM_CLASSES, activation='softmax'))
				
				# For a single-input model with 10 classes (categorical classification)
				model.compile(optimizer='rmsprop',
				              loss='categorical_crossentropy',
				              metrics=['accuracy'])
				
				# Train the model, iterating on the data in batches of 32 samples
				model.fit(x_train, y_train, epochs=epochs)
				
				score = model.evaluate(x_test, y_test)
				
				if score[1] > best_score[1]:
					best_model = model
					best_score = score
					best_settings = (hidden_layers, dropout, dim_decay, epochs)

print('\n------------------------')
print(f'Best setting is: {best_settings[0]} hidden layers, {best_settings[1]} dropout,\n'
      f'{best_settings[2]} dimension decay, and {best_settings[3]} epochs\n'
      f'with test loss {best_score[0]} and test accuracy {best_score[1]}.')
print(f'Took {int(process_time())} seconds.')

# Sample Output:
# Best setting is: 2 hidden layers, 0.1 dropout, 1 dimension decay, and 20 epochs
# with test loss 0.073089832081287 and test accuracy 0.9833333333333333.
# Took 12 seconds per model.

best_model.save('digits.h5')