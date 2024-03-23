using MLDatasets
using Flux
using Flux: onehotbatch, onecold, crossentropy, throttle, @epochs
using Statistics: mean
using ProgressMeter
using BSON: @save, @load

# Load the MNIST dataset
train_x, train_y = MNIST.traindata()
test_x, test_y = MNIST.testdata()

# Reshape and normalize the input data
train_x = reshape(float.(train_x), 28, 28, 1, :) ./ 255
test_x = reshape(float.(test_x), 28, 28, 1, :) ./ 255

# One-hot encode the labels
train_y = onehotbatch(train_y, 0:9)
test_y = onehotbatch(test_y, 0:9)

# Define the neural network architecture
model = Chain(
    Conv((3, 3), 1 => 16, pad=(1, 1), relu),
    MaxPool((2, 2)),
    Conv((3, 3), 16 => 32, pad=(1, 1), relu),
    MaxPool((2, 2)),
    Conv((3, 3), 32 => 64, pad=(1, 1), relu),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(576, 128, relu),
    Dense(128, 10),
    softmax
)

# Define the loss function and accuracy metric
loss(x, y) = crossentropy(model(x), y)
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

# Define the optimization algorithm and learning rate
optimizer = ADAM(0.001)

# Define the training loop
function train(epochs)
    @epochs epochs begin
        @showprogress for batch in Iterators.partition(shuffle(1:length(train_x)), 128)
            x, y = train_x[:, :, :, batch], train_y[:, batch]
            gs = gradient(() -> loss(x, y), params(model))
            Flux.update!(optimizer, params(model), gs)
        end
        train_acc = accuracy(train_x, train_y)
        test_acc = accuracy(test_x, test_y)
        println("Epoch: $epoch, Train accuracy: $train_acc, Test accuracy: $test_acc")
    end
end

# Train the model
train(10)

# Save the trained model
@save "mnist_model.bson" model

# Load the trained model
@load "mnist_model.bson" model

# Evaluate the model on the test set
test_loss = loss(test_x, test_y)
test_accuracy = accuracy(test_x, test_y)
println("Test loss: $test_loss")
println("Test accuracy: $test_accuracy")

# Make predictions on new data
new_data = test_x[:, :, :, 1:10]
predictions = onecold(model(new_data)) .- 1
ground_truth = onecold(test_y[:, 1:10]) .- 1
println("Predictions: $predictions")
println("Ground Truth: $ground_truth")

# Visualize the filters learned by the first convolutional layer
using Images
filters = model[1].weight
for i in 1:size(filters, 4)
    filter_img = filters[:, :, 1, i]
    save("filter_$i.png", colorview(Gray, filter_img))
end