# %% [markdown]
# ### Homework 3 — CompSci 389 — University of Massachusetts — Spring 2024
# Assigned: March 11, 2024;  Due: March 18, 2024@ 11:59 PM EST
# 
# ## Instructions!
# 
# First up, we'll be using a new library today, PYTORCH! -- you'll need to install this library, you can find instructions to install it [here](https://pytorch.org/get-started/locally/). Some Windows users have issue using pip to install it so I recommend in that case to use [anaconda](https://docs.conda.io/en/latest/miniconda.html). 
# 
# We're going to do some vision stuff today -- that means we get to make and use networks on stuff we can actually look at! Exciting!
# 

# %%
# Step 1: Let's import some libraries and say hi!

import numpy as np     # <----- Our old friend

import torch            # |  Our new best friends -- this is the main pytroch library
import torch.nn as nn   # |  This is just shortening the name of this module since we're gonna use it a lot -- this is the one that has neural network objects (nn.modules)
import torchvision      # |  This is for importing the vision datasets we'll use
from torch.utils.data import Dataset, DataLoader, random_split, Subset, ConcatDataset # | These are particular objects that we use to load our data (and shuffle it and whatnot) we'll talk more about these later
import torchvision.transforms as tt # | Allows us to transform our data while we load it (or after) such as rotating, flipping, ocluding, etc. 
from torchvision.datasets import ImageFolder # | ^^ less important for you


import torch.nn.functional as F # | This is for functional / in-place operations for example if I wanted to do a sigmoid operation, but not as a neural net object (though I can still update through it)



from torchvision.utils import make_grid  # |   Utility stuff for plotting
import matplotlib.pyplot as plt          # |  <- I use this one a lot for plotting, seaborn is a good alternative
from matplotlib.image import imread      # |  it reads images... (png -> usable input (like a numpy array for ex))
import os
import random
from tqdm import tqdm  # | This one is a cute one for making a loading bar, I like it and we'll use it here


# %% [markdown]
# ## Our Dataset
# You remeber our good friend MNIST? Well it's about time that you two get acquainted. MNIST is a dataset of 60,000 training and 10,000 testing images of handwritten digits, with (human done) labels of which digit is written. You can find the MNIST official website [here](http://yann.lecun.com/exdb/mnist/) and a description a little less 80's [here](https://deepai.org/dataset/mnist)
# 
# Notably this dataset has no colors -- and because of how (relatively) simple this dataset is we consider this to be the "hello world" of vision datasets.
# 
# Here's what the dataset looks like to us humans:
# 
# <center><img src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftheanets.readthedocs.io%2Fen%2Fstable%2F_images%2Fmnist-digits-small.png&f=1&nofb=1" width="350" height="250" /><center>
# 
# 
# 
# 
# 

# %% [markdown]
# ## Datasets, Dataloaders (10 points)
# In this assignment you will be loading your own data set in order to practice working with *Datasets* in PyTorch. To iterate over your data, PyTorch has two very helpful mechanisms: ```Dataset``` and ```Dataloader```. A ```Dataset``` is a type of PyTorch class which makes it easier to access your data. 
# 
# Here's a reference API that you should 100% use for [Datasets and Dataloaders](https://pytorch.org/vision/0.8/datasets.html#mnist)
# 
# You can create an object of a ```Dataset()``` class, and then access the size of the data set as ```len(dataset)```.  You can access the actual data at any given index by calling ```dataset[index]```. 
# 
# You can also apply *transformations* to the dataset, which are created by calling some predefined functions in PyTorch (or Torchvision) called [transforms](https://pytorch.org/vision/stable/transforms.html). For this data and HW the only transform you need to use is ```ToTensor()``` which will give us our data as a Tensor (a generalization of scalar, vector, and matrix with arbitrary dimesnions), which will allow us to do gradient descent with Pytorch
# 
# Once you create a ```Dataset``` class with your needed transformations, you can feed it into a ```Dataloader```. A ```Dataloader``` is an iterable over the ```Dataset``` (so we can for loop through our dataset (and batch + shuffle it)). For example, if your ```Dataset``` returns a (sample, target) pair, then you can iterate over the ```Dataloader``` as:
# 
# ```
# train_ds = MyDatasetClass()
# train_dl = Dataloader(train_dl, batch_size)
# for input, output in train_dl:
#     ...
# ```
# Remember that the sizes of ```input``` and ```output``` are specified by the ```batch_size``` that you selected earlier!

# %%
# dataset and dataloader
# If you're having trouble here go look at the API from above

def load_mnist(batch_size=32, train=True):

    '''
    Using the dataset and dataloader classes you should be able to make an MNIST set and loader
    the loader should use the 'batch_size' argument and the dataset should use'train'

    Also, the 'ToTensor' transform is given, you should set the transform of the dataset to just this
    '''

    to_tensor_transform = torchvision.transforms.ToTensor()
    # TODO create a dataset and then dataloader object for MNIST using
    # the torchvision library

    dataset = torchvision.datasets.MNIST(root='./df', train=train, transform=to_tensor_transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train)

    #############################################
    

    ##############################################

    return dataset, dataloader


# %% [markdown]
# ### Now let's see what our data looks like!

# %%
def plot_image_and_label(image, label):
    
    '''
    Takes in an image and label and shows them using matplotlib 
    this is used to visualize the data and also the outputs of our network
    '''

    plt.imshow(image)
    if type(label) is not int:
        _,predicted = torch.max(label,1)
        plt.title("Best label = " + str(predicted.item()) + ", with Score: " + str(round(label[0][predicted].item() * 100,2)))
    else:
        plt.title("Label = " + str(label))
    plt.show()
    return

# %%
# This will just test whether your dataset and loader work 
# They might still have issues, but if an example image shows here you're on the right track!

train_dataset, train_dataloader = load_mnist(batch_size=1, train=True)
ex_image, ex_label = train_dataset[random.randint(0,1000)]
plot_image_and_label(ex_image.reshape(28,28), ex_label)

# %% [markdown]
# ## Models with Pytorch (17 points)
# 
# So here's the part where you're going to love me and then hate me when you realize I didn't let you use this on HW2.
# 
# Pytorch is a library that will allow you to define neural network objects (nn.modules) for your entire network of whatever operations you want, select a loss function, and then ...
#  automatically calculate the gradient for you
# 
# Pytorch is built off of modules (called ```nn.Module```) which consist of 2 parts: The initialization (defined in ```__init__()``` -- note that this the python convention for initalizing classes) and the forward pass (defined aptly as ```forward()```)
# 
# What is magical about Pytorch is that you simply define these two things and then the gradient can be found *auotmatically*. So all that grueling code you wrote last time... totally unnecessary now. It still helps you in the long run though -- I promise. 
# 
# Documentation for a pytorch module can be found [here](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
# 
# In our first model we will just be creating a perceptron which will use a single ```nn.Linear()``` module -- you can find documentation for that [here](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
# 
# In later models we'll use nonlinearities (and that neat convolution thing) -- documentation ```for nn.ReLU()``` can be found [here](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)
# 
# 
# 
# 
# 
# 
# 
# 

# %%
class MyPerceptron(nn.Module):

    '''
    This is your very first Pytorch module -- in init you should initialize your model
    in forward you just need to use the layers initialized in init to get the output of your model

    Documentation will help quite a bit if you're stuck, you should only need to write 1 line of code per TODO
    '''

    def __init__(self, input_size, output_size):
        super(MyPerceptron, self).__init__() 

        self.input_size = input_size # making the input size accessible

        # TODO initalize your layer here 
        # that makes up the model -- use nn.Linear
        #################################

        self.lin = nn.Linear(input_size, output_size)

        #################################


    def forward(self, x):

        x = x.view(-1, self.input_size)   # This reshapes the input to work with the batches, (the -1 will become any necessary shape to make the other dim work, here batches)

        # TODO perform the forward pass of you model 
        # use the module you initialized above
        #################################

        out = self.lin(x)

        #################################

        return out



# %%
# This takes our randomly initialized perceptron and sees its prediction on a random input from MNIST
test_model = MyPerceptron(784, 10)
test_output = test_model(ex_image.flatten()) # .flatten() here is converting the set of 2d matrices into a batch of vectors (so it can be passed through a matrix mult operation)

plot_image_and_label(ex_image.reshape(28,28), test_output)

# %%
class MyMLP(nn.Module):

    '''
    Now that you're an expert in pytorch you can make a slightly more serious model

    You can include as many linear layers as you'd like (I used 3 in my testing), but make sure to have a ReLU
    in between each (but not at the end)

    I also included a SoftMax (which acts similarly to a max but returns values which add up to 1 -- 
    and is differentiable) at the end of the model to allow it to compare to MNIST labels
    '''


    def __init__(self, input_size, output_size):
        super(MyMLP, self).__init__()

        self.input_size = input_size # making the input size accessible

        # TODO initalize your layers here 
        # that makes up the model -- use nn.Linear and nn.ReLU
        # this would be convention of what we would need to init for a 3 layer network with nonlinearities
        # Note that I onlt defined one nonlinearity (cause they dont have parameters)
        #################################

        self.lin1 = nn.Linear(input_size, 500)
        self.lin2 = nn.Linear(500, 500)
        self.lin3 = nn.Linear(500, output_size)

        self.relu = nn.ReLU()

        #################################


    def forward(self, x):

        x = x.view(-1, self.input_size)   # This reshapes the input to work with the batches

        # TODO perform the forward pass of you model 
        # use the modules you initialized above (each should be used)
        #################################

        out = self.lin3(self.relu(self.lin2(self.relu(self.lin1(x)))))

        #################################

        return out

# %%
# Shows the prediction of the model without training
# Not very good huh? (though theres a small chance it is lol)

test_model = MyMLP(784, 10)
test_output = test_model(ex_image.flatten()) # Notice how we flatten the 2d image into 1d to use the MLP

plot_image_and_label(ex_image.reshape(28,28), test_output)

# %% [markdown]
# ### Choosing Loss and Optimizer (5 points)
# 
# Before we go training our model we need to define *what* we are optimizing -- we know this as out loss function. Take a look at the documentation for different loss functions included in pytorch [here](https://pytorch.org/docs/stable/nn.html#loss-functions). I recommend you use CrossEntropy since it will automatically convert the interger labels into a 10 long vector, but up to you
# 
# We also need to define *how* we are updating. We learned basic SGD (stochastic gradient descent) in class, but there are other options. Check out the documentation for included optimizers [here](https://pytorch.org/docs/stable/optim.html)
# 
# Anyway, I gave you some code that does a single update using the loss and optimizer you fill in -- if this doesn't work then you did something wrong lol
# 
# 
# 

# %%
## Fill in the loss_function and optimizer below and run this cell to see if they are valid!

model = MyMLP(784, 10)                                        ## This is your model, no need to change this                                            

# TODO fill out the loss_function and optimizer

#############################################

loss_function = nn.CrossEntropyLoss()                        ## You should use CrossEntropyLoss, use the API to decide how to define this 
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)                            ## You can use SGD for this, which is defined in torch.optim -- look up some API stuff

#############################################

# This checks that your model, loss and optimizer are valid -- checkout what they print!
print("BEFORE GRADIENT STEP:")
ex_pred = model(ex_image.flatten()) 
print("prediction:",ex_pred)
ex_label = torch.Tensor([1]).long()
print("label:",ex_label)


optimizer.zero_grad() # Sets the gradient to 0 so that gradients don't stack together

ex_loss1 = loss_function(ex_pred, ex_label)
print("loss",ex_loss1.item())

ex_loss1.backward() # This gets the gradient of the loss function w.r.t all of your model's params

print()
print("AFTER GRADIENT STEP:")
optimizer.step() # This takes the step to train

ex_pred = model(ex_image.flatten())
print("prediction:",ex_pred)
ex_label = torch.Tensor([1]).long()
print("label:",ex_label)

ex_loss2 = loss_function(ex_pred, ex_label)
print("loss",ex_loss2.item())

print()
print("Difference in loss:", (ex_loss1 - ex_loss2).item())
print("This should be some positive number to say we reduced loss")


# %% [markdown]
# ### Training loop (10 points)
# Now you are finally ready to train your neural network! Complete your ```training()``` function. You can iterate over your data for 30 epochs (epochs = number of times you iterate over all your data) to begin with. In order to write your training loop, the following general structure can be followed:
# ```
# # initialize loss_function and optimizer
# model = MyMLP()
# for iteration in range(epochs):
#     for input, output in train_dl:
#         reset optimizer gradients 
#         my_output = model(input)
#         loss = loss_function(my_output, loss)
#         step over gradients using optimizer
# ```
# 
# 
# [Hint for reseting the optimizer](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html#torch.optim.Optimizer.zero_grad)
# 
# [Hint for stepping with the optimizer](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.step.html#torch.optim.Optimizer.step) (You'll have to use .backward() to get the gradient)
# 
# At this point you should record your training and validation *losses* and *accuracies* **(four lists in total)**. You'll need these values for the written section, where you will be plotting them.

# %%
# training function here

def training(model, loss_function, optimizer, train_dataloader, n_epochs, update_interval):

    '''
    Updates the parameters of the given model using the optimizer of choice to
    reduce the given loss_function

    This will iterate over the dataloader 'n_epochs' times training on each batch of images
    
    To get the gradient (which is stored internally in the model) use .backward() from the loss tensor
    and to apply it use .step() on the optimizer

    In between steps you need to zero the gradient so it can be recalculated -- use .zero_grad for this
    '''
    
    losses = []
    # model.train()

    for n in range(n_epochs):
        print(f'{n}th epoch')
        for i, (image, label) in enumerate(tqdm(iter(train_dataloader))):

            # TODO Complete the training loop using the instructions above
            # Hint: the above code essentially does one training step

            ##############################################################
            optimizer.zero_grad()
            my_output = model(image)
            loss = loss_function(my_output, label)
            loss.backward()
            optimizer.step()


            ##############################################################
        
            if i % update_interval == 0:
                # print("--------------&&&&&&&&&&&--------------")
                losses.append(round(loss.item(), 2)) # This will append your losses for plotting -- please use "loss" as the name for your loss
        
    return model, losses



# %%
# Plug in your model, loss function, and optimizer 
# Try out different hyperparameters and different models to see how they perform

lr = 0.1              # The size of the step taken when doing gradient descent
batch_size = 128        # The number of images being trained on at once
update_interval = 100   # The number of batches trained on before recording loss
n_epochs = 8            # The number of times we train through the entire dataset

train_dataset, train_dataloader = load_mnist(batch_size=batch_size, train=True)
# train_dataset.data = train_dataset.data.to(device)
# train_dataset.targets = train_dataset.targets.to(device)
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = MyMLP(784, 10)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

trained_model, losses = training(model, loss_function, optimizer, train_dataloader, n_epochs=n_epochs, update_interval=update_interval)

plt.plot(np.arange(len(losses)) * batch_size * update_interval, losses)
plt.title("training curve")
plt.xlabel("number of images trained on")
plt.ylabel("loss")
plt.show()


# %%
# Displays the prediction of the (now trained) model on the same example image
# Notice that it worked and we have a better prediction (if your code works)
trained_output = trained_model(ex_image.flatten()) # Notice how we flatten the 2d image into 1d to use the MLP

plot_image_and_label(ex_image.reshape(28,28), trained_output)

# %% [markdown]
# ### Testing 
# 
# Since the testing loop and training loop are so similar I'm going to go ahead and just give it to you -- but you gotta promise to at least look at the method to see how similar they are! 

# %%
def test_accuracy(model, loss_function, test_data):

    '''
    This function will test the given model on the given test_data
    it will return the accuracy and the test loss (given by loss_function) 
    '''
    
    sum_loss = 0
    n_correct = 0
    total = 0


    for i, (image, label) in enumerate(tqdm(iter(test_data))):

        # This is essentially exactly the same as the training loop 
        # without the, well, training, part (and we record the accuracy too)
        pred = model(image)
        loss = loss_function(pred, label)
        sum_loss += loss.item()

        _, predicted = torch.max(pred,1)
        n_correct += (predicted == label).sum()
        total += label.size(0)
    
    test_acc = round(((n_correct / total).item() * 100), 2)
    avg_loss = round(sum_loss / len(test_data), 2)

    print("test accuracy:", test_acc)
    print("test loss:", avg_loss )

    return test_acc, avg_loss

# %%
# To see how well your model is doing without hyperpameter tuning

print("testing the previously trained model on test dataset of MNIST")
test_dataset, test_dataloader = load_mnist(batch_size=10000, train=False)
test_acc,avg_loss = test_accuracy(trained_model, loss_function, test_dataloader)

print("Testing accuracy of your first model:", test_acc)
print("Average loss of your first model:", avg_loss)

# %% [markdown]
# ### Pytorch Question (2 points)
# Now, lets take a bit of break from coding and do some writing (I know you all love that right?)
# Fill out your answer  in the empty markdown cell below 

# %% [markdown]
# Imagine we didn't zero our gradient inbetween update steps in our MLP. Predict what we would expect from the model -- then (after you're commited to a prediction) try commenting it out to see what happens. Why do you think it behaves this way? 

# %% [markdown]
# If we did not zero out our gradients in each of our update steps during train, the gradients would accumualte at every step. Each step's backpropogated gradinet would be added to the previous step's backpropogated gradient, leading to incorrect values and incorrect training. If the gradient becomes too large, the model will start taking very big steps in order to minimze loss, and too large steps might skip over minimas. In some cases the gradients of different steps might also add in a way such that the overall gradient becomes zero, hence making the model take very niglible small steps. After commenting out the `optimizer.zero_grad()` 

# %% [markdown]
# ### K Fold Validation (12 points): 
# Now we get to automate our hyperpameter tuning too! 
# 
# Refer to the slides for the K-fold validation algorithm, you will need to choose which search method you'd like to use as well as the metric to choose your best model by (and your K value too!) 

# %%
def k_fold_split(dataloader, k):
    dataset = dataloader.dataset
    n_samples = len(dataset)
    fold_size = n_samples // k
    indices = list(range(n_samples))
    folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(k)]
    fold_loaders = [DataLoader(Subset(dataset, fold), batch_size=dataloader.batch_size) for fold in folds]
    return fold_loaders


def generate_random_hyperparams(param_ranges, num_random_combs):
    hyperparams = []
    for i in range(num_random_combs):
        hyperparam = {}
        for key, value in param_ranges.items():
            hyperparam[key] = random.choice(value)
        hyperparams.append(hyperparam)
    return hyperparams


def K_fold_validation(model, K=3):
    '''
    This function will take a model, a training dataset, and an int K and will perform
    K-fold validation using the 'test_accuracy' function above

    It returns both the best model (after being trained) and the losses recorded during its training

    Each of the K dataset portions should be used exactly once as the validation set and should be
    uniformly randomly selected portions (decide if it is better with or without replacement)  
    '''
    # TODO Implement K_fold validation
    # The 'test_accuracy' function returns both the test accuracy and the average loss, choose how to use them to pick the best model
    # Try both grid (slow) and randomized (fast) searches -- be careful about your run time, grid search can take a LONG time if you get carried away. 
    ##############################################################
    # _, train_dataloader = load_mnist(batch_size=batch_size, train=True) # This will be stuck as mnist for now, but you can change this for the later Extra credit to use CIFAR10
    
    best_model = None
    best_losses = []
    best_acc = 0
    param_ranges = {'lr': np.arange(0.08, 0.11, 0.01), 'batch_size': [int(i) for i in np.arange(140, 160, 5)], 'n_epochs': [int(i) for i in np.arange(5, 12, 2)]}
    random_hyperparams = generate_random_hyperparams(param_ranges, 3)
    print(random_hyperparams)
    best_hyperparams = random_hyperparams[0]  # You may add more hyperparmeters (though it may require you to change your 'training' function)
    count_hyperparam = 0
    for hyperparams in random_hyperparams:
        count_hyperparam += 1
        print("Current Hyperparam number: ", count_hyperparam)
        lr = hyperparams['lr']
        batch_size = hyperparams['batch_size']
        n_epochs = hyperparams['n_epochs']
        print("lr:", lr, "batch_size:", batch_size, "n_epochs:", n_epochs)
        for i in range(K):
            print("Fold:", i)
            model.__init__(784, 10)
            folds = k_fold_split(train_dataloader, K)
            k_fold_test_dataloader = folds[i]
            k_fold_train_dataloader = DataLoader(ConcatDataset([folds[j].dataset for j in range(K) if j != i]), batch_size=batch_size)
            loss_function = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            trained_model, losses = training(model, loss_function, optimizer, k_fold_train_dataloader, n_epochs=n_epochs, update_interval=10)
            test_acc, avg_loss = test_accuracy(trained_model, loss_function, k_fold_test_dataloader)
            if test_acc > best_acc:
                best_acc = test_acc
                best_model = trained_model
                best_losses = losses
                best_hyperparams['lr'] = lr
                best_hyperparams['batch_size'] = batch_size
                best_hyperparams['n_epochs'] = n_epochs
    ##############################################################
    
    return best_model, best_losses, best_hyperparams

# %%
# Now let's see how much better you do with K-fold validation!

model = MyMLP(784, 10)
best_model, best_losses, best_hyperparams = K_fold_validation(model, K=3) # You can choose the value of K however you'd like

print("testing the previously trained model on test dataset of MNIST")
test_dataset, test_dataloader = load_mnist(batch_size=10000, train=False)
test_acc,avg_loss = test_accuracy(trained_model, loss_function, test_dataloader)

print("Testing accuracy of your first model:", test_acc)
print("Average loss of your first model:", avg_loss)

print("testing your hyperparam tuned best model on test dataset of MNIST")
best_acc,best_avg_loss = test_accuracy(best_model, loss_function, test_dataloader)

print("Testing accuracy of your best model:", best_acc)
print("Average loss of your best model:", best_avg_loss)

# %%
# Recording the best hyperparameters, might need these later

print("best batch size:", best_hyperparams['batch_size'])
print("best learning rate:", best_hyperparams['lr'])
print("best epoch count:", best_hyperparams['n_epochs'])

# print(best_hyperparams) # If you added any hyperparameters to your search uncomment this or add a print statement for each hyperparameter

# %%
plt.plot(np.arange(len(best_losses)) * best_hyperparams['batch_size'] * update_interval, best_losses, label="non-Tuned hyperparameters")
plt.plot(np.arange(len(losses)) * batch_size * update_interval, losses, label="tuned hyperparam")
plt.title("training curve")
plt.xlabel("number of images trained on")
plt.ylabel("loss")
plt.legend()
plt.show()

# %%
# Choose hyperparameters such that your final test accuracy is over 80%

## TODO Set each of these to whatever were the best options you found above
    
# We'll be seeing if we can retrain with the same hyperparameter values to get the same quality of model, put your guesses in now if it will work!

##############################################################

lr = best_hyperparams['lr']
batch_size = best_hyperparams['batch_size']
n_epochs = best_hyperparams['n_epochs']

##############################################################

update_interval = 10   # The number of batches trained on before recording loss

model = MyMLP(784, 10)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

retrained_model, retrained_loss = training(model, loss_function, optimizer, train_dataloader, n_epochs, update_interval)
retrain_acc, _ = test_accuracy(retrained_model, loss_function, test_dataloader)

print("Accuracy of the model retrained with the best param settings:", retrain_acc)


# %% [markdown]
# ## K-Fold Written Question (6 points)
# Fill out your answer  in the empty markdown cell below 
# 
# What value of K do you think would perform best for a K-fold validation on an N element training set? Is there any downside to using your choice of K? Justify your answer (There are pretty deep mathematical justifications for answers to this question, which are outside the scope of 389 -- you should either have a moderate mathematical justification or an intuition and experimental result).  

# %% [markdown]
# Based on my experiments, I found that the values of K that work best for my model were between 8 and 10 (however that took quite a long time to run). I believe that this would allow my model to get trained on between around 80-90% of the data, and get tested on the rest, which would allow the model to not overfit or underfit, and would help it generalize on the data it is being trained on. The downside of the choosing a bad K value would be choosing a K that is too high or low. For example, choosing a K value that is equal to the length of the dataset will lead to overfitting, as the model would be using almost all of the training data to train the model. The downside our using a value of K between 8-10 is that incase of large datasets, the model would take a very long time to train (which I can see even right now)

# %% [markdown]
# ## Convolutional Neural Networks (20 points)
# 
# Now it's time for us to make a Convolutional Neural Net (CNN)! 
# 
# Fundementally, CNNs are made the same way as MLPs using Pytorch (it really is great isn't it). The only difference is that we are going to be using a new type of layer. In our case we are using ```Conv2d``` layers which preform the convlution we learned from class. You can learn more about them [here](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
# 
# For this model you must use at least 2 layers of ```Conv2d``` with a ```ReLU``` inbetween. You can include more layers if you'd like, but you need at least 2. The output of these will be passed into a ```MyMLP```. You must make sure that the input size of the ```MyMLP``` is correct so that it matches the output size of your Convolutions when they are ```flattened``` (flattened into a single dimension -- cause that's what Linear layers can take as an input)
# 

# %%
class MyCNN(nn.Module):

    def __init__(self, input_size, output_size):
        super(MyCNN, self).__init__()

        self.input_size = input_size # making the input size accessible

        # TODO initalize your layers here 
        # that makes up the model -- use nn.Conv2d : https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # Your model can use multiple conv2d layers, but keep track of the dimensions -- they can be tricky
        #################################

        # You can use more if you'd like, but you need at least 2 conv2d layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()       # you only need to define one relu, you can use this one
        
        cnn_out_size = 28*28*64         # You need to find what size input MLP you should use (hint: find the overall size of the output of the convs)
                                    # Hint: This will relate to the size and number of filtered images outputed by your last conv layer
                                    
        #################################

        self.mlp = MyMLP(cnn_out_size, output_size) # No need to change this


    def forward(self, x):

        x = x.view(-1, 1, 28, 28)   # This reshapes the input to work with the batches

        # TODO perform the forward pass of you model 
        # use the modules you initialized above (each should be used)
        # You should also include the self.relu between each layer (including conv2d ones)
        #################################

        out = self.conv2(self.relu(self.conv1(x)))




        #################################

        self.filtered = out  ## Saving the output from the convolutions so that we can see them 
        out = out.flatten()  # This is the flattening that we keep talking about (note that it will still be a batch on outputs)
        out = self.mlp(out)  

        return out

# %%
# This will show the output of our model before training

test_model = MyCNN(784, 10)
test_output = test_model(ex_image)

plot_image_and_label(ex_image.reshape(28,28), test_output)

# %% [markdown]
# ### This is our model's prediction before training ^
# 
# Let's train it to see how it compares! Tune the hyperparameters below until you have test accuracy better than your MLP above 
# 
# Notice how each of the hyperparameters (lr, batch_size, and n_epochs) affect the behavior and training of the model

# %%

model = MyCNN(784, 10)                                          # Your Model for the CNN
loss_function = nn.CrossEntropyLoss()                           # This is a standard loss function for situations like this -- check it out on the API!
optimizer = torch.optim.Adam(model.parameters(), lr=lr)       # This is an improved version of SGD which decreases the learning rate over time to avoid leaving a minima

trained_cnn_model, cnn_losses, cnn_best_hyperparams = K_fold_validation(model, K=2)  # Let's use our implementation of k_fold_validation!

# %%
print("testing your tuned MLP on test dataset of MNIST")
test_dataset, test_dataloader = load_mnist(batch_size=10000, train=False)
best_acc,best_avg_loss = test_accuracy(best_model, loss_function, test_dataloader)

print("Testing accuracy of your best MLP:", best_acc)
print("Average loss of your best MLP:", best_avg_loss)

print("testing your hyperparam tuned CNN model on test dataset of MNIST")
cnn_acc, cnn_avg_loss = test_accuracy(trained_cnn_model, loss_function, test_dataloader)

print("Testing accuracy of your best CNN:", cnn_acc)
print("Average loss of your best CNN:", cnn_avg_loss)

# %%
plt.plot(np.arange(len(losses)) * batch_size * update_interval, losses, label="non-tuned hyperparam (MLP)")
plt.plot(np.arange(len(best_losses)) * best_hyperparams['batch_size'] * update_interval, best_losses, label="Tuned hyperparameters (MLP)")
plt.plot(np.arange(len(cnn_losses)) * cnn_best_hyperparams['batch_size'] * update_interval, cnn_losses, label="Tuned hyperparameters (CNN)")
plt.title("training curve")
plt.xlabel("number of images trained on")
plt.ylabel("loss")
plt.legend()
plt.show()

# %%
# Let's see what it thinks now
# Try to get the test accuracy as high as you can! 
# You need over 90% for full credit (it's definitely possible)

test_output = trained_cnn_model(ex_image)

plot_image_and_label(ex_image.reshape(28,28), test_output)

# %% [markdown]
# ### This is our model's prediction after training ^
# 
# Try changing the hyperparameters until you have a final test accuracy over 95% !
# 
# Now let's take a look at what the model filtered out of the image above!

# %%
def display_filters(params, name):

    '''
    This is a little bit of code I wrote that will plot out an arbitrary
    number of images in one big plot so that it works for whatever size conv you used
    '''


    fig = plt.figure(figsize=(10, 7)) ## You MIGHT need to change this if you made really big convolutions

    rows = len(params)//5 + 1

    for i,filter in enumerate(params):
        fig.add_subplot(rows, 5, i+1)
        plt.imshow(filter)
        plt.axis('off')
        plt.title(name + "filter #"+str(i))
    
    plt.show

# %%
# This is the example image fed through your convolutions
# Looks funky right? Why do you think it looks like that?

filtered = trained_cnn_model.filtered[0].detach().numpy()
display_filters(filtered, "")


# %%
# This code will show the weights of both (or the first two) convolution layers of your CNN
params = trained_cnn_model.parameters()
conv1_params = next(params)[:,0,:,:].detach().numpy() # 5 7x7 conv filters 
relu = next(params)
conv2_params = next(params)[:,0,:,:].detach().numpy() # 10 5x5 conv filters


# Code to plot the weights of the first convolution layer
display_filters(conv1_params, "Layer 1 ")

# Code to plot the weights of the second convolution layer
display_filters(conv2_params, "Layer 2 ")




