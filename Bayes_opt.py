"""1. Setting search ranges for the parameters of interest.
"""

dim_learning_rate = Real(low=1e-4, 
                         high=1e-1,
                         prior='log-uniform',
                         name='learning_rate')
dim_num_dense_layers = Integer(low=1,
                               high=5,
                               name='num_dense_layers')
dim_num_dense_nodes = Integer(low=64,
                              high=512,
                              name='num_dense_nodes')
dim_activation = Categorical(categories=['relu', 'sigmoid'], name='activation')

dimensions = [dim_learning_rate, dim_num_dense_layers,
              dim_num_dense_nodes, dim_activation] #Defining a list with all the parameters
 
defaut_parameters = [0.1, 2, 128, 'relu'] #Setting initial parameters

"""2. Defining a funtion top log the training process to visualize.
"""

def log_dir_name(learning_rate, num_dense_layers,
                 num_dense_nodes, activation):
    s = "./19_logs/lr_{0:.0e}_layers_{1}_nodes_{2}_{3}/" #Directory name
    
    log_dir = s.format(learning_rate, num_dense_layers,
                       num_dense_nodes, activation) #Directory name + parameters
    return log_dir
  
  

  """3. Defining a neural netwotk create model function
  """

def create_model(learning_rate, num_dense_layers, num_dense_nodes, activation):
  """Returns a tensor flow sequential fully connected neural network.
  It uses a stochastic gradien descent SGD optimizer that will be used to find it global minima.
    Parameters
    ----------
    learning_rate: int
      Step size to be taken by the optimizer towards its minima.
     num_dense_layers: int
      Number of hidden layers to be included in the model.
     num_dense_nodes: int
      Number of neurons to be inlcuded in each hidden layer.
     activation: str
      Activation function, either 'relu' or 'sigmoid.

    Returns
    ----------
    model:
      Neural network ready to be trained.
    """
  
  model = Sequential()
  num_features = x_train_nn.shape[1]
  model.add(layers.InputLayer(input_shape=(num_features, )))

  for i in range(num_dense_layers):                        #Adding hidden layers
    name = 'layer_dense_{0}'.format(i+1)
    model.add(Dense(num_dense_nodes,                       #Input layer
                    activation=activation,
                    name=name,
                    kernel_initializer='random_normal',    #Random normal weight initialization
                    bias_initializer='zeros'))             #Zeros bias initialization
    
  model.add(Dense(12,                                      #Output layer - 12 outputs
                  activation='softmax',                    #Softmax activation fuction
                  kernel_initializer='random_normal',      #Random normal weight initialization
                  bias_initializer='zeros'))               #Zeros bias initialization

  opt = SGD(learning_rate=learning_rate,                   #Adam optimizer  
            momentum=0.1)                                  #momentum to help covergence

  model.compile(optimizer=opt,                             #Compiling the model
                loss='sparse_categorical_crossentropy',    #Sparse categorical crossentropy loss
                metrics=['accuracy'])                          
  return model



"""4. Defining optimization fitness function
"""
path_best_model = '19_best_model.h5' #Path where accuracy history will be stored
best_accuracy = 0 #Initializing global accuracy
validation_data = (x_test_nn, y_test) # Setting validations data 

def fitness(learning_rate, num_dense_layers,
            num_dense_nodes, activation):
  """Fintion to be iterated several times by calling the create_model function and fitting the sequential fully
  connected neural network on a different set of parameters for 7 epochs on each, then it stores the best result
  and parameters on the assigned directory. 
    Parameters
    ----------
    learning_rate: int
      Step size to be taken by the optimizer towards its minima.
     num_dense_layers: int
      Number of hidden layers to be included in the model.
     num_dense_nodes: int
      Number of neurons to be inlcuded in each hidden layer.
     activation: str
      Activation function, either 'relu' or 'sigmoid.

    Returns
    ----------
    -accuracy:
      The negative accuracy obtained on each set of hyper-parameters.
      The minus only alows us to treat the optimization as a minimization problem by using scikit optimizer skopt.
    """
  
    #Displaying selected hyper-parameters 
    print("""learning rate: {0:.1e},
              num_dense_layers: {},
              num_dense_nodes: {}, 
              activation: {}""".format(learning_rate, num_dense_layers, num_dense_nodes, activation))
   
    #Calling create_model function
    model = create_model(learning_rate=learning_rate,
                         num_dense_layers=num_dense_layers,
                         num_dense_nodes=num_dense_nodes,
                         activation=activation)

    #Storing parameters on the assigned directory
    log_dir = log_dir_name(learning_rate, num_dense_layers,
                           num_dense_nodes, activation)
    
    #Defining a call back to be called during training to avoid overfitting
    callback_log = TensorBoard(log_dir=log_dir,
                               histogram_freq=0,
                               write_graph=True,
                               write_grads=False,
                               write_images=False)
   
    #Fitting the model for 7 epochs while validating on the open test data 
    history = model.fit(x= x_train_nn,
                        y= y_train,
                        epochs=7,
                        batch_size=256,
                        validation_data=validation_data,
                        callbacks=[callback_log])

    # Displaying the validation accuracy after the 7th epoch
    accuracy = history.history['val_accuracy'][-1]
    print("Accuracy: {0:.2%}".format(accuracy))

    #Updating and storing the global accuracy if the selected hyper-parameters achieves to do so
    global best_accuracy
    if accuracy > best_accuracy:
        model.save(path_best_model)
        best_accuracy = accuracy

    #Clearing the model from memory before runnijng the next model
    del model
    K.clear_session()
    
    return -accuracy
# This function exactly comes from :Hvass-Labs, TensorFlow-Tutorials


"""5. Running optimization
"""
optimization = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI', # Expected Improvement.
                            n_calls=75,
                            x0=defaut_parameters)

plot_convergence(optimization) #Displaying best set of hyper-paremeters
sorted(zip(optimization.func_vals, optimization.x_iters)) #Plotting convergence
