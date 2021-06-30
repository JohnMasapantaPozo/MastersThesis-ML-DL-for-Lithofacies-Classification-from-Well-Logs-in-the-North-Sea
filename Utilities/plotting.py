
"""Log plotting functions

This script holds different functions such as raw_plot, augmented_logs, and litho_predictions, 
which plot wireline raw logs, augmented logs, and lithology predictions, respecively. 
They can be imported as a modules when needed.

They require some functionalities from libraries such as  pandas, numpy, matplotlib,
and mpl_toolkits. 
"""

#PLOTTING RAW LOGS
def raw_logs(logs, well_num):

  """Plots the raw logs contained in the original datasets after they have been formated.

  Parameters
  ----------
  logs: dataframe
    The raw logs once the headers and necessary columns have been formated and fixed.
  well_num: int
    The number of the well to be plotted. raw_logs internally defines a list of weells 
    contained by the dataset, each of them could be called by its list index.

  Returns
  ----------
  plot:
    Different tracks having one well log each and a final track containing the 
    lithofacies interpretation.
  """

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    facies_colors = ['#F4D03F','#7ccc19','#196F3D','#160599','#2756c4','#3891f0','#80d4ff','#87039e','#ec90fc','#FF4500','#000000','#DC7633']
    facies_labels = ['SS', 'S-S', 'SH', 'MR', 'DOL','LIM', 'CH','HAL', 'AN', 'TF', 'CO', 'BS']

    facies_color_map = {} # creating facies color map
    for ind, label in enumerate(facies_labels):
        facies_color_map[label] = facies_colors[ind]

    wells = logs['WELL'].unique() # creating a wells list
    logs = logs[logs['WELL'] == wells[well_num]] # selecting well by index number
    logs = logs.sort_values(by='DEPTH_MD') # sorting well log by depth
    cmap_facies = colors.ListedColormap(facies_colors[0:len(facies_colors)], 'indexed')
    
    top = logs.DEPTH_MD.min()
    bot = logs.DEPTH_MD.max()
    
    real_label = np.repeat(np.expand_dims(logs['LITHO'].values, 1), 100, 1)

    f, ax = plt.subplots(nrows=1, ncols=17, figsize=(20, 12))
    log_colors = ['black', 'red', 'blue', 'green', 'purple','black', 'red', 'blue', 'green', 'purple', 'black',
                  'red', 'blue', 'green', 'purple', 'black', 'black', 'red', 'blue', 'green', 'purple', 'black',
                  'red', 'blue', 'green', 'purple', 'black']

    for i in range(7,23):
      ax[i-7].plot(logs.iloc[:,i], logs.DEPTH_MD, color=log_colors[i]) # plotting each well log on each track
      ax[i-7].set_ylim(top, bot)
      ax[i-7].set_xlabel(str(logs.columns[i]))
      ax[i-7].invert_yaxis()
      ax[i-7].grid()

    im = ax[-1].imshow(real_label, interpolation='none', aspect='auto', cmap=cmap_facies, vmin=0, vmax=12)
    ax[-1].set_xlabel('LITHO') # creating a facies log on the final track

    divider = make_axes_locatable(ax[-1]) # appending legend besides the facies log
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar=plt.colorbar(im, cax=cax)
    cbar.set_label((12*' ').join(['SS', 'S-S', 'SH', 'MR', 'DOL','LIM', 'CH','HAL', 'AN', 'TF', 'CO', 'BS']))
    
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
        
    f.suptitle('WELL LOGS '+str(wells[well_num]), fontsize=16,y=0.9)


#PLOTTING LOGS AUGMENTED BY ML
def augmented_logs(logs, well_num):

    """Plots the raw, predicted, and augmented wireline logs after applying data augmentation.

    Parameters
    ----------
    logs: dataframe
      The raw, predicted, and augmented logs.
    well_num: int
      The number of the well to be plotted. augmented_logs internally defines a list of 
      weells contained by the logs dataframe, each of which could be called by its list index.

    Returns
    ----------
    plot:
      Different tracks containing the raw, predicted, and augmented logs.
      Augmented logs mean that the missing values hbeen filled up by machine-learning
      predicted readings.
    """

    #auxiliar libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    facies_colors = ['#F4D03F','#7ccc19','#196F3D','#160599','#2756c4','#3891f0','#80d4ff','#87039e','#ec90fc','#FF4500','#000000','#DC7633']
    facies_labels = ['SS', 'S-S', 'SH', 'MR', 'DOL','LIM', 'CH','HAL', 'AN', 'TF', 'CO', 'BS']

    facies_color_map = {}  # creating facies color map
    for ind, label in enumerate(facies_labels):
        facies_color_map[label] = facies_colors[ind]

    wells = logs['WELL'].unique()
    logs = logs[logs['WELL'] == wells[well_num]] # selecting well by index number
    logs = logs.sort_values(by='DEPTH_MD') # sorting well log by depth
    cmap_facies = colors.ListedColormap(facies_colors[0:len(facies_colors)], 'indexed')
    
    top = logs.DEPTH_MD.min()
    bot = logs.DEPTH_MD.max()
    
    real_label = np.repeat(np.expand_dims(logs['LITHO'].values, 1), 100, 1)

    f, ax = plt.subplots(nrows=1, ncols=13, figsize=(20, 12))
    log_colors = ['black', 'red', 'blue', 'green', 'purple','black', 'red', 'blue', 'green', 'purple', 'black',
                  'red', 'blue', 'green', 'purple', 'black', 'black', 'red', 'blue', 'green', 'purple', 'black',
                  'red', 'blue', 'green', 'purple', 'black']

    for i in range(3,15):
      ax[i-3].plot(logs.iloc[:,i], logs.DEPTH_MD, color=log_colors[i]) # plotting raw, predicted, and augmented logs
      ax[i-3].set_ylim(top, bot)

      ax[i-3].set_xlabel(str(logs.columns[i]))
      ax[i-3].invert_yaxis()
      ax[i-3].grid()

    im = ax[-1].imshow(real_label, interpolation='none', aspect='auto', cmap=cmap_facies, vmin=0, vmax=12)
    ax[-1].set_xlabel('LITHO') # creating a facies log on the final track

    divider = make_axes_locatable(ax[-1]) # appending legend besides the facies log
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar=plt.colorbar(im, cax=cax)
    cbar.set_label((12*' ').join(['SS', 'S-S', 'SH', 'MR', 'DOL','LIM', 'CH','HAL', 'AN', 'TF', 'CO', 'BS']))
    
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
        
    f.suptitle('WELL LOGS '+str(wells[well_num]), fontsize=16,y=0.9)


#PLOTTING LITHOFACIES PREDICTION
def litho_prediction(logs, well_num, n_pred):

    """Plots the raw logs, the lihtology interpretation, and the n_pred number of predcted 
    lithologies by machine learning.

    Parameters
    ----------
    logs: dataframe
      Dataframe holding the raw wireline logs, true lithology, and n_pred columns 
      containing different ML model predictions each.
    well_num: int
      The number of the well to be plotted. litho_prediction internally defines a list of 
      weells contained by the logs dataframe, each of which could be called by its list index.

    Returns
    ----------
    plot:
      Different track plots representing each wireline log, the true lihtology and the 
      predicted lithologies by dfferent mane-learning models.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    facies_colors = ['#F4D03F','#7ccc19','#196F3D','#160599','#2756c4','#3891f0','#80d4ff','#87039e','#ec90fc','#FF4500','#000000','#DC7633']
    facies_labels = ['SS', 'S-S', 'SH', 'MR', 'DOL','LIM', 'CH','HAL', 'AN', 'TF', 'CO', 'BS']

    facies_color_map = {} #creating facies coorap
    for ind, label in enumerate(facies_labels):
        facies_color_map[label] = facies_colors[ind]

    wells = logs['WELL'].unique() # well names list
    logs = logs[logs['WELL'] == wells[well_num]]
    logs = logs.sort_values(by='DEPTH_MD') # sorting the plotted well logs by depth       
    cmap_facies = colors.ListedColormap(facies_colors[0:len(facies_colors)], 'indexed')
    
    top = logs.DEPTH_MD.min()
    bot = logs.DEPTH_MD.max()
       
    f, ax = plt.subplots(nrows=1, ncols=(12+n_pred), figsize=(20, 12))
    log_colors = ['black', 'red', 'blue', 'green', 'purple','black', 'red', 'blue', 'green', 'purple', 'black',
                  'red', 'blue', 'green', 'purple', 'black', 'black', 'red', 'blue', 'green', 'purple', 'black',
                  'red', 'blue', 'green', 'purple', 'black']

    for i in range(7,18):
      ax[i-7].plot(logs.iloc[:,i], logs.DEPTH_MD, color=log_colors[i]) # plotting continuous wireline logs
      ax[i-7].set_ylim(top, bot)
      ax[i-7].set_xlabel(str(logs.columns[i]))
      ax[i-7].invert_yaxis()
      ax[i-7].grid()

    for j in range((-1-n_pred), 0): # ploting the lithology predictions obtainedby ML
      label = np.repeat(np.expand_dims(logs.iloc[:,j].values, 1), 100, 0)
      im = ax[j].imshow(label, interpolation='none', aspect='auto', cmap=cmap_facies, vmin=0, vmax=12)
      ax[j].set_xlabel(str(logs.columns[j]))

    divider = make_axes_locatable(ax[-1]) # appending lithology legend
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar=plt.colorbar(im, cax=cax)
    cbar.set_label((12*' ').join(['SS', 'S-S', 'SH', 'MR', 'DOL','LIM', 'CH','HAL', 'AN', 'TF', 'CO', 'BS']))
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
        
    f.suptitle('WELL LOGS '+str(wells[well_num]), fontsize=14,y=0.94)