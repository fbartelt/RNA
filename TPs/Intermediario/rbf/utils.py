import numpy as np
import plotly.graph_objects as go

def normalize_features(X, mean, std):
    Xtemp = np.copy(X)
    Xtemp = Xtemp - mean
    Xtemp = Xtemp / std
    return Xtemp

def delete_features(X, feat_idx):
    # Returns matrix X with features indexes in feat_idx ignored
    Xtemp = np.copy(X)
    Xtemp = np.delete(Xtemp, feat_idx,1)
    return Xtemp

def eval_accuracy(y_hat, y, nn_type = 0):
    # Divides quadratic error by 4 if activation function is tanh
    N = np.shape(y)[0] * (4 ** nn_type)
    return (1 - ((y-y_hat).T @ (y-y_hat)) / N).ravel()

def split_set(X, y_sample, train_prop = 0.7):
    N = X.shape[0]

    # Get indexes corresponding to each class
    idx1 = [idx for idx, val in enumerate(y_sample.flatten()) if val==1]
    idx0 = sorted(list(set(range(0,N)) - set(idx1)))
    N0,N1 = len(idx0), len(idx1)
    N_train0, N_train1 = round(train_prop*N0), round(train_prop*N1)
    # Randomize indexes
    np.random.default_rng().shuffle(idx0)
    np.random.default_rng().shuffle(idx1)

    # Select samples for training and testing
    x_train = X[np.append(idx0[0:N_train0], idx1[0:N_train1]),:]
    x_test = X[np.append(idx0[N_train0::], idx1[N_train1::]),:]
    y_train = y_sample[np.append(idx0[0:N_train0], idx1[0:N_train1]),:]
    y_test = y_sample[np.append(idx0[N_train0::], idx1[N_train1::]),:]

    return (x_train, x_test, y_train, y_test)

def plot_accuracy_std(cluster_rng, accs, stds, data_names, title):
    """
        Plots scatter plot of accuracies and standard deviations for multiple data

        Parameters:
        -----------
            cluster_rng: Tuple
                specifies cluster numbers range as (c_min, c_max)
            accs: Tuple
                tuple of accuracies tuple ((acc1_train, acc1_test), (acc2_train, acc2_test)...)
            stds: Tuple
                tuple of standard deviation tuple ((std1_train, std1_test), (std2_train, std2_test)...)
            data_names: Tuple
                tuple with plot names ('method1', 'method2')
            title: String
                text that will be showed after default - 'Mean accuracy and standard deviation <title>'

        Returns:
        --------
            fig: graph_object
                graph_object figure made with plotly
            
    """
    fig = go.Figure()
    xmin, xmax = cluster_rng

    for acc, std, name in zip(accs, stds, data_names):
        fig.add_trace(go.Scatter(x = np.arange(xmin, xmax), y=acc[0], error_y=dict(type='data', 
                                 array=std[0],visible=True, thickness=0.7), 
                                 name = '{} (train)'.format(name)))
        fig.add_trace(go.Scatter(x = np.arange(xmin, xmax), y=acc[1], error_y=dict(type='data', 
                                 array=std[1],visible=True, thickness=0.7), 
                                 name = '{} (test)'.format(name)))
    
    fig.update_layout(title = {'text':'Mean accuracies and standard deviations {}'.format(title),
                      'font_size':15}, width=800, height=500)
    fig.update_yaxes(title = 'Acurracy')
    fig.update_xaxes(title = 'k (number of clusters)')
    return fig