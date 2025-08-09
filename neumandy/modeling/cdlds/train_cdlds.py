
import argparse
from ast import Pass
import itertools
from os import name
from re import U
from models import CDLDSModel
import stat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from datasets import CdLDSDataGenerator
import util
import slim
import plotly.graph_objects as go
import seaborn as sns
import plotly
import plotly.express as px


class extract_tensor(nn.Module):
    def forward(self, x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        # Reshape shape (batch, hidden)
        return tensor


class TimeSeriesDataset(torch.utils.data.Dataset):
    # TODO: make more explicit
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], idx


def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)

    return torch.tensor(np.array(X)), torch.tensor(np.array(y))




def calculate_best_correlation(ground_truth, learned, num_subdyn):
    # Calculate pairwise correlations between ground truth and learned coefficients
    
    gt = ground_truth.detach().numpy()
    ld = learned.detach().numpy()
    
    gt_centered = gt - np.mean(gt, axis=0)
    ld_centered = ld - np.mean(ld, axis=0)
    
    gt_std = np.std(gt, axis=0)
    ld_std = np.std(ld, axis=0)
    
    gt_norm = gt_centered / gt_std
    ld_norm = ld_centered / ld_std
    
    corr_mat = gt_norm.T @ ld_norm / (gt.shape[0] - 1)
    
    best_corr = np.mean(np.abs(corr_mat))
    
    return corr_mat, best_corr

def main(args):

    fix_point_change = args.fix_point_change
    eigenvalue_radius = args.eigenvalue_radius
    

    if args.generate_data:

        num_true_subdyn = args.num_subdyn

        generator = CdLDSDataGenerator(
            K=num_true_subdyn, D_control=args.control_size, fix_point_change=fix_point_change, eigenvalue_radius=float(eigenvalue_radius), set_seed=num_true_subdyn)

        time_points = 1000

        # generate toy data
        timeseries = generator.generate_data(
            time_points, sigma=0.01).T
        states = generator.states_
        coefficients = generator.coefficients_
        controls = generator.U_[:args.control_size, :]
        true_dynamics = generator.A

    else:
        timeseries = np.load(args.data_path)
        coefficients = np.load(args.state_path)
        controls = np.load(args.control_path)

    # train-test split for time series
    train_size = int(len(timeseries) * 1.0)
    train, test = timeseries[:train_size], timeseries[train_size:]

    X_train = torch.tensor(train[:-1]) 
    y_train = torch.tensor(train[1:]) 

    X_train_idx = np.arange(len(train)-1)


    hidden_size = 4
    input_size = train.shape[1]

    model = CDLDSModel(input_size=input_size, hidden_size=hidden_size,
                       output_size=input_size, time_points=len(timeseries), num_subdyn=args.num_subdyn, control_size=args.control_size).float()
    
    if args.generate_data and not args.randomize:

        with torch.no_grad():
        # initialize model with true coefficients
            model.coeffs = torch.nn.Parameter(
                torch.tensor(coefficients, requires_grad=True, dtype=torch.float32))
            if args.control_size > 0:
                model.U = torch.nn.Parameter(torch.tensor(
                controls, requires_grad=True, dtype=torch.float32))
            
            # initialize dynamics
            for f_i, A in zip(model.F, true_dynamics):
                #f_i.weight = torch.nn.Parameter(torch.tensor(A).float())
                f_i.weight.copy_(torch.tensor(A).float())
            
            #for f_i, A in zip(model.F, true_dynamics):
                # + torch.randn_like(f_i.weight) * args.sigma)
                #f_i.weight = torch.nn.Parameter(torch.tensor(A).float())
            #    f_i.weight.copy_(torch.tensor(A).float())
            
            
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    # loader = TimeSeriesDataset(data.TensorDataset(X_train.float(), y_train.float()), shuffle=True, batch_size=8)
    data = TimeSeriesDataset(list(zip(X_train.float(), y_train.float())))

    # create an iterable over our data, no shuffling because we want to keep the temporal information
    loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, shuffle=True, drop_last=True)

    n_epochs = args.epochs


    single_reconstruction_loss_history = []
    coeff_sparsity_loss_history = []
    control_sparsity_loss_history = []
    smooth_reg_loss_history = []
    loss_history = []

    for epoch in range(n_epochs):
        model.train()
        for (X_batch, y_batch), idx in loader:

            y_pred = model(X_batch.float(), X_train_idx[idx])


            coeff_delta = model.coeffs[:,1:] - model.coeffs[:, :-1]
            
            smooth_reg = torch.norm(coeff_delta, p=2)
            smooth_reg_loss = args.smooth * smooth_reg * input_size
            smooth_reg_loss_history.append(smooth_reg_loss.item())

            coeff_sparsity_loss = args.reg * model.coeff_sparsity_loss()
            coeff_sparsity_loss_history.append(coeff_sparsity_loss.item())

            rec_loss = loss_fn(y_pred, y_batch)
            single_reconstruction_loss_history.append(rec_loss.item())

            control_sparsity = model.control_sparsity_loss()
            control_sparsity_loss = args.control_sparsity_reg * \
                control_sparsity
            control_sparsity_loss_history.append(control_sparsity_loss.item())

            loss = smooth_reg_loss + rec_loss + coeff_sparsity_loss + control_sparsity_loss
            loss_history.append(loss.item())
            
            
           
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 7)
            optimizer.step()

        # Validation
        if epoch % 10 != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train.float(), X_train_idx)
            train_rmse = np.sqrt(loss_fn(y_pred, y_train))
            
        print("Epoch %d: train RMSE %.4f" %
              (epoch, train_rmse))

    print("Training finished")
    print('Storing visualizations..')
    

    
    # set plotly theme
    custom_template = go.layout.Template(
    layout=dict(
        xaxis=dict(
            showline=True,          # Show the axis line
            linecolor="grey",      # Dark axis color
            linewidth=2,            # Thick axis line
            showgrid=True,         # Hide the grid
            zeroline=False
        ),
        yaxis=dict(
            showline=True,          # Show the axis line
            linecolor="grey",      # Dark axis color
            linewidth=2,            # Thick axis line
            showgrid=True,         # Hide the grid
            zeroline=False
        ),
        plot_bgcolor='white',  # Optional: Set background color to white for contrast
        colorway=px.colors.qualitative.Plotly  # Set the color sequence
        
    )
)
    
    
    
    plotly.io.templates['custom_template'] = custom_template
    plotly.io.templates.default = 'custom_template'


        
    corr_mat, _ = calculate_best_correlation(
        torch.tensor(coefficients).T, model.coeffs.T, args.num_subdyn)
    
    # visualize correlation matrix
    fig = px.imshow(corr_mat, labels=dict(x="learned coefficients", y="ground truth coefficients", color="correlation"))
    fig.update_layout(title='Correlation Matrix of Learned and Ground Truth Coefficients')
    fig.update_xaxes(tickvals=np.arange(args.num_subdyn), ticktext=[f'state {i+1}' for i in range(args.num_subdyn)])
    fig.update_yaxes(tickvals=np.arange(args.num_subdyn), ticktext=[f'state {i+1}' for i in range(args.num_subdyn)])
    plotly.io.write_image(fig, 'correlation_matrix.svg', width=800, height=800)
    print(f'Correlation between learned and ground truth coefficients: {np.mean(np.abs(corr_mat))}')
    
    
    if args.control_size > 0:
        _, control_best_corr = calculate_best_correlation(
            torch.tensor(controls).T, model.effective_U.T, args.control_size)
        control_loss = torch.square(1-torch.tensor(control_best_corr))
        print(f'Control loss: {control_loss.item()}')

            


    model_coeffs = pd.DataFrame(model.coeffs.detach().numpy()[
        :, :train_size].T)
    
    # change column names to match the true coefficients
    model_coeffs.columns = [f'state {i+1}' for i in range(model_coeffs.shape[1])]
    
    if not args.generate_data:
        model_coeffs['time'] = model_coeffs.index/3.26
        model_coeffs.index = model_coeffs.loc[:,'time']
        model_coeffs.drop(columns=['time'], inplace=True)
    if args.generate_data:
        fig = util.plotting(model_coeffs, title='coefficients', plot_states=args.plot_states, states=states)
    else:
        fig = util.plotting(model_coeffs, title='coefficients', plot_states=args.plot_states)
    df_len = len(model_coeffs)
    time = np.arange(0, df_len, 1) 
    # divide by 3.26 to get time in seconds
    time = time/3.26
    fig.update_layout(xaxis = dict(ticktext=time))
    fig.update_xaxes(showline=True, linewidth=2, linecolor='grey', title='time')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='grey', title='magnitude')
    # update legend names
        
        
    # TODO plot coefficients
    plotly.io.write_image(fig, 'coeffs.svg', width=1600, height=400)
    

    # time_series, _ = create_dataset(timeseries, lookback=lookback)
    # multi-step reconstruction by always using the last prediction as input
    ts = torch.tensor(timeseries)
    recon = torch.zeros_like(ts)
    recon[0] = ts[0]

    recon_single = torch.zeros_like(recon)
    recon_single[0] = ts[0]

    with torch.no_grad():
        for i in range(1, len(timeseries)):
            y_pred = model(recon[i-1, :].float().unsqueeze(0),
                           i-1)
            recon[i] = y_pred[-1]

            y_pred_single = model(ts[i-1].float().unsqueeze(0),
                                  i-1)
            recon_single[i] = y_pred_single[-1]

    fig = util.plotting(recon.detach().numpy(), title='multi-step reconstruction',
                        stack_plots=False)
    # TODO plot multi-step reconstruction

    timeseries_df = pd.DataFrame(timeseries)
    if not args.generate_data:
        timeseries_df['time'] = timeseries_df.index/3.26
        timeseries_df.index = timeseries_df.loc[:,'time']
        timeseries_df.drop(columns=['time'], inplace=True)

    recon_df = pd.DataFrame(recon_single.detach().numpy())
    if not args.generate_data:
        recon_df['time'] = recon_df.index/3.26
        recon_df.index = recon_df.loc[:,'time']
        recon_df.drop(columns=['time'], inplace=True)
    # result = model(time_series.float(), torch.arange(len(timeseries)-lookback))
    fig = util.plotting([timeseries_df, recon_df
                         ], title='single-step + ground truth reconstruction', stack_plots=True, plot_states=args.plot_states, states=states)
    
    # df_len = len(timeseries)
    # time = np.arange(0, df_len, 1) 
    #divide by 3.26 to get time in seconds
    # time = time/3.26
    # fig.update_layout(xaxis = dict(ticktext=time))
    fig.update_xaxes(showline=True, linewidth=2, linecolor='grey', title='time')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='grey', title='amplitude')

    plotly.io.write_image(fig, 'reconstruction.svg', width=1600, height=400)

    if args.control_size > 0:
        ctrl = pd.DataFrame(model.effective_U.detach().numpy()[
            :, :train_size].T)
        if not args.generate_data:
            ctrl['time'] = ctrl.index/3.26
            ctrl.index = ctrl.loc[:,'time']
            ctrl.drop(columns=['time'], inplace=True)
        # control plot
        fig = util.plotting(ctrl, title='Control Signals')
        fig.update_traces(name='learned control signals')
        
        if args.generate_data:
            fig.add_trace(go.Scatter(
            x=np.arange(train_size),
            y=controls[0],
            mode='lines',
            line=dict(dash='dash',color='black', width=1),
            name='true control signals'
        ))


        plotly.io.write_image(fig, 'control_matrix.svg', width=1600, height=400)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/data.npy')
    parser.add_argument('--state_path', type=str, default='data/states.npy')
    parser.add_argument('--control_path', type=str, default='data/worm_controls.npy')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_subdyn', type=int, default=2)
    parser.add_argument('--reg', type=float, default=0.001)
    parser.add_argument('--control_sparsity_reg', type=float, default=0.0001)
    parser.add_argument('--smooth', type=float, default=0.001)
    parser.add_argument('--control_size', type=int, default=1)
    parser.add_argument('--fix_point_change', type=bool, default=True),
    parser.add_argument('--eigenvalue_radius', type=float, default=0.94),
    parser.add_argument('--sigma', type=float, default=0.01)
    parser.add_argument('--plot_states', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--generate_data', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--no-randomize', action=argparse.BooleanOptionalAction, default=False, dest='randomize',
                        help='If set, the model will not be initialized with random coefficients. Instead, it will use the true coefficients from the data generator. This is useful for debugging and testing purposes.')
    args = parser.parse_args()
    print(args)
    main(args)
