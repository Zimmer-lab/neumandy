
import argparse
import itertools
from os import name
from re import U
from models import CDLDSModel, IdentityModel
import stat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import wandb
from datasets import CdLDSDataGenerator
from models import DLDSwithControl
import util
import slim
import plotly.graph_objects as go
from scipy.optimize import linear_sum_assignment
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

def find_highlight_regions(states):
    regions = []
    start = None
    for i in range(len(states)):
        if states[i] == 1 and start is None:
            start = i
        elif states[i] == 0 and start is not None:
            regions.append((start, i - 1))
            start = None
    # If the states array ends with a 1, capture that region as well
    if start is not None:
        regions.append((start, len(states) - 1))
    return regions


def calculate_best_correlation(ground_truth, learned, num_subdyn):
    # Calculate pairwise correlations between ground truth and learned coefficients
    corr_matrix = torch.zeros((num_subdyn, num_subdyn))

    from scipy.stats import pointbiserialr

    for i in range(num_subdyn):
        for j in range(num_subdyn):
            gt_flat = ground_truth[:, i].flatten()
            learned_flat = learned[:, j].flatten()

            r_pb, _ = pointbiserialr(
                gt_flat.detach().numpy(), learned_flat.detach().numpy())
            # Compute correlation between ground truth[i] and learned[j]
            # torch.corrcoef(torch.stack((gt_flat, learned_flat)))[0, 1]
            corr_matrix[i, j] = r_pb

    # print(corr_matrix)
    # Convert correlation matrix to negative for minimization
    cost_matrix = -corr_matrix.detach().numpy()

    # Use the Hungarian algorithm to find the best assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Get the optimal permutation
    best_permutation = col_ind

    # Compute the average correlation for the best permutation
    # average
    best_corr_sum = corr_matrix[row_ind, col_ind].mean()

    # Return the correlation matrix and the best permutation
    return corr_matrix.detach().numpy(), best_corr_sum, best_permutation


def main(args):

    fix_point_change = args.fix_point_change
    eigenvalue_radius = args.eigenvalue_radius
    

    if args.generate_data:

        num_true_subdyn = args.num_subdyn

        generator = DLDSwithControl(CdLDSDataGenerator(
            K=num_true_subdyn, D_control=args.control_size, fix_point_change=fix_point_change, eigenvalue_radius=float(eigenvalue_radius), set_seed=num_true_subdyn))

        time_points = 1000

        # generate toy data
        timeseries = generator.datasets.generate_data(
            time_points, sigma=0.01).T
        true_dynamics = generator.datasets.A
        states = generator.datasets.states_
        coefficients = generator.datasets.coefficients_
        controls = generator.datasets.U_[:args.control_size, :]
        B = generator.datasets.B[:, :args.control_size]
        bias = generator.datasets.B[:, args.control_size:]

    else:
        timeseries = np.load(args.data_path)
        # coefficients = np.load(args.state_path)
        controls = np.load(args.control_path)
        states = np.load(args.state_path)

    # train-test split for time series
    train_size = int(len(timeseries) * 1.0)
    # test_size = len(timeseries) - train_size
    train, test = timeseries[:train_size], timeseries[train_size:]

    X_train = torch.tensor(train[:-1])  # .unsqueeze(0)
    y_train = torch.tensor(train[1:])  # .unsqueeze(0)

    # X_test = torch.tensor(test[:-1])  # .unsqueeze(0)
    # y_test = torch.tensor(test[1:])  # .unsqueeze(0)

    X_train_idx = np.arange(len(train)-1)
    # X_test_idx = np.arange(len(train), len(timeseries)-1)


    #corr_states = states.copy()

    # shift states for plotting according to lookback
    # states = states[lookback:]

    wandb.login(key='a79ac9d4509caa0d5e477c939a41d790e7711171')

    if args.eigenvalue_radius < 0.999:
        project_name = f"Oscillatory_FastDecay"
    else:
        project_name = f"Oscillatory_SlowDecay"

    if args.wandb:
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"PROTOTYPE_{str(args.num_subdyn)}_State_With_Control_Worm_DS1",
            # dir=f'/state_{str(args.num_subdyn)}/fixpoint_change_{str(args.fix_point_change)}', # This is not a wandb feature yet, see issue: https://github.com/wandb/wandb/issues/6392
            # name of the run is a combination of the model name and a timestamp
            # reg{str(round(args.reg, 3))}_
            name=f"smooth{str(round(args.smooth, 3))}_fixpoint_change_{str(args.fix_point_change)}_batch_size_{str(args.batch_size)}_epochs_{str(args.epochs)}",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "num_subdyn": args.num_subdyn,
                "smooth": args.smooth,
                "control_sparsity_reg": args.control_sparsity_reg,
                "reg": args.reg,
            },
        )

    hidden_size = 4
    input_size = train.shape[1]

    model = CDLDSModel(input_size=input_size, hidden_size=hidden_size,
                       output_size=input_size, time_points=len(timeseries), num_subdyn=args.num_subdyn, control_size=args.control_size).float()
    
    if args.generate_data:
        with torch.no_grad():
            model.U = torch.nn.Parameter(torch.tensor(
                 controls, requires_grad=True, dtype=torch.float32))
             
            #model.coeffs = torch.nn.Parameter(torch.tensor(
            #    coefficients, requires_grad=True, dtype=torch.float32))
            
            #for f_i, A in zip(model.F, true_dynamics):
                # + torch.randn_like(f_i.weight) * args.sigma)
                #f_i.weight = torch.nn.Parameter(torch.tensor(A).float())
            #    f_i.weight.copy_(torch.tensor(A).float())
            
    with torch.no_grad():
            model.U = torch.nn.Parameter(torch.tensor(
                 controls, requires_grad=True, dtype=torch.float32))


    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    # loader = TimeSeriesDataset(data.TensorDataset(X_train.float(), y_train.float()), shuffle=True, batch_size=8)
    data = TimeSeriesDataset(list(zip(X_train.float(), y_train.float())))

    # create an iterable over our data, no shuffling because we want to keep the temporal information
    loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, shuffle=True, drop_last=True)

    initial_teacher_forcing_ratio = 0.5
    final_teacher_forcing_ratio = 0.0
    n_epochs = args.epochs
    for epoch in range(n_epochs):
        model.train()
        for (X_batch, y_batch), idx in loader:

            teacher_forcing_ratio = initial_teacher_forcing_ratio - \
                (initial_teacher_forcing_ratio -
                 final_teacher_forcing_ratio) * (epoch / n_epochs)

            y_pred = model(X_batch.float(), X_train_idx[idx])

            # indices of the batch
            #y_pred = dummy_model(X_batch.float(), X_train_idx[idx])


            # sparsifying control inputs with mask
            # model.sparsity_mask()
            # with torch.no_grad():
            #    model.U = torch.relu(model.U)



            # coeff_delta = model.coeffs[:, 1:] - model.coeffs[:, :-1]

            coeff_delta = model.coeffs[:,
                                                   1:] - model.coeffs[:, :-1]
            # L2 norm of the difference between consecutive coefficients
            # smooth_reg = torch.norm(coeff_delta, p=2)

            smooth_reg = torch.norm(coeff_delta, p=2)

            # smooth_reg_loss = args.smooth * smooth_reg * input_size

            smooth_reg_loss = args.smooth * smooth_reg * input_size


            coeff_sparsity_loss = args.reg * model.coeff_sparsity_loss()
            
            # coeff_sparsity_loss = args.reg * model.coeff_sparsity_loss()


            # single_loss = loss_fn(y_pred, y_batch)

            rec_loss = loss_fn(y_pred, y_batch)
            control_sparsity = model.control_sparsity_loss()
            control_sparsity_loss = args.control_sparsity_reg * \
                control_sparsity

            loss = smooth_reg_loss + rec_loss + coeff_sparsity_loss + control_sparsity_loss
            wandb.log({'single_reconstruction_loss': rec_loss.item()})
            wandb.log({'coeff_sparsity': coeff_sparsity_loss.item()})
            wandb.log({'control_sparsity': control_sparsity_loss.item()})
            wandb.log({'loss': loss.item()})
            # wandb.log({'sparsity_loss': sparsity_loss.item()})
            # wandb.log({'smooth_reg_loss': smooth_reg_loss.item()})
            wandb.log({'smooth_reg_loss': smooth_reg_loss.item()})
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
            # y_pred = dummy_model(X_test.float(), X_test_idx)
            # test_rmse = np.sqrt(loss_fn(y_pred, y_test))
        print("Epoch %d: train RMSE %.4f" %
              (epoch, train_rmse))

    print("Training finished")
    print('Storing visualizations..')


    
    # set plotly theme
    #plotly.io.templates.default = 'plotly_white'
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


    model_coeffs = pd.DataFrame(model.coeffs.detach().numpy()[
        :, :train_size].T)
    
    # change column names to match the true coefficients
    model_coeffs.columns = [f'system {i+1}' for i in range(model_coeffs.shape[1])]
    
    if not args.generate_data:
        model_coeffs['time'] = model_coeffs.index/3.26
        model_coeffs.index = model_coeffs.loc[:,'time']
        model_coeffs.drop(columns=['time'], inplace=True)
    fig = util.plotting(model_coeffs, title='coefficients', plot_states=args.plot_states, states=states)
    df_len = len(model_coeffs)
    # time = np.arange(0, df_len, 1) 
    # divide by 3.26 to get time in seconds
    # time = time/3.26
    # fig.update_layout(xaxis = dict(ticktext=time))
    fig.update_xaxes(showline=True, linewidth=2, linecolor='grey', title='time')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='grey', title='magnitude')
    wandb.log({"coeffs": fig})
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
    wandb.log({"multi-step reconstruction": fig})

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
    wandb.log({"single-step + ground truth reconstruction": fig})
    plotly.io.write_image(fig, 'reconstruction.svg', width=1600, height=400)

    # print(model.U.detach().numpy())

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


        wandb.log({"Control Matrix": fig})

    run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/')
    parser.add_argument('--data_path', type=str, default='worm_DS1_low_TE.npy')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_subdyn', type=int, default=2)
    parser.add_argument('--reg', type=float, default=0.001)
    parser.add_argument('--control_sparsity_reg', type=float, default=0.0001)
    parser.add_argument('--smooth', type=float, default=0.0001)
    parser.add_argument('--dynamics_path', type=str, default='As.npy')
    parser.add_argument('--state_path', type=str, default='states.npy')
    parser.add_argument('--control_path', type=str, default='worm_controls.npy')
    parser.add_argument('--control_size', type=int, default=1)
    parser.add_argument('--fix_point_change', type=bool, default=True),
    parser.add_argument('--eigenvalue_radius', type=float, default=0.94),
    parser.add_argument('--sigma', type=float, default=0.01)
    parser.add_argument('--loss_reg', type=float, default=0.1)
    parser.add_argument('--plot_states', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--generate_data', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--with_init', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    print(args)
    main(args)
