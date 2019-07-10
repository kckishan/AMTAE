from utils import *
import numpy as np
from models import MDA
import torch
from scipy import io as sio
from sklearn.preprocessing import minmax_scale
from tqdm import trange
import pandas as pd
from torch.utils.data import DataLoader
from argument_parser import argument_parser


def main():

    args = argument_parser()
    table_printer(args)
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

    num_nodes = pd.read_csv(args.data_folder + args.dataset + "/" +
                            args.dataset+"_string_genes.txt", header=None).shape[0]
    string_nets = ['neighborhood', 'fusion', 'cooccurence',
                   'coexpression', 'experimental', 'database']

    Nets = []
    F = []
    for net in range(len(string_nets)):
        print("Loading network for ", string_nets[net])
        N = sio.loadmat(args.data_folder + args.dataset + "/" + args.annotations_path + args.dataset + '_net_' +
                        str(net+1) + '_K3_alpha0.98.mat')
        Net = N['Net'].todense()
        Nets.append(minmax_scale(Net))
        F.append(Net.shape[1])

    tr_x_noisy, tr_x, ts_x_noisy, ts_x = split_data(Nets)
    num_networks = len(string_nets)
    z_dim = [args.hidden_size] * num_networks
    latent_dim = args.hidden_size
    model = MDA(F, z_dim, latent_dim)
    model.to(device)
    print(model)

    train_dataset = NetworksDataset(tr_x_noisy, tr_x)
    test_dataset = NetworksDataset(ts_x_noisy, ts_x)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=ts_x_noisy[0].shape[0])

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                weight_decay=0, nesterov=False)
    epochs = trange(args.epochs, desc="Validation Loss")
    best_loss = np.Inf
    no_improvement = 0
    early_stopping_limit = 10
    for epoch in epochs:
        model.train()
        for i, X in enumerate(trainloader):
            input_x, output_x = X
            input_x = list_to_gpu(input_x, device)
            output_x = list_to_gpu(output_x, device)
            # ---------------forward------------------
            z, out = model(input_x)
            loss = 0
            m_loss = []
            for i in range(num_networks):
                l = criterion(out[i], output_x[i])
                loss += l
                m_loss.append(l.item())
            # ---------------backward------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            total_val_loss = []
            for i, test_X in enumerate(testloader):
                test_input_x, test_output_x = test_X
                test_input_x = list_to_gpu(test_input_x, device)
                test_output_x = list_to_gpu(test_output_x, device)
                _, val_out = model(test_input_x)

                val_loss = 0
                for i in range(num_networks):
                    val_l = criterion(val_out[i], test_output_x[i])
                    val_loss += val_l.item()
                total_val_loss.append(val_loss)
            epochs.set_description("Validation Loss: %g" % round(np.mean(total_val_loss), 4))
        if val_loss < best_loss:
            no_improvement = 0
            best_loss = val_loss
            torch.save(model.state_dict(), "best_model.pkl")
        else:
            no_improvement += 1
            if no_improvement == early_stopping_limit:
                epochs.close()
                break

    # load functional labels of proteins
    GO = sio.loadmat(args.data_folder + args.dataset + "/" +
                     args.annotations_path + args.dataset + '_annotations.mat')

    model.load_state_dict(torch.load('best_model.pkl'))
    with torch.no_grad():
        model.eval()
        features, _ = model(list_to_gpu(Nets, device))
        features = features.cpu().detach().numpy()
        features = minmax_scale(features)

    for level in args.label_names:
        print("### Running for level: %s" % (level))
        perf = cross_validation(features, GO[level],
                                n_trials=10)
        avg_micro = 0.0
        for ii in range(0, len(perf['fmax'])):
            print('%0.5f %0.5f %0.5f %0.5f\n' %
                  (perf['pr_micro'][ii], perf['pr_macro'][ii], perf['fmax'][ii], perf['acc'][ii]))
            avg_micro += perf['pr_micro'][ii]
        avg_micro /= len(perf['fmax'])
        print("### Average (over trials): m-AUPR = %0.3f" % (avg_micro))


if __name__ == "__main__":
    main()
