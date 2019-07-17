from utils import *
import numpy as np
from models import MDA, kl_divergence
import torch
from scipy import io as sio
from sklearn.preprocessing import minmax_scale
from tqdm import trange
import pandas as pd
from torch.utils.data import DataLoader
from argument_parser import argument_parser
import warnings
warnings.filterwarnings("ignore")


def main():

    args = argument_parser()
    table_printer(args)
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

    num_nodes = pd.read_csv(args.data_folder + args.dataset + "/" +
                            args.dataset+"_string_genes.txt", header=None).shape[0]

    Nets = []
    F = []
    for net in range(len(args.network_types)):
        print("Loading network for ", args.network_types[net])
        N = sio.loadmat(args.data_folder + args.dataset + "/" + args.annotations_path + args.dataset + '_net_' +
                        str(net+1) + '_K3_alpha0.98.mat')
        Net = N['Net'].todense()
        Nets.append(minmax_scale(Net))
        F.append(Net.shape[1])

    use_sparse = True
    BETA = 0.5
    tr_x_noisy, tr_x, ts_x_noisy, ts_x = split_data(Nets)
    num_networks = len(args.network_types)
    z_dim = [args.hidden_size] * num_networks
    latent_dim = args.latent_size
    model = MDA(F, z_dim, latent_dim)
    model.to(device)
    print(model)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters:", count_parameters(model)/1000000)

    train_dataset = NetworksDataset(tr_x_noisy, tr_x)
    test_dataset = NetworksDataset(ts_x_noisy, ts_x)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=ts_x_noisy[0].shape[0])
    RHO = 0.01
    rho = torch.FloatTensor([RHO for _ in range(latent_dim)]).unsqueeze(0).to(device)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=1e-4, nesterov=True)
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
            encoded, enc, dec, out = model(input_x)
            rec_loss, m_loss = criterion_for_list(criterion, output_x, out)
            if use_sparse:
                rho_hat = torch.sum(encoded, dim=0, keepdim=True)
                sparsity_penalty = BETA * kl_divergence(rho, rho_hat)
                loss = rec_loss + sparsity_penalty
            else:
                loss = rec_loss
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
                _, _, _, val_out = model(test_input_x)

                val_loss, _ = criterion_for_list(criterion, output_x, out)
                total_val_loss.append(val_loss.item())
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

    input_x = list_to_cpu(input_x)
    output_x = list_to_cpu(output_x)
    test_input_x = list_to_cpu(test_input_x)
    test_output_x = list_to_cpu(test_output_x)

    model.load_state_dict(torch.load('best_model.pkl'))
    with torch.no_grad():
        model.eval()
        features, _, _, _ = model(list_to_gpu(Nets, device))
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
