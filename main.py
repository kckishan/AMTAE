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
import warnings
warnings.filterwarnings("ignore")


def main():

    args = argument_parser()
    table_printer(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print("Training in ", device)
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
    #
    path_to_string_nets = args.data_folder + args.dataset + "/" + args.networks_path
    Adjs, A = load_networks(path_to_string_nets, num_nodes, mtrx='adj')

    tr_x_noisy, tr_x, ts_x_noisy, ts_x = split_data(Nets)
    num_networks = len(args.network_types)
    z_dim = [args.hidden_size] * num_networks
    latent_dim = args.latent_size
    model = MDA(F, z_dim, latent_dim, args)
    model.to(device)
    model_name = "model_"+args.attn_type+"_"+str(args.hidden_size)+'_'+str(args.latent_size)+'.pkl'
    fout = open('./results/output_'+args.attn_type+"_"+args.dataset+'_'+str(args.hidden_size) +
                '_'+str(args.latent_size)+'_'+'.txt', 'w+')
    fout.write('### %s\n' % (args.dataset))
    fout.write('\n')
    print(model)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)/10**6
    print("Number of trainable parameters: %0.2f millions" % count_parameters(model))
    fout.write("Number of trainable parameters: %0.2f millions" % (count_parameters(model)))
    fout.write('\n')

    train_dataset = NetworksDataset(tr_x_noisy, tr_x)
    test_dataset = NetworksDataset(ts_x_noisy, ts_x)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=ts_x_noisy[0].shape[0])
    criterion = torch.nn.L1Loss()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                momentum=args.momentum, nesterov=False)
    epochs = trange(args.epochs, desc="Validation Loss")
    best_loss = np.Inf
    no_improvement = 0
    early_stopping_limit = 10
    for epoch in epochs:
        model.train()
        for i, X in enumerate(trainloader):
            input_x, output_x, ind = X
            input_x = list_to_gpu(input_x, device)
            output_x = list_to_gpu(output_x, device)
            # ---------------forward------------------
            z, enc, enc_rec, out, attn = model(input_x)
            deg = A[ind].to(device)
            # print(attn.shape, deg.shape)
            rec_loss = criterion_for_list(criterion, output_x, out)
            attn_loss = criterion(attn, deg)
            loss = rec_loss + args.beta * attn_loss
            # ---------------backward------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            total_val_loss = []
            for i, test_X in enumerate(testloader):
                test_input_x, test_output_x, test_ind = test_X
                test_input_x = list_to_gpu(test_input_x, device)
                test_output_x = list_to_gpu(test_output_x, device)
                _, _, _, val_out, test_attn = model(test_input_x)

                test_deg = A[test_ind].to(device)
                val_rec_loss = criterion_for_list(criterion, output_x, out)
                val_attn_loss = criterion(test_attn, test_deg)

                val_loss = val_rec_loss + args.beta * val_attn_loss
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

    # input_x = list_to_cpu(input_x)
    # output_x = list_to_cpu(output_x)
    # test_input_x = list_to_cpu(test_input_x)
    # test_output_x = list_to_cpu(test_output_x)

    model.load_state_dict(torch.load('best_model.pkl'))
    with torch.no_grad():
        model.eval()
        features, _, _, _, attn_weights = model(list_to_gpu(Nets, device))
        if torch.cuda.is_available():
            features = features.cpu()
        features = features.detach().numpy()
        features = minmax_scale(features)

        if torch.cuda.is_available():
            attn_weights = attn_weights.cpu()
        np.savetxt("results/attn_weights.txt", attn_weights.detach().numpy())
    for level in args.label_names:
        print("### Running for level: %s" % (level))
        perf = cross_validation(features, GO[level],
                                n_trials=10)
        avg_micro = 0.0
        fout.write('### %s trials:\n' % (level))
        fout.write('aupr[micro], aupr[macro], F_max, accuracy\n')
        for ii in range(0, len(perf['fmax'])):
            fout.write('%0.5f %0.5f %0.5f %0.5f\n' % (
                perf['pr_micro'][ii], perf['pr_macro'][ii], perf['fmax'][ii], perf['acc'][ii]))
            avg_micro += perf['pr_micro'][ii]
        fout.write('\n')
        avg_micro /= len(perf['fmax'])
        print("### Average (over trials): m-AUPR = %0.3f" % (avg_micro))


if __name__ == "__main__":
    main()
