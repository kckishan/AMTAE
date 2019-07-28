import argparse


def argument_parser():
    """
    Parses the argument to run the model.

    Returns
    -------
    model parameters: ArgumentParser Namespace
        The ArgumentParser Namespace that contains the model parameters.
    """
    parser = argparse.ArgumentParser(description="Run MultiNetwork Fusion")

    parser.add_argument("--data-folder", nargs="?", default="./data/", help="The data folder.")
    parser.add_argument("--dataset", nargs="?", default="yeast",
                        help="The name of the dataset. Default is yeast.")
    parser.add_argument("--annotations-path", nargs="?", default="annotations/",
                        help="folder that contains annotations.")
    parser.add_argument("--networks-path", nargs="?", default="networks/",
                        help="folder that contains adjacency lists.")
    parser.add_argument("--label-names", nargs="+",
                        help="The level of Gene ontology as functional labels.")
    parser.add_argument("--network-types", nargs="+", help="The type of interaction networks.")
    parser.add_argument("--attn-type", type=str, default='sparsemax',
                        help="Attention type to combine different networks")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs. Default is 20.")
    parser.add_argument("--early-stopping", type=int, default=5,
                        help="Number of early-stopping iterations. Default is 5.")
    parser.add_argument("--testing_percentage", type=int, default=.2,
                        help="Percentage of training nodes. Default is 20%.")
    parser.add_argument("--dropout", type=float, default=0, help="Dropout rate. Default is 0.5.")
    parser.add_argument("--learning-rate", type=float, default=5 * 1e-4,
                        help="Learning rate. Default is 0.01.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum for SGD. Default is 0.9.")
    parser.add_argument("--batch-size", type=float, default=32,
                        help="Batch size. Default is 32.")
    parser.add_argument("--hidden-size", type=int, default=2000,
                        help="Dimension of hidden representation. Default is 32.")
    parser.add_argument("--latent-size", type=int, default=600,
                        help="Dimension of latent representation. Default is 32.")
    parser.add_argument("--beta", type=float, default=0.3,
                        help="Weight of attention loss")
    parser.add_argument('--use-cuda', action='store_true', help="Use CPU only")
    parser.set_defaults(network_types=['neighborhood', 'fusion',
                                       'cooccurence', 'coexpression', 'experimental', 'database'])
    parser.set_defaults(label_names=['level1', 'level2', 'level3'])

    return parser.parse_args()
