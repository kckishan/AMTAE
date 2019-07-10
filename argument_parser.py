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
                        help="folder that contains classes.")
    parser.add_argument("--label-names", nargs="+",
                        help="The level of Gene ontology as functional labels.")
    parser.add_argument("--network-types", nargs="+", help="The type of interaction networks.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs. Default is 200.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train-test split. Default is 42.")
    parser.add_argument("--early-stopping", type=int, default=10,
                        help="Number of early-stopping iterations. Default is 10.")
    parser.add_argument("--testing_percentage", type=int, default=.2,
                        help="Percentage of training nodes. Default is 40%.")
    parser.add_argument("--dropout", type=float, default=0, help="Dropout rate. Default is 0.5.")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                        help="Learning rate. Default is 0.001.")
    parser.add_argument("--batch-size", type=float, default=64,
                        help="Batch size. Default is 64.")
    parser.add_argument("--hidden-size", type=int, default=256,
                        help="Dimension of latent representation. Default is 256.")
    parser.add_argument("--latent-size", type=int, default=128,
                        help="Dimension of latent representation. Default is 128.")
    parser.set_defaults(network_types=['neighborhood', 'fusion',
                                       'cooccurence', 'coexpression', 'experimental', 'database'])
    parser.set_defaults(label_names=['level1', 'level2', 'level3'])

    return parser.parse_args()
