import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Args for ex2: Analysis of Orientation & ' \
                                                 'Direction Selectivity in V1 Cells',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    
    # General parameters
    parser.add_argument('--n_units', 
                        type=int, 
                        default=10, 
                        help='Number of recorded units')
    
    parser.add_argument('--n_directions', 
                        type=int, 
                        default=12,
                        help='Number of stimulus directions')
    
    parser.add_argument('--n_repetitions', 
                        type=int, 
                        default=200, 
                        help='Number of repetitions per direction')
    
    parser.add_argument('--n_bins', 
                        type=int, 
                        default=64,
                        help='Number of bins for PSTH')
    
    return parser.parse_args()