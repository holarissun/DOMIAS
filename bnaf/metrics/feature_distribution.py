
# Import necessary packages
import numpy as np

def feature_distribution (orig_data, synth_data):
    """Compare feature distribution between orig data and synth data
    
    Args:
        orig_data: original data
        synth_data: synthetically generated data
        
    Returns:
        dist_comp_table: distribution comparison table
    """
    
    orig_data = np.asarray(orig_data)
    
    # Parameters
    no, dim = np.shape(orig_data)
        
    # Output initialization
    dist_comp_table = np.zeros([dim, 4])
        
    for i in range(dim):
                        
        if len(np.unique(orig_data[:, i])) > 2:
            dist_comp_table[i,0] = np.mean(synth_data[:,i])
            dist_comp_table[i,1] = np.std(synth_data[:,i])
                            
            dist_comp_table[i,2] = np.mean(orig_data[:,i])
            dist_comp_table[i,3] = np.std(orig_data[:,i])
            
        else:
            dist_comp_table[i,0] = np.sum(synth_data[:,i]==1)
            dist_comp_table[i,1] = np.sum(synth_data[:,i]==1) / float(no)
                        
            dist_comp_table[i,2] = np.sum(orig_data[:,i]==1)
            dist_comp_table[i,3] = np.sum(orig_data[:,i]==1) / float(no)
                        
    return dist_comp_table
