import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata
import squidpy as sq
import random
from tqdm import tqdm
from typing import Tuple, List, Dict

def simulate_cells(
    height:int, 
    width: int, 
    ncells: int, 
    cell_r_range: Tuple[int, int],
    n_cell_types: int,
) -> List[Dict[str, int]]:
    '''simulate cell locations for a given canvas

    Input
    ------
        height: number of total spots in the height direction
        width:  number of total spots in the width direction
        ncells: number of cells to simulate. Each cell is an approximated circle area.
        cell_r_range: radius range of the cells. The actual radius will be uniformly sampled from this range.
        n_cell_types: number of cell types to simulate. The actual cell type will be unifomrly sampled.

    Output
    ------
        cells: list of cells generated. Each cell is dictionary containing following keys:
            x: cell centroid x location
            y: cell centroid y location
            r: cell radius
            ct: cell type
    '''
    cell_x = np.random.randint(width-1, size=ncells)
    cell_y = np.random.randint(height-1, size=ncells)
    cell_r = np.random.randint(cell_r_range[0], cell_r_range[1], size=ncells)
    cell_ct = np.random.randint(n_cell_types, size=ncells)
    cells = []
    for i in tqdm(range(ncells), total=ncells, desc='generating cells'):
        cells.append({'x':cell_x[i], 'y':cell_y[i], 'r':cell_r[i], 'ct':cell_ct[i]})
    return cells

def simulate_spot_bin_cells(height, width, cells, non_cell_ct):
    '''Simulator: input size in number of spot; define cell center and radius, output simulated spatial locations for arbitrary true cells.
    Return
    ----------
    x,y. location of spot on x or y
    x_grid, y_grid. grid location of spot on x and y 
    binxs, binys. bin boundary of 4 by 4 spot
    mask1, mask2. mask of cell1 and cell2 in x_grid and y_grid.
    Example
    ----------
    x,y,x_grid, y_grid, binxs, binys, mask1, mask2 = simulate_spot_bin_2cell( 
        side_length = 15, num_points = 16, 
        center1 = (4.5, 4.5) ,
        radius1 = 3,
        center2 = (9, 8) ,
        radius2 =4,
        )
    '''
    ########## 2um spots ##########
    x = np.linspace(0, width-1 , width)
    y = np.linspace(0, height-1 , height)
    x_grid, y_grid = np.meshgrid(x, y)
    ######## cell mask ##########
    
    distances = np.zeros([len(cells), height, width])
    for i, cell in tqdm(enumerate(cells), total=len(cells), desc='calculating distance'):
        cur_distance = np.sqrt((x_grid - cell['x'])**2 + (y_grid - cell['y'])**2)
        cur_distance[cur_distance>cell['r']] = np.inf
        distances[i] = cur_distance
    
    mask = distances.argmin(0)
    for i in tqdm(range(width), total=width, desc='calculating cellid mask'):
        for j in range(height):
            if set(distances[:, j, i]) == {np.inf}:
                mask[j,i] = -1

    cell_type_mask = np.ones(mask.shape) * non_cell_ct
    for i in tqdm(range(width), total=width, desc='calculating celltype_id mask'):
        for j in range(height):
            if mask[j,i] != -1:
                cell_type_mask[j,i] = cells[mask[j,i]]['ct']
    
    return x, y, x_grid, y_grid, distances, mask, cell_type_mask

def simulate_spot_expr_cells_direct(x_grid, y_grid, ct_mask, cell_type_by_gene_matrix,):
    ''' Simulate counts based on inputs. 
    Inputs:
        x_grid, y_grid: defines input field. It matches ct_mask.
        ct_mask: illustrating which spot from input field belongs to which cell type
        cell_type_by_gene_matrix: illustrate the relative expression level for each gene in each cell type.
    Outputs:
        adata: An adata object from simulation
    '''

    n_spot = x_grid.shape[0] * x_grid.shape[1]
    n_celltype = cell_type_by_gene_matrix.shape[0]
    n_gene = cell_type_by_gene_matrix.shape[1]

    spotid_list = np.arange(n_spot)
    celltype_list = np.arange(n_celltype)
    gene_list = np.arange(n_gene)

    iscelltype_k = dict()
    for i in celltype_list:
        iscelltype_k[i] = spotid_list[(ct_mask==i).flatten()]

    expr = np.zeros([n_spot, n_gene])
    for ct in tqdm(celltype_list, total=n_celltype, desc='generate cell counts'):
        for gene in gene_list:
            cur_spot_idx = iscelltype_k[ct]
            expr[cur_spot_idx, gene] = np.random.poisson(lam = cell_type_by_gene_matrix[ct, gene], size=len(cur_spot_idx))

    return expr.reshape(x_grid.shape[0], x_grid.shape[1], -1)

def simulate_spot_expr_cells(x_grid, y_grid, ct_mask, cell_type_by_gene_matrix,):
    expr = simulate_spot_expr_cells_direct(x_grid, y_grid, ct_mask, cell_type_by_gene_matrix,)
    expr = expr.reshape(x_grid.shape[0] * x_grid.shape[1], -1)

    # construct adata
    obs = pd.DataFrame( {'spotid': spotid_list},index=['barcode_'+str(t) for t in spotid_list])
    obs['celltype'] = [f'ct_{int(ct)}' for ct in ct_mask.flatten()]
    obs['x'] = x_grid.flatten()
    obs['y'] = y_grid.flatten()
    var = pd.DataFrame(index=[f'gene_{i}' for i in gene_list])
    obsm = {'spatial': obs[['x', 'y']].values}
    expr = pd.DataFrame(expr, index=obs.index, columns=var.index)
    adata= anndata.AnnData(expr, obs=obs,var=var, obsm = obsm)
    return adata

def split_list_into_bins(elements, x):
    """split elements into x bins"""
    res = []
    for i in range(x):
        res.append([])

    k=0
    for val in elements:
        res[k].append(val)
        k+=1
        if k==x:
            k=0
    
    return res

from STHD import refscrna
def create_matrix(filtered=None):
    genemeanpd_filtered = refscrna.load_scrna_ref('/hpc/group/yizhanglab/yz812/STHD/testdata/crc_average_expr_genenorm_lambda_98ct_4618gs.txt')
    if filtered is not None:
        genemeanpd_filtered = genemeanpd_filtered.loc[filtered]
    cell_type_by_gene_matrix = genemeanpd_filtered.values.T.copy() + 0.000001
    n_celltypes, n_gene = cell_type_by_gene_matrix.shape
    # dmin = 1
    # dmax = 60
    # d_simulated = np.random.randint(dmin, dmax, size=n_celltypes)
    d_mean=  47
    d_simulated = np.random.exponential(d_mean, size=n_celltypes)
    for i in range(n_celltypes):
        cell_type_by_gene_matrix[i] = cell_type_by_gene_matrix[i]*max(1, d_simulated[i])
    cell_type_by_gene_matrix_normalized = (cell_type_by_gene_matrix.T / cell_type_by_gene_matrix.sum(axis=1)).T
    cell_type_by_gene_matrix_df = pd.DataFrame(cell_type_by_gene_matrix_normalized)
    cell_type_by_gene_matrix_df.columns = list('gene_'+cell_type_by_gene_matrix_df.columns.astype(str).values)
    cell_type_by_gene_matrix_df.index = list('ct_'+cell_type_by_gene_matrix_df.index.astype(str).values)

    return n_celltypes, n_gene, cell_type_by_gene_matrix_normalized

def create_ct(width, height, ncells, n_celltypes, matrix, cell_r_range=(20,30)):
    cells = simulate_cells(
        height, 
        width, 
        ncells, 
        cell_r_range, 
        n_celltypes-1 # reserving the last one to fill noncell regions
    )

    x, y, x_grid, y_grid, distances, mask, ct_mask = simulate_spot_bin_cells(
        height, 
        width, 
        cells,
        n_celltypes-1 # this is thie cell type for noncell regions
    )

    data = simulate_spot_expr_cells_direct(
        x_grid, 
        y_grid, 
        ct_mask,
        matrix,
    )

    return data, ct_mask

def diffuse_counts(data, distance, p, sigma, random_state=None):
    """
    Diffuse counts from each cell to its neighbors based on a two-dimensional Gaussian kernel.
    
    Each count in a cell/channel diffuses with probability p. If a count diffuses, it is subtracted 
    from the source cell and added to a destination cell (same channel) chosen randomly with probability 
    proportional to exp(-distance^2/(2*sigma^2)), where 'distance' is given by the corresponding row in 
    the distance matrix. Self-diffusion (i.e. the count migrating to the same cell) is excluded.
    
    Parameters:
    -----------
    data : np.ndarray, shape (n, m)
        A matrix where each row corresponds to a cell and each column corresponds to a channel.
        The entries are the counts/measurements.
        
    distance : np.ndarray, shape (n, n)
        A symmetric matrix with pairwise distances between cells. distance[i, j] gives the 
        distance between cell i and cell j.
        
    p : float
        The probability that a given count diffuses to a neighboring cell.
        
    sigma : float
        The standard deviation of the Gaussian distribution used to weight distances.
        
    random_state : int or np.random.Generator, optional
        For reproducibility, you can pass a random seed (int) or a numpy Generator.
    
    Returns:
    --------
    new_data : np.ndarray, shape (n, m)
        The updated data matrix after the diffusion event.
    """
    # Set up the random number generator
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, (int, np.integer)):
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state
    
    n, m = data.shape
    # Arrays to keep track of counts lost (diffused out) and counts gained (diffused in)
    diffused_out = np.zeros_like(data, dtype=int)
    diffused_in = np.zeros_like(data, dtype=int)
    
    # Loop over each cell
    for i in tqdm(range(n)):
        # Compute Gaussian weights for diffusion from cell i using its distances to all cells.
        # Here, we exclude self-diffusion by setting the weight for cell i to zero.
        weights = np.exp(- (distance[i, :]**2) / (2 * sigma**2))
        weights[i] = 0
        weight_sum = np.sum(weights)
        # If there is at least one neighbor, normalize the weights to form probabilities.
        if weight_sum > 0:
            probs = weights / weight_sum
        else:
            # In the unlikely event that no neighbor has a weight, use a zero probability vector.
            probs = np.zeros_like(weights)
        
        # Loop over each channel in cell i
        for c in range(m):
            # Number of counts available in the original data for this cell/channel.
            count = data[i, c]
            if count <= 0:
                continue
            # Determine how many of these counts will diffuse (binomial sampling)
            num_diffuse = rng.binomial(count, p)
            if num_diffuse > 0:
                diffused_out[i, c] = num_diffuse
                # For each diffusing count, sample a destination cell based on the Gaussian probabilities.
                dest_indices = rng.choice(n, size=num_diffuse, replace=True, p=probs)
                # Add the diffused counts to the destination cells for the same channel.
                for j in dest_indices:
                    diffused_in[j, c] += 1
    
    # Compute the new data matrix: remove diffused counts and add those that arrived.
    new_data = data - diffused_out + diffused_in
    return new_data

def diffuse_adata(
    adata, 
    p, 
    sigma=2.0 # diffusion distance of 2-sigma is one bin (4 spots)
):
    adata_diffusion = adata.copy()
    sq.gr.spatial_neighbors(
            adata_diffusion, spatial_key="spatial", coord_type="generic", 
            n_neighs=48 # within radius of 4, exclude the spot iself
        )
    new_data = diffuse_counts(adata_diffusion.X, adata_diffusion.obsp['spatial_distances'].toarray(), p, sigma, random_state=3)
    adata_diffusion.X = new_data
    return adata_diffusion
