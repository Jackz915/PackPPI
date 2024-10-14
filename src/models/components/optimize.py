import torch
from src.models.components.clash import compute_residue_clash


def find_clash_mask(batch, SC_D, violation_tolerance_factor, clash_overlap_tolerance):
    per_residue_clash = compute_residue_clash(batch, SC_D,
                                              violation_tolerance_factor,
                                              clash_overlap_tolerance)

    mean_clash = per_residue_clash.mean()
    SC_D_clash_mask = (per_residue_clash > mean_clash).unsqueeze(-1).expand(-1, -1, 4)
        
    # median_clash = per_residue_clash.median(dim=-1)[0].unsqueeze(-1).expand(-1, per_residue_clash.shape[1])
    # SC_D_clash_mask = (per_residue_clash > median_clash).unsqueeze(-1).expand(-1, -1, 4)
    
    # top_residue_clash, _ = torch.topk(per_residue_clash, 10)
    # SC_D_clash_mask = (per_residue_clash >= top_residue_clash.squeeze()[-1]).unsqueeze(-1).expand(-1, -1, 4)
    return SC_D_clash_mask


def proximal_optimizer(batch, SC_D, 
                       violation_tolerance_factor, 
                       clash_overlap_tolerance,
                       lamda, 
                       num_steps=50):

    assert batch.num_proteins == 1

    SC_D_clash_mask = find_clash_mask(batch, SC_D, violation_tolerance_factor, clash_overlap_tolerance)
    
    z = SC_D * SC_D_clash_mask

    def optimization_function(x):
        x = x * SC_D_clash_mask
        x = torch.where(SC_D_clash_mask, x, SC_D)

        per_residue_clash = compute_residue_clash(batch, x, 
                                                  violation_tolerance_factor, 
                                                  clash_overlap_tolerance)

        sc_loss = (torch.abs(x - z) ** 2).sum(dim=(-1)).mean() 
        clash_loss = per_residue_clash.mean() 
        
        loss = sc_loss + lamda * clash_loss
        return loss

    SC_D_resample = z.clone()
    SC_D_resample.requires_grad = True

    # optimizer = torch.optim.SGD([SC_D_resample], lr=5e-2)
    optimizer = torch.optim.Adam([SC_D_resample], lr=1e-2)

    initial_loss = optimization_function(SC_D_resample).item()
    
    # List to store the results at each step
    SC_D_resample_list = []
    loss_list = []
    
    # Run optimizer for multiple steps
    for _ in range(num_steps):
        optimizer.zero_grad()
        loss = optimization_function(SC_D_resample)
        loss.backward()
        # print(loss.item())
        optimizer.step()

        # Store the current SC_D_resample
        SC_D_resample_clone = SC_D_resample.detach().clone()
        SC_D_resample_clone = torch.where(SC_D_clash_mask, SC_D_resample_clone, SC_D)
        SC_D_resample_list.append(SC_D_resample_clone)
        loss_list.append(loss.item())

    return SC_D_resample_list, loss_list
