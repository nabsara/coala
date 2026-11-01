import torch


def compute_control_dim_mean_var_per_class(ck, ak, num_classes):
    sums = torch.zeros(num_classes, device=ck.device)
    counts = torch.zeros(num_classes, device=ck.device)
    ak = ak.to(torch.int64)  # scatter(): Expected dtype int64 for index
    # sum values per class
    sums.scatter_add_(0, ak, ck)

    # count occurences per class
    ones = torch.ones_like(ck)
    counts.scatter_add_(0, ak, ones)
    
    # identify missing classes and replace 0 by 1 to avoid nan 
    classes_represented = counts.nonzero().squeeze()
    m = (counts == 0)
    counts[m] = 1 # avoid nan
    # compute mean per class
    mean = sums / counts

    # compute variance per class
    # var = mean((x - mean)^2)
    diff2 = (ck - mean[ak])**2
    var = torch.zeros(num_classes, device=ck.device)
    var.scatter_add_(0, ak, diff2)
    var = var / counts

    return mean, var, classes_represented


def compute_aggregated_c_mean_var_per_class(means, vars, classes):
    n_classes = means[0].shape[0]
    
    # initialize accumulators
    mean_sum_per_class = torch.zeros(n_classes).to(means[0].device)
    var_sum_per_class = torch.zeros(n_classes).to(means[0].device)
    count_per_class = torch.zeros(n_classes).to(means[0].device)

    # vectorized accumulation using scatter_add_
    for mi, vi, ai in zip(means, vars, classes):
        mask = torch.zeros(n_classes, dtype=torch.bool).to(means[0].device)
        mask[ai] = True
        mean_sum_per_class.scatter_add_(0, ai, mi[ai])
        var_sum_per_class.scatter_add_(0, ai, vi[ai])
        count_per_class.scatter_add_(0, ai, torch.ones_like(ai, dtype=torch.float))

    # compute means (avoid division by zero)
    mean = mean_sum_per_class / count_per_class.clamp(min=1)
    var = var_sum_per_class / count_per_class.clamp(min=1)

    classes_available = torch.unique(torch.cat(classes))

    return mean, var, classes_available
