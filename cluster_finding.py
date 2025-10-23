import numpy as np
import torch
import torch.nn.functional as F

torch.set_default_tensor_type(torch.cuda.FloatTensor
                              if torch.cuda.is_available()
                              else torch.FloatTensor)


def find(vertex, parent):
    if parent[vertex] != vertex:
        parent[vertex] = find(parent[vertex], parent)
    return parent[vertex]


def union(u, v, parent):
    root_u = find(u, parent)
    root_v = find(v, parent)
    if root_u != root_v:
        parent[root_u] = root_v


def find_cluster_GPU(lattice, edges, coordinate):
    # lattice: (batch, n, length), bool
    # edges: (n_edge, 2), int
    batch, n, length = lattice.shape
    if length < 2:
        return lattice, torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)
    label = torch.zeros_like(lattice, dtype=torch.int64)
    label = label.reshape(-1)
    n_label = lattice.sum()
    label[lattice.reshape(-1) > 0] = torch.arange(1, n_label + 1, dtype=torch.int64)
    label = label.reshape(batch, n, length)

    batch_idx = torch.arange(batch).reshape(batch, 1)
    equivalence = []
    # for i in range(length):
    #     equivalence.append(label[:, :, i][batch_idx.reshape(batch, 1, 1), edges].reshape(-1, 2))  # (batch*n_edge, 2)
    for i in range(length - 1):
        equivalence.append(torch.stack([label[:, :, i], label[:, :, i + 1]], dim=2).reshape(-1, 2))  # (batch*n, 2)
        equivalence.append(torch.stack([label[:, :, i][batch_idx, edges[:, 0]],
                                        label[:, :, i + 1][batch_idx, edges[:, 1]]], dim=2).reshape(-1, 2))  # (batch*n_edge, 2)
        equivalence.append(torch.stack([label[:, :, i][batch_idx, edges[:, 1]],
                                        label[:, :, i + 1][batch_idx, edges[:, 0]]], dim=2).reshape(-1, 2))  # (batch*n_edge, 2)
    equivalence = torch.cat(equivalence, dim=0)

    nonzero_mask = (equivalence > 0).all(dim=1)
    equivalence = equivalence[nonzero_mask]
    equivalence = torch.unique(equivalence, dim=0)
    u = equivalence[:, 0]
    v = equivalence[:, 1]

    parent = torch.arange(n_label + 1, dtype=torch.int64)
    parent[equivalence[:, 1]] = parent[equivalence[:, 0]]
    # i = 0
    while True:
        new_parent = parent.clone()
        new_parent.scatter_reduce_(0, u, parent[v], reduce='amin')
        new_parent.scatter_reduce_(0, v, parent[u], reduce='amin')
        n_updated = (new_parent != parent).sum()
        if n_updated == 0:
            break
        parent = torch.minimum(parent, new_parent)

        # print(f'Iteration {i}, updated {n_updated}/{len(parent)} labels')
        # i += 1

    unique_labels, inv_idx = torch.unique(parent, return_inverse=True)

    if unique_labels.numel() <= 1:
        return label, torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)
    else:
        relabeled = torch.arange(len(unique_labels), dtype=torch.int64)
        value_map = relabeled[inv_idx]

        # relabel the lattice
        label = value_map[label].to(torch.int64)

        # Optional: each site can only be counted once within each cluster, remove the duplicates
        new_label, indices = label.sort(dim=2)
        new_label[:, :, 1:] *= (torch.diff(new_label, dim=2) != 0).to(torch.int64)
        indices = indices.sort(dim=2)[1]
        label = torch.gather(new_label, 2, indices)

        # Compute the duration of each avalanche
        min_times = torch.zeros(len(unique_labels), dtype=torch.int64)
        max_times = torch.zeros(len(unique_labels), dtype=torch.int64)
        time_indices = torch.arange(length).reshape(1, 1, length).expand(label.shape).reshape(-1)
        min_times.scatter_reduce_(0, label.reshape(-1), time_indices, reduce='amin', include_self=False)
        max_times.scatter_reduce_(0, label.reshape(-1), time_indices, reduce='amax', include_self=False)
        duration = max_times - min_times + 1
        duration = duration[1:]

        weight = lattice.reshape(-1)
        cluster_sizes = torch.zeros(len(unique_labels), dtype=torch.float)
        cluster_sizes.scatter_add_(0, label.reshape(-1), weight.to(torch.float))
        cluster_sizes = cluster_sizes[1:]

        if coordinate is not None:
            min_x = torch.zeros(len(unique_labels), dtype=torch.float)
            max_x = torch.zeros(len(unique_labels), dtype=torch.float)
            min_x.scatter_reduce_(0, label.reshape(-1), coordinate[:, 0].reshape(n, 1).expand_as(label).reshape(-1),
                                  reduce='amin', include_self=False)
            max_x.scatter_reduce_(0, label.reshape(-1), coordinate[:, 0].reshape(n, 1).expand_as(label).reshape(-1),
                                  reduce='amax', include_self=False)
            min_x = min_x[1:]
            max_x = max_x[1:]
            is_percolating = (min_x == 0) & (max_x == coordinate[:, 0].max())

            n, D = coordinate.shape
            Rs = torch.zeros(len(unique_labels) - 1, dtype=torch.float)
            for di in range(D):
                moment_1 = torch.zeros(len(unique_labels), dtype=torch.float)
                moment_2 = torch.zeros(len(unique_labels), dtype=torch.float)
                moment_1.scatter_add_(0, label.reshape(-1), coordinate[:, di].unsqueeze(1).expand_as(label).reshape(-1))
                moment_2.scatter_add_(0, label.reshape(-1), (coordinate[:, di] ** 2).unsqueeze(1).expand_as(label).reshape(-1))
                moment_1 = moment_1[1:] / cluster_sizes
                moment_2 = moment_2[1:] / cluster_sizes
                Rs_i = moment_2 - moment_1 ** 2
                Rs += Rs_i
            Rs = Rs.sqrt()

            # Float32 does not have enough precision
            time_idx = torch.arange(length).unsqueeze(0).expand_as(label).reshape(-1).double()
            moment_1 = torch.zeros(len(unique_labels), dtype=torch.double)
            moment_2 = torch.zeros(len(unique_labels), dtype=torch.double)
            moment_1.scatter_add_(0, label.reshape(-1), time_idx)
            moment_2.scatter_add_(0, label.reshape(-1), time_idx ** 2)
            moment_1 = moment_1[1:] / cluster_sizes
            moment_2 = moment_2[1:] / cluster_sizes
            Rs_t = (moment_2 - moment_1 ** 2).sqrt().float()
        else:
            is_percolating = torch.zeros(len(unique_labels) - 1, dtype=torch.bool)
            Rs = torch.zeros(len(unique_labels) - 1, dtype=torch.float)
            Rs_t = torch.zeros(len(unique_labels) - 1, dtype=torch.float)

        return label, cluster_sizes, duration, Rs, Rs_t, is_percolating


def find_cluster_graph(lattice, edges, coordinate):
    # lattice: (n, length)
    n, length = lattice.shape
    if length < 2:
        return lattice, torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)
    label = torch.zeros_like(lattice, dtype=torch.int)
    label = label.reshape(-1)
    label[lattice.reshape(-1) > 0] = torch.arange(1, (lattice.reshape(-1) > 0).sum() + 1, dtype=torch.int)
    label = label.reshape(n, length)

    equivalence = []
    # for i in range(length):
    #     equivalence.append(label[:, i][edges])
    for i in range(length - 1):
        equivalence.append(torch.stack([label[:, i], label[:, i + 1]], dim=1))
        equivalence.append(torch.stack([label[:, i][edges[:, 0]], label[:, i + 1][edges[:, 1]]], dim=1))
        equivalence.append(torch.stack([label[:, i][edges[:, 1]], label[:, i + 1][edges[:, 0]]], dim=1))
    equivalence = torch.cat(equivalence, dim=0)

    nonzero_mask = (equivalence > 0).all(dim=1)
    equivalence = equivalence[nonzero_mask]
    equivalence = torch.unique(equivalence, dim=0)

    # find connected components of the equivalence graph
    graph_edges = equivalence.cpu().numpy()
    nodes = np.arange(1, lattice.sum().cpu().numpy().item() + 1)
    parent = {key: value for key, value in zip(nodes, nodes)}

    for edge in graph_edges:
        union(edge[0], edge[1], parent)

    value_map = torch.tensor([find(node, parent) for node in nodes])
    unique_labels = torch.unique(value_map)
    if unique_labels.numel() == 0:
        return label, torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)
    else:
        relabeled = torch.arange(1, len(unique_labels) + 1, dtype=torch.int)
        relabel_map = torch.zeros(unique_labels.max() + 1, dtype=torch.int)
        relabel_map[unique_labels] = relabeled
        value_map = relabel_map[value_map]

        value_map = torch.cat([torch.zeros(1, dtype=torch.int), value_map], dim=0)

        # relabel the lattice
        label = value_map[label].to(torch.int64)

        # Optional: each site can only be counted once within each cluster, remove the duplicates
        new_label, indices = label.sort(dim=1)
        new_label[:, 1:] *= (torch.diff(new_label, dim=1) != 0).to(torch.int64)
        indices = indices.sort(dim=1)[1]
        label = torch.gather(new_label, 1, indices)
        index = label.reshape(-1).to(torch.int64)

        # Compute the duration of each avalanche
        min_times = torch.zeros(len(unique_labels) + 1, dtype=torch.int64)
        max_times = torch.zeros(len(unique_labels) + 1, dtype=torch.int64)
        time_indices = torch.arange(label.shape[1]).unsqueeze(0).expand(label.shape).reshape(-1)
        min_times.scatter_reduce_(0, label.reshape(-1), time_indices, reduce='amin', include_self=False)
        max_times.scatter_reduce_(0, label.reshape(-1), time_indices, reduce='amax', include_self=False)
        duration = max_times - min_times + 1
        duration = duration[1:]

        weight = lattice.reshape(-1)
        cluster_sizes = torch.zeros(value_map.max() + 1, dtype=torch.float)
        cluster_sizes.scatter_add_(0, index, weight.to(torch.float))
        cluster_sizes = cluster_sizes[1:]

        min_x = torch.zeros(len(unique_labels) + 1, dtype=torch.float)
        max_x = torch.zeros(len(unique_labels) + 1, dtype=torch.float)
        min_x.scatter_reduce_(0, label.reshape(-1), coordinate[:, 0].reshape(n, 1).expand_as(label).reshape(-1),
                              reduce='amin', include_self=False)
        max_x.scatter_reduce_(0, label.reshape(-1), coordinate[:, 0].reshape(n, 1).expand_as(label).reshape(-1),
                              reduce='amax', include_self=False)
        min_x = min_x[1:]
        max_x = max_x[1:]
        is_percolating = (min_x == 0) & (max_x == coordinate[:, 0].max())

        n, D = coordinate.shape
        Rs = torch.zeros(len(unique_labels), dtype=torch.float)
        for di in range(D):
            moment_1 = torch.zeros(len(unique_labels) + 1, dtype=torch.float)
            moment_2 = torch.zeros(len(unique_labels) + 1, dtype=torch.float)
            moment_1.scatter_add_(0, label.reshape(-1), coordinate[:, di].unsqueeze(1).expand_as(label).reshape(-1))
            moment_2.scatter_add_(0, label.reshape(-1), (coordinate[:, di] ** 2).unsqueeze(1).expand_as(label).reshape(-1))
            moment_1 = moment_1[1:] / cluster_sizes
            moment_2 = moment_2[1:] / cluster_sizes
            Rs_i = moment_2 - moment_1 ** 2
            Rs += Rs_i
        Rs = Rs.sqrt()

        # Float32 does not have enough precision
        time_idx = torch.arange(length).unsqueeze(0).expand_as(label).reshape(-1).double()
        moment_1 = torch.zeros(len(unique_labels) + 1, dtype=torch.double)
        moment_2 = torch.zeros(len(unique_labels) + 1, dtype=torch.double)
        moment_1.scatter_add_(0, label.reshape(-1), time_idx)
        moment_2.scatter_add_(0, label.reshape(-1), time_idx ** 2)
        moment_1 = moment_1[1:] / cluster_sizes
        moment_2 = moment_2[1:] / cluster_sizes
        Rs_t = (moment_2 - moment_1 ** 2).sqrt().float()

        return label, cluster_sizes, duration, Rs, Rs_t, is_percolating

