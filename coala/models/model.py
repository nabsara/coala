import gin
import torch
import torch.nn as nn
import pytorch_lightning as pl
from einops import rearrange
from pytorch_lightning.utilities import CombinedLoader
# from pytorch_lightning.trainer.supporters import CombinedLoader

from typing import Callable, List, Union

from coala.helpers.data_visualization import plot_class_distribution_per_control_dim


@gin.configurable
class MyDataModule(pl.LightningDataModule):

    def __init__(
            self,
            unlabel_data : Callable,
            label_data : Callable,
            val_data : Callable
        ):
        super().__init__()
        self.train_loader_1 = unlabel_data()
        self.train_loader_2 = label_data()
        self.val_loader = val_data()

    def train_dataloader(self):
        return CombinedLoader(
            {
                "unlabel": self.train_loader_1, 
                "label":  self.train_loader_2, 
            },
            mode="max_size_cycle",
        )

    def val_dataloader(self):
        return self.val_loader  


@gin.configurable
class CoALa(pl.LightningModule):

    def __init__(
        self,
        control_extractor : Callable[[], nn.Module],
        latent_projector : Callable[[], nn.Module],
        cdim : int = 1,
        attributes : List[str] = ['rms'],
        n_classes_per_factor : List[int] = [20],
        class_values : List[List[int]] = None,
        quantized : List[bool] = [True],
        bins_values: List[Union[List[None], torch.Tensor]] = [],
        rm_silent_ref : List[bool] = [False],
        lr : float = 1e-4,
        beta_kl_c : float = 1.,
        warmup_kl_c : int = 1,
        use_contrast : bool = True,
        beta_contrast : float = 1.,
        warmup_contrast : int = 1,
        temperature : float = 0.07,
        n_ex_val : int = 4,
        ref_mode : str = 'agg', # ['agg', 'agg_k', 'onset', 'all'] 
        n_max_frames : List[int] = [None],
        shuffle : bool = False,
        use_seqids : bool = False,
        rm_zero_q : bool = False, 
        eps : float = 1e-2,
        n_ref_seq : int = 1,
        version : str = 'v4',
        same_class : bool = True, 
    ):
        super().__init__()
    
        self.control_extractor = control_extractor()
        self.latent_projector = latent_projector()

        self.latent_size = self.control_extractor.latent_dim
        self.control_dim = self.control_extractor.control_dim
        self.cdim = cdim
        
        self.all_attributes = attributes
        self.n_controls = len(self.all_attributes)

        self.n_classes_per_factor = n_classes_per_factor
        self.quantized = quantized
        self.bins_values = [torch.tensor(v) for v in bins_values]

        self.rm_silent_ref = rm_silent_ref
        
        self.class_values = class_values

        # for ELBO
        self.warmup_kl_c = warmup_kl_c
        self.beta_kl_c = beta_kl_c

        # for contrastive loss
        self.use_contrast = use_contrast
        self.warmed_up_contrast = False
        self.warmup_contrast = warmup_contrast
        self.temperature = temperature
        self.beta_contrast = beta_contrast
        
        self.ref_mode = ref_mode
        self.shuffle = shuffle
        self.use_seqids = use_seqids
        self.same_class = same_class

        self.n_max_frames = n_max_frames
        
        self.rm_zero_q = rm_zero_q
        self.eps = eps

        self.n_ref_seq = n_ref_seq

        self.version = version

        self.lr = lr
        self.automatic_optimization = False

        # for validation step
        self.n_examples = n_ex_val
        self.validation_step_outputs = {}


    def configure_optimizers(self):
        params = list(self.control_extractor.parameters())
        params += list(self.latent_projector.parameters())

        # not sure which one to use depending on the architecture (convnet, transformer, etc.)
        # opt = torch.optim.AdamW(params, self.lr, (.9, .999), weight_decay=1e-2)
        opt = torch.optim.Adam(params, self.lr, (.9, .999))

        return opt
    
    ###

    def convert_class_values_to_indices(self, a):
        new_a = []
        for i in range(self.n_controls):
            if len(self.class_values[i]) > 0:
                classes_i = torch.tensor(self.class_values[i]).to(a.device)
                a_i = a[:, i, :]
                a_ids = torch.searchsorted(classes_i.contiguous(), a_i.contiguous())
            else:
                a_ids = a[:, i, :]
            new_a.append(a_ids.unsqueeze(1))
        new_a = torch.cat(new_a, dim=1)
        return new_a

    def process_attributes(self, attr):
        attr = attr.to(self.device)
        if len(attr.shape) == 1:
            attr = attr.unsqueeze(1)
        if self.class_values is not None:
            attr = self.convert_class_values_to_indices(attr)

        # assign class value to quantized attributes
        for k in range(self.n_controls):
            if self.quantized[k]:
                data = attr[:, k, :].flatten()
                classes = torch.bucketize(data, self.bins_values[k].to(attr.device))
                attr[:, k, :] = classes.reshape(-1, attr.shape[-1])
                
        attr = attr.to(torch.int32)
        
        return attr
    
    ### cf. ELBO:

    @staticmethod
    def reparametrize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        z = torch.randn_like(mu) * std + mu
        return z
    
    @staticmethod
    def compute_kl(mu, logvar):
        kl = (- 0.5 * (1 + logvar - mu.pow(2) - torch.exp(logvar))).sum(1).mean()
        return kl
    
    @staticmethod
    def compute_reconstruction_loss(x, x_hat, method='bce'):
        if method == 'bce':
            loss = torch.nn.functional.binary_cross_entropy(x_hat, x, reduction="none").sum(1).mean()
        elif method == 'mse':
            loss = torch.nn.functional.mse_loss(x, x_hat, reduction="none").sum(1).mean()
        else:
            raise NotImplementedError(f'no implementaion for method {method}')
        return loss

    ### Contrastive loss 

    ### reference selection mode
    
    @staticmethod
    def force_onsets_every_n(mask, n_max_frames):
        batch_size, seq_length = mask.shape
        out = mask.clone()

        for b in range(batch_size):
            count = 0
            for t in range(seq_length):
                if out[b, t]:
                    count = 0
                else:
                    count += 1
                    if count >= n_max_frames:
                        out[b, t] = True
                        count = 0

        return out


    def aggregate_consecutive_control_frames(self, c, labels, k, all_seg_ids=None):
        if all_seg_ids is None:
            # detect segment boundaries per batch
            change = torch.ones_like(labels, dtype=torch.bool)
            change[:, 1:] = labels[:, 1:] != labels[:, :-1]

            if self.n_max_frames[k] is not None:
                change = self.force_onsets_every_n(change, self.n_max_frames[k])

            batch_ids = torch.zeros_like(labels) + torch.arange(labels.shape[0], device=labels.device).unsqueeze(-1)
            batch_ids = batch_ids[change]

            # get averaged frames labels
            a_avg = labels[change]

            # compute per-batch segment IDs
            seg_ids = change.cumsum(dim=1) - 1
        else:
            seg_ids = all_seg_ids[:,k,:]
            change = torch.ones_like(seg_ids, dtype=torch.bool)
            change[:, 1:] = seg_ids[:, 1:] != seg_ids[:, :-1]
            batch_ids = torch.zeros_like(seg_ids) + torch.arange(seg_ids.shape[0], device=seg_ids.device).unsqueeze(-1)
            batch_ids = batch_ids[change]
            # get averaged frames labels
            a_avg = labels[change]
            
            
        # offset segment IDs so they are unique across batch
        seg_ids_max = seg_ids.max(dim=1).values + 1
        offsets = seg_ids_max.cumsum(dim=0)
        seg_offsets = torch.zeros_like(offsets)
        seg_offsets[1:] = offsets[:-1]
        seg_ids_unique = seg_ids + seg_offsets[:, None]

        # flatten c and seg_ids
        c_flat = rearrange(c, "b d t -> (b t) d")
        seg_flat = seg_ids_unique.reshape(-1)

        # control_dim = self.control_dim
        control_dim = c_flat.shape[-1]
        # compute mean per varying-length window control frames
        n_frames = seg_flat[-1].item() + 1
        sums = torch.zeros(n_frames, control_dim, device=c.device)
        counts = torch.zeros(n_frames, device=c.device)

        sums.scatter_add_(0, seg_flat.unsqueeze(1).expand(-1, control_dim), c_flat)
        counts.scatter_add_(0, seg_flat, torch.ones_like(seg_flat, dtype=torch.float))

        c_avg = sums / counts.unsqueeze(1) 
        return c_avg, a_avg, batch_ids


    def process_reference_control_embeddings(self, c, labels, k, seg_ids=None):
        if self.ref_mode == 'agg':
            c_ref, a_ref, batch_ids = self.aggregate_consecutive_control_frames(c, labels, k, seg_ids)
        else:
            raise NotImplementedError(f"No reference selection method implemented for mode: {self.ref_mode}")
        
        return c_ref, a_ref, batch_ids
    
    ###

    def pad_by_random_sampling(self, indices, max_len):
        indices_filled = torch.empty(len(indices), max_len, dtype=indices[0].dtype, device=self.device)

        for i, li in enumerate(indices):
            len_i = li.numel()
            indices_filled[i, :len_i] = li

            if len_i < max_len:
                # sample with replacement from existing values
                idx = torch.randint(len_i, (max_len - len_i,), device=self.device)
                indices_filled[i, len_i:] = li[idx]

        return indices_filled
    
    def partition_reference_embeddings(self, c, a, lut, classes, shuffle=False):
        n_classes = classes.size(0)

        # map each sample to its class index
        class_indices = lut[a]
        valid_samples = class_indices >= 0
        a = a[valid_samples]
        c = c[valid_samples]
        class_indices = class_indices[valid_samples]

        if shuffle:
            # shuffle all samples
            perm = torch.randperm(len(a), device=a.device)
            a = a[perm]
            c = c[perm]
            seq_ids = seq_ids[perm]
            class_indices = class_indices[perm]

        # sort by class
        sort_idx = torch.argsort(class_indices)
        c_sorted = c[sort_idx]
        class_indices_sorted = class_indices[sort_idx]
        a_sorted = a[sort_idx]

        # find class boundaries
        counts = torch.bincount(class_indices_sorted, minlength=n_classes)
        max_len = counts.max().item()
        max_per_class = (max_len // 2) + max_len % 2
        cum_counts = torch.cumsum(counts, 0)
        starts = torch.cat([torch.tensor([0], device=c.device), cum_counts[:-1]])

        # partition each classes
        counts_part = counts // 2 + counts % 2
        end_query = starts + counts_part
        idx_query = [torch.arange(si, ei, device=self.device) for si, ei in zip(starts.tolist(), end_query.tolist())]
        idx_query_balanced = self.pad_by_random_sampling(idx_query, max_per_class)
        a_query = a_sorted[idx_query_balanced]
        c_query = c_sorted[idx_query_balanced]

        idx_pos = [torch.arange(si, ei, device=self.device) for si, ei in zip(end_query.tolist(), cum_counts.tolist())]
        idx_pos_balanced = self.pad_by_random_sampling(idx_pos, max_per_class)
        a_pos = a_sorted[idx_pos_balanced]
        c_pos = c_sorted[idx_pos_balanced]

        return c_query, a_query, c_pos, a_pos
    
    
    def partition_reference_embeddings_using_seq_ids(self, c, a, seq_ids, lut, classes, shuffle=True):
        device = c.device

        # map samples to classes
        class_indices = lut[a]
        valid = class_indices >= 0

        c = c[valid]
        a = a[valid]
        seq_ids = seq_ids[valid]
        class_indices = class_indices[valid]

        # filter classes that are not in at least 2 sequences
        valid_class_mask = torch.zeros_like(class_indices, dtype=torch.bool)

        for cls in torch.unique(class_indices):
            cls_mask = class_indices == cls
            n_seq = torch.unique(seq_ids[cls_mask]).numel()
            if n_seq >= 2:
                valid_class_mask |= cls_mask

        c = c[valid_class_mask]
        a = a[valid_class_mask]
        seq_ids = seq_ids[valid_class_mask]
        class_indices = class_indices[valid_class_mask]

        # reindex classes to [0..n_valid-1]
        unique_classes, class_indices = torch.unique(
            class_indices, return_inverse=True
        )
        n_classes = unique_classes.numel()

        
        if shuffle:
            perm = torch.randperm(len(a), device=device)
            c = c[perm]
            a = a[perm]
            seq_ids = seq_ids[perm]
            class_indices = class_indices[perm]

        # sort by (class, seq)
        _, seq_remap = torch.unique(seq_ids, return_inverse=True)
        multiplier = seq_remap.max().item() + 1
        sort_key = class_indices * multiplier + seq_remap
        sort_idx = torch.argsort(sort_key)

        c = c[sort_idx]
        a = a[sort_idx]
        class_sorted = class_indices[sort_idx]
        seq_sorted = seq_remap[sort_idx]

        # detect block
        is_new_block = torch.ones_like(class_sorted, dtype=torch.bool)
        is_new_block[1:] = (
            (class_sorted[1:] != class_sorted[:-1]) |
            (seq_sorted[1:] != seq_sorted[:-1])
        )

        block_ids = is_new_block.cumsum(0) - 1
        n_blocks = block_ids.max().item() + 1

        block_class = class_sorted[is_new_block]
        block_sizes = torch.bincount(block_ids)

        block_ends = torch.cumsum(block_sizes, 0)
        block_starts = torch.cat(
            [torch.zeros(1, device=device, dtype=torch.long), block_ends[:-1]]
        )

        # split block per class
        blocks_per_class = torch.bincount(block_class, minlength=n_classes)
        query_blocks_per_class = (blocks_per_class + 1) // 2

        class_block_offsets = torch.cat(
            [torch.zeros(1, device=device, dtype=torch.long),
            torch.cumsum(blocks_per_class, 0)[:-1]]
        )

        block_ids_all = torch.arange(n_blocks, device=device)
        block_pos_in_class = block_ids_all - class_block_offsets[block_class]

        query_block_mask = block_pos_in_class < query_blocks_per_class[block_class]
        query_blocks = block_ids_all[query_block_mask]
        pos_blocks = block_ids_all[~query_block_mask]

        # collect indices per class
        idx_query = [[] for _ in range(n_classes)]
        idx_pos = [[] for _ in range(n_classes)]

        for bid in query_blocks.tolist():
            cls = block_class[bid].item()
            idx_query[cls].append(
                torch.arange(block_starts[bid], block_ends[bid], device=device)
            )

        for bid in pos_blocks.tolist():
            cls = block_class[bid].item()
            idx_pos[cls].append(
                torch.arange(block_starts[bid], block_ends[bid], device=device)
            )

        idx_query = [
            torch.cat(v) for v in idx_query
        ]
        idx_pos = [
            torch.cat(v) for v in idx_pos
        ]

        max_query = max(idx.numel() for idx in idx_query)
        max_pos = max(idx.numel() for idx in idx_pos)
        max_samples = max(max_query, max_pos)

        idx_query_bal = self.pad_by_random_sampling(idx_query, max_samples)
        idx_pos_bal = self.pad_by_random_sampling(idx_pos, max_samples)

        c_query = c[idx_query_bal]
        a_query = a[idx_query_bal]
        c_pos = c[idx_pos_bal]
        a_pos = a[idx_pos_bal]

        return c_query, a_query, c_pos, a_pos

    
    def sample_reference_sequences(self, c_query, a_query, c_pos, a_pos, batch_size, seq_length):
        control_dim = c_query.shape[-1]
        c_query_flat = c_query.view(-1, control_dim)
        
        idx = torch.randint(0, c_query_flat.shape[0], (batch_size, seq_length), device=c_query.device)

        c_ref_query = c_query_flat[idx].permute(0, 2, 1)  # (B, D, T)

        c_pos_flat = c_pos.view(-1, control_dim)
        c_ref_pos = c_pos_flat[idx].permute(0, 2, 1)  # (B, D, T)

        a_ref_query = a_query.view(-1)[idx] # (B, T)
        a_ref_pos = a_pos.view(-1)[idx] # (B, T)

        return c_ref_query, c_ref_pos, a_ref_query, a_ref_pos
    
    def compute_control_embedding_distance(self, c1, c2):
        return torch.abs(c1 - c2)
    
    def shuffle_cross_class(self, a):
        batch_size = a.size(0)

        # mask[i, j] = True if j can be assigned to i
        mask = a.unsqueeze(1) != a.unsqueeze(0)  # (B, B)

        # random scores
        rand = torch.rand(batch_size, batch_size, device=self.device)

        # invalidate same-class entries
        rand = rand.masked_fill(~mask, -1e9)

        # pick one valid index per row
        idx = rand.argmax(dim=1)  # (B,)

        return idx
    
    def compute_variation_sets(self, c_u, c_s, a_k, k, seg_ids=None, rm_other_controls=False):
        # compute aggregated average consecutive frames control embeddings for sampling reference
        batch_size = c_u.shape[0]
        seq_length = c_u.shape[-1]
        c_ref, a_ref, batch_ids = self.process_reference_control_embeddings(c_s, a_k, k, seg_ids) 
        if self.rm_silent_ref[k]:
            m = a_ref == 0
            c_ref = c_ref[~m]
            a_ref = a_ref[~m]
            batch_ids = batch_ids[~m]
        # process available classes in the current batch
        batch_classes, counts = a_ref.unique(return_counts=True)
        batch_classes = batch_classes[counts > 1]
        counts = counts[counts > 1]
        n_classes = batch_classes.size(0)

        # create fast class-to-index mapping (GPU LUT)
        max_class = a_ref.max().item()
        lut = torch.full((max_class + 1,), -1, device=a_k.device, dtype=torch.long)
        lut[batch_classes] = torch.arange(n_classes, device=a_k.device)

        # partition into 2 complementary sets to sample reference embeddings
        if self.use_seqids:
            c_query, a_query, c_pos, a_pos = self.partition_reference_embeddings_using_seq_ids(c_ref, a_ref, batch_ids, lut, batch_classes, shuffle=self.shuffle)
        else:
            c_query, a_query, c_pos, a_pos = self.partition_reference_embeddings(c_ref, a_ref, lut, batch_classes, shuffle=self.shuffle)

        # sample query and positive reference sequences 
        c_ref_query, c_ref_pos, a_ref_query, a_ref_pos = self.sample_reference_sequences(c_query, a_query, c_pos, a_pos, batch_size, seq_length)

        # compute variation sets on batch sequences
        v_seq_query = self.compute_control_embedding_distance(c_ref_query, c_u) # (B, D, T)
        v_seq_pos = self.compute_control_embedding_distance(c_ref_pos, c_u) # (B, D, T)

        v_seq_neg = torch.cat([v_seq_query.clone(), v_seq_pos.clone()], dim=-1) # (B, D, 2*T)
        v_seq_neg[:,k,:] = 0.
        v_seq_neg = v_seq_neg.detach()

        # compute variations on reference
        v_ref_query = []
        v_ref_pos = []
        v_ref_neg = []
        aq_flat = a_ref_query.view(-1) # (B * T)
        for j in range(self.n_ref_seq):
            idx_other_classes = self.shuffle_cross_class(aq_flat) # (B * T)
            if self.same_class:
                idx_pos_other_classes = idx_other_classes
            else:
                ap_flat = a_ref_pos.view(-1) # (B * T)
                idx_pos_other_classes = self.shuffle_cross_class(ap_flat) # (B * T)

            c_ref_query_flat = rearrange(c_ref_query, "b d t -> (b t) d")
            cq_ref_neg = c_ref_query_flat[idx_other_classes] # (B * T, D)
            cq_ref_neg = rearrange(cq_ref_neg, "(b t) d -> b d t", b=batch_size, t=seq_length) # (B, D, T)
            
            c_ref_pos_flat = rearrange(c_ref_pos, "b d t -> (b t) d")
            cp_ref_neg = c_ref_pos_flat[idx_pos_other_classes] # (B * T, D)
            cp_ref_neg = rearrange(cp_ref_neg, "(b t) d -> b d t", b=batch_size, t=seq_length) # (B, D, T)

            v_ref_query_j = self.compute_control_embedding_distance(c_ref_query, cp_ref_neg) # (B, D, T)
            v_ref_query.append(v_ref_query_j)
            v_ref_pos_j = self.compute_control_embedding_distance(c_ref_pos, cq_ref_neg)
            v_ref_pos.append(v_ref_pos_j)

            c_ref_neg_query = cq_ref_neg.clone()
            c_ref_neg_query[:,k,:] = c_ref_pos[:,k,:]

            c_ref_neg_pos = cp_ref_neg.clone()
            c_ref_neg_pos[:,k,:] = c_ref_query[:,k,:]

            c_ref_neg = torch.cat([c_ref_neg_query, c_ref_neg_pos], dim=-1)
            c_ref_query_pos = torch.cat([c_ref_query, c_ref_pos], dim=-1)
            v_ref_neg_j = self.compute_control_embedding_distance(c_ref_query_pos, c_ref_neg)
            v_ref_neg_j = v_ref_neg_j.detach()
            v_ref_neg.append(v_ref_neg_j)

        # final variation sets
        v_query = torch.cat([v_seq_query] + v_ref_query, dim=-1) # (B, D, (1+n)*T)
        v_pos = torch.cat([v_seq_pos] + v_ref_pos, dim=-1) # (B, D, (1+n)*T)
        v_neg = torch.cat([v_seq_neg] + v_ref_neg, dim=-1) # (B, D, 2(1+n)*T)

        if rm_other_controls:
            v_query = torch.cat([v_query[:,k:k+1,:], v_query[:,self.n_controls:,:]], dim=1)
            v_pos = torch.cat([v_pos[:,k:k+1,:], v_pos[:,self.n_controls:,:]], dim=1)
            v_neg = torch.cat([v_neg[:,k:k+1,:], v_neg[:,self.n_controls:,:]], dim=1)

        return v_query, v_pos, v_neg


    def infonce_sequence_level(self, q, p, n, init_seq_length, rm_zero_q=False, k=0, eps=1e-2, rm_other_controls=False):
        """
        q: (B, D, T)   queries
        p: (B, D, T)   positives (aligned with q)
        n: (B, D, 2T)  negatives (per batch element)
        """
        batch_size = q.shape[0]
        seq_length = q.shape[-1]            

        if rm_zero_q:
            if rm_other_controls:
                cdim = 0
            else: 
                cdim = k
            # FIX!!!
            # q_valid = (q[:,k,:] >= eps) # (B, T)
            q_valid = torch.ones_like(q[:, cdim, :], dtype=torch.bool)
            q_valid[:, :init_seq_length] = q[:, cdim, :init_seq_length] >= eps # (B, T)
            
            mask_flat = q_valid.view(batch_size * seq_length)
        
        q = torch.nn.functional.normalize(q, dim=1) # D dim
        p = torch.nn.functional.normalize(p, dim=1)
        n = torch.nn.functional.normalize(n, dim=1)

        # (B, T)
        l_pos = (q * p).sum(dim=1)

        # (B, T, 2T)
        l_neg = torch.einsum("bdt,bdk->btk", q, n)

        # logits: (B, T, 1 + 2T)
        logits = torch.cat([l_pos.unsqueeze(-1), l_neg], dim=-1)
        logits /= self.temperature

        logits_flat = logits.view(batch_size * seq_length, -1)
        
        # labels positive key at index 0 
        labels_flat = torch.zeros(batch_size * seq_length, dtype=torch.long, device=q.device)
        
        # cross-entropy over last dimension
        if rm_zero_q:
            loss = torch.nn.functional.cross_entropy(
                logits_flat[mask_flat],
                labels_flat[mask_flat],
                reduction="mean"
            )
        else:
            loss = torch.nn.functional.cross_entropy(
                logits_flat,
                labels_flat,
                reduction="mean"
            )

        # compute accuracy argmax over (1 + 2T) classes
        predicted_class = torch.argmax(logits, dim=-1).view(-1)
        if rm_zero_q:
            accuracy = (predicted_class[mask_flat] == labels_flat[mask_flat]).float().mean()
        else:
            accuracy = (predicted_class == labels_flat).float().mean()

        # log average cosine similarities
        self.log(f"cosine_similarity/pos_{k}", l_pos.mean(), prog_bar=True, on_step=True)
        self.log(f"cosine_similarity/neg_{k}", l_neg.mean(-1).mean(), prog_bar=True, on_step=True)
        
        return loss, accuracy
    
    ###

    @staticmethod
    def get_beta(beta, current_step, warmup_steps):
        warmed_up = False
        current_beta = 0.
        if current_step >= warmup_steps:
            warmed_up = True
            current_beta = min(beta, beta * (current_step - warmup_steps) / warmup_steps)
        return current_beta, warmed_up

    @staticmethod
    def get_beta_annealing(beta_max, step, warmup_steps):
        return min(beta_max, beta_max * step / warmup_steps)
    
    def training_step(self, batch, batch_idx):
    
        opt = self.optimizers()
        
        z_u, _ = batch["unlabel"]

        seg_ids = None
        if len(batch["label"]) == 3:
            z_s, a, seg_ids = batch["label"]
            a = a.to(self.device)
            seg_ids = seg_ids.to(self.device)
        else:
            z_s, y = batch["label"]

        z_u = z_u.to(self.device)
        z_s = z_s.to(self.device)

        n_u = z_u.shape[0]
        n_s = z_s.shape[0]

        init_seq_length = z_u.shape[-1]

        z = torch.cat([z_u, z_s], dim=0)

        # 1 - COMPUTE ELBO Latent VAE 
        # encode control
        mu_c, logvar_c = self.control_extractor(z)
        c = self.reparametrize(mu_c, logvar_c)

        # decode control
        z_hat = self.latent_projector(c)

        loss_rec_z = self.compute_reconstruction_loss(z, z_hat, method='mse')

        beta_kl_c = self.get_beta_annealing(self.beta_kl_c, self.global_step, self.warmup_kl_c)
        kl_controls_all = self.compute_kl(mu_c, logvar_c)
        kl_controls = (beta_kl_c * kl_controls_all)


        # 2 - COMPUTE CONTRASTIVE LOSS
        loss_contrastive = None
        accuracy = None
        if self.use_contrast:

            if seg_ids is None:
                a = self.process_attributes(y.clone())

            c_u = c[:n_u]
            c_s = c[n_u:]
            assert c_u.shape[0] == n_u
            assert c_s.shape[0] == n_s

            loss_contrastive = []
            accuracy = []

            for k in range(self.n_controls):
                q_set, pos_set, neg_set = self.compute_variation_sets(c_u, c_s, a[:,k,:], k, seg_ids, rm_other_controls=True)
                loss_contrastive_k, accuracy_k = self.infonce_sequence_level(q_set, pos_set, neg_set, init_seq_length, rm_zero_q=self.rm_zero_q, k=k, eps=self.eps, rm_other_controls=True)                            
                loss_contrastive.append(loss_contrastive_k)
                accuracy.append(accuracy_k)

            loss_contrastive = torch.stack(loss_contrastive).mean()

        # FULL LOSS
        loss = 0.

        loss += loss_rec_z

        loss += kl_controls

        if self.use_contrast:
            beta_contrast = self.get_beta_annealing(self.beta_contrast, self.global_step, self.warmup_contrast)
            loss += (beta_contrast * loss_contrastive)


        # OPTIMIZATION
        opt.zero_grad()
        loss.backward()
        opt.step()

        # LOG
        if loss is not None:
            self.log("loss", loss)

        if loss_rec_z is not None:
            self.log("loss reconstruction z", loss_rec_z)
        
        if kl_controls is not None:
            self.log("kl_c", kl_controls)
        
        if loss_contrastive is not None:
            self.log("loss contrastive", loss_contrastive) 
            for k in range(self.n_controls):
                if accuracy[k] is not None:
                    self.log(f"loss contrastive accuracy c{k}", accuracy[k]) 


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if self.trainer is not None:
            
            if len(batch) == 3:
                z, a, _ = batch
                a = a.to(self.device)
            else:
                z, y = batch
                a = self.process_attributes(y)
            z = z.to(self.device)


            # Evaluate reconstruction error
            c, _ = self.control_extractor(z)
            z_hat = self.latent_projector(c)
            loss_rec_z = self.compute_reconstruction_loss(z, z_hat, method='mse')
            self.log('validation', loss_rec_z)

            if self.n_examples > 0:
                c_flat = rearrange(c, "b d t -> (b t) d")
                a_flat = rearrange(a, "b k t -> (b t) k")
                perm = torch.randperm(len(a), device=a.device)
                a_flat = a_flat[perm]
                c_flat = c_flat[perm]

                for k in range(self.n_controls):
                    ex_ck = self.validation_step_outputs.get(f"c{k}", [])
                    n_ex_ck = 0
                    if len(ex_ck) > 0:
                        for n in ex_ck:
                            n_ex_ck += n.shape[0]

                    if len(ex_ck) == 0 or n_ex_ck < self.n_examples:
                        n_ex = self.n_examples - n_ex_ck
                        
                        self.validation_step_outputs[f"c{k}"] = self.validation_step_outputs.get(f"c{k}", []) + [c_flat[:n_ex, k].squeeze()]
                        self.validation_step_outputs[f"a{k}"] = self.validation_step_outputs.get(f"a{k}", []) + [a_flat[:n_ex, k].squeeze()]

    def on_validation_epoch_end(self):
        n_ex = len(self.validation_step_outputs.get("c0", []))
        if n_ex > 0:
            for k in range(self.n_controls):
                ck = torch.cat(self.validation_step_outputs.get(f"c{k}"))
                ak = torch.cat(self.validation_step_outputs.get(f"a{k}"))
                f = plot_class_distribution_per_control_dim(ck, ak, self.n_classes_per_factor[k], cdim=k, figsize=(8, 3))
                self.logger.experiment.add_figure(f"control distributions c_{k} - {self.all_attributes[k]}", f, self.global_step)

        self.validation_step_outputs = {}
