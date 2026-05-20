from __future__ import annotations

from itertools import permutations

import torch
import torch.nn.functional as F

from src.losses.losses import min_snr_weight, p2_weight, snr_weight, truncated_snr_weight
from src.metrics.metrics import chamfer_distance, earth_movers_distance, square_distance
from src.utils.symmetry_planes import normalize_active_planes, reflect_points

try:
    from scipy.optimize import linear_sum_assignment as scipy_linear_sum_assignment
except Exception:
    scipy_linear_sum_assignment = None

def weighted_chamfer_distance(x: torch.Tensor, y: torch.Tensor, weight_x: torch.Tensor, weight_y: torch.Tensor) -> torch.Tensor:
    dist_sq = square_distance(x, y)
    min_dist_x_to_y = torch.min(dist_sq, dim=2)[0]
    min_dist_y_to_x = torch.min(dist_sq, dim=1)[0]
    return torch.mean(min_dist_x_to_y * weight_x, dim=1) + torch.mean(min_dist_y_to_x * weight_y, dim=1)



class TrueJointSymmetryPlaneLoss:
    def __init__(
        self,
        lambda_diff: float = 1.0,
        lambda_plane: float = 1.0,
        lambda_recon: float = 0.0,
        lambda_plane_consistency: float = 0.0,
        lambda_boundary: float = 0.0,
        boundary_margin: float = 0.01,
        boundary_frac: float = 0.05,
        recon_cd_sigma: float = 1.0,
        plane_normal_weight: float = 1.0,
        plane_offset_weight: float = 1.0,
        metric: str = "cd",
        warmup_steps: int = 1000,
        geometry_mode: str = "half",
        weighting: str = "none",
        recon_timestep_weighting: str = "loss",
        gamma: float = 5.0,
        k: float = 1.0,
        min_snr: float = 0.01,
        max_snr: float = 100.0,
        plane_matching: str = "none",
        hungarian_backend: str = "auto",
        matching_normal_weight: float = 1.0,
        matching_offset_weight: float = 0.25,
        matching_presence_weight: float = 0.25,
        mask_plane_loss_by_presence: bool = False,
        lambda_plane_presence: float = 0.0,
        presence_pos_weight: float = 1.0,
        inactive_plane_norm_threshold: float = 1e-5,
    ):
        self.lambda_diff = float(lambda_diff)
        self.lambda_plane = float(lambda_plane)
        self.lambda_recon = float(lambda_recon)
        self.lambda_plane_consistency = float(lambda_plane_consistency)
        self.lambda_boundary = float(lambda_boundary)
        self.boundary_margin = float(boundary_margin)
        self.boundary_frac = float(boundary_frac)
        self.recon_cd_sigma = float(recon_cd_sigma)
        self.plane_normal_weight = float(plane_normal_weight)
        self.plane_offset_weight = float(plane_offset_weight)
        self.metric = str(metric).lower()
        self.warmup_steps = int(warmup_steps)
        self.geometry_mode = str(geometry_mode).lower()
        self.weighting = str(weighting).lower()
        self.recon_timestep_weighting = str(recon_timestep_weighting).lower()
        self.gamma = float(gamma)
        self.k = float(k)
        self.min_snr = float(min_snr)
        self.max_snr = float(max_snr)
        self.plane_matching = str(plane_matching).strip().lower()
        self.hungarian_backend = str(hungarian_backend).strip().lower()
        self.matching_normal_weight = float(matching_normal_weight)
        self.matching_offset_weight = float(matching_offset_weight)
        self.matching_presence_weight = float(matching_presence_weight)
        self.mask_plane_loss_by_presence = bool(mask_plane_loss_by_presence)
        self.lambda_plane_presence = float(lambda_plane_presence)
        self.presence_pos_weight = float(presence_pos_weight)
        self.inactive_plane_norm_threshold = float(inactive_plane_norm_threshold)
        self._perm_bank_cache: dict[int, torch.Tensor] = {}

        if self.plane_matching not in {"none", "hungarian"}:
            raise ValueError("loss.plane_matching must be 'none' or 'hungarian'")
        if self.hungarian_backend not in {"auto", "scipy", "bruteforce"}:
            raise ValueError("loss.hungarian_backend must be 'auto', 'scipy', or 'bruteforce'")

    def _presence_target(self, plane: torch.Tensor) -> torch.Tensor:
        return (torch.norm(plane[..., :3], dim=-1) > self.inactive_plane_norm_threshold).to(dtype=plane.dtype)

    def _gather_slots(self, tensor: torch.Tensor | None, slot_idx: torch.Tensor) -> torch.Tensor | None:
        if tensor is None:
            return None
        if tensor.dim() < 2:
            return tensor
        if tensor.shape[1] != slot_idx.shape[1]:
            return tensor
        if tensor.dim() == 2:
            return torch.gather(tensor, 1, slot_idx)
        view_shape = [slot_idx.shape[0], slot_idx.shape[1]] + [1] * (tensor.dim() - 2)
        gather_idx = slot_idx.view(*view_shape).expand(-1, -1, *tensor.shape[2:])
        return torch.gather(tensor, 1, gather_idx)

    def _plane_matching_cost(
        self,
        pred_plane_x0: torch.Tensor,
        gt_plane: torch.Tensor,
        pred_presence_prob: torch.Tensor | None,
    ) -> torch.Tensor:
        pred_plane_x0 = normalize_active_planes(pred_plane_x0, threshold=self.inactive_plane_norm_threshold)
        gt_plane = normalize_active_planes(gt_plane, threshold=self.inactive_plane_norm_threshold)

        normal_cost = ((pred_plane_x0[:, :, None, :3] - gt_plane[:, None, :, :3]) ** 2).sum(dim=-1)
        offset_cost = torch.abs(pred_plane_x0[:, :, None, 3] - gt_plane[:, None, :, 3])
        cost = (self.matching_normal_weight * normal_cost) + (self.matching_offset_weight * offset_cost)

        if self.matching_presence_weight > 0.0:
            gt_presence = self._presence_target(gt_plane)
            if pred_presence_prob is None:
                pred_presence = (torch.norm(pred_plane_x0[..., :3], dim=-1) > self.inactive_plane_norm_threshold).to(pred_plane_x0.dtype)
            else:
                pred_presence = pred_presence_prob
            cost = cost + (self.matching_presence_weight * torch.abs(pred_presence[:, :, None] - gt_presence[:, None, :]))
        return cost

    def _perm_bank(self, num_planes: int, device: torch.device) -> torch.Tensor:
        bank = self._perm_bank_cache.get(int(num_planes))
        if bank is None:
            bank = torch.tensor(list(permutations(range(int(num_planes)))), dtype=torch.long)
            self._perm_bank_cache[int(num_planes)] = bank
        return bank.to(device=device)

    def _solve_assignment_bruteforce(self, cost: torch.Tensor) -> torch.Tensor:
        bsz, num_planes, _ = cost.shape
        perm_bank = self._perm_bank(num_planes, device=cost.device)
        gt_idx = torch.arange(num_planes, device=cost.device, dtype=torch.long)
        score_list = []
        for k in range(perm_bank.shape[0]):
            perm = perm_bank[k]
            score_list.append(cost[:, perm, gt_idx].sum(dim=-1))
        scores = torch.stack(score_list, dim=1)
        best_idx = scores.argmin(dim=1)
        return perm_bank[best_idx]

    def _solve_assignment_scipy(self, cost: torch.Tensor) -> torch.Tensor:
        if scipy_linear_sum_assignment is None:
            raise RuntimeError("scipy is not available for hungarian_backend='scipy'")
        bsz, num_planes, _ = cost.shape
        out = torch.empty((bsz, num_planes), dtype=torch.long, device=cost.device)
        cost_np = cost.detach().cpu().numpy()
        for b in range(bsz):
            rows, cols = scipy_linear_sum_assignment(cost_np[b])
            pred_for_gt = torch.empty(num_planes, dtype=torch.long)
            pred_for_gt[torch.as_tensor(cols, dtype=torch.long)] = torch.as_tensor(rows, dtype=torch.long)
            out[b] = pred_for_gt.to(device=cost.device)
        return out

    def _solve_assignment(self, cost: torch.Tensor) -> torch.Tensor:
        if self.hungarian_backend == "bruteforce":
            return self._solve_assignment_bruteforce(cost)
        if self.hungarian_backend == "scipy":
            return self._solve_assignment_scipy(cost)
        # auto
        if cost.shape[1] <= 7:
            return self._solve_assignment_bruteforce(cost)
        if scipy_linear_sum_assignment is not None:
            return self._solve_assignment_scipy(cost)
        return self._solve_assignment_bruteforce(cost)

    def _align_predicted_planes(
        self,
        eps_plane_pred: torch.Tensor,
        plane_x0_pred: torch.Tensor,
        plane_target: torch.Tensor,
        presence_logits: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.plane_matching != "hungarian":
            return eps_plane_pred, plane_x0_pred, presence_logits
        if eps_plane_pred.dim() != 3 or plane_x0_pred.dim() != 3 or plane_target.dim() != 3:
            return eps_plane_pred, plane_x0_pred, presence_logits

        pred_presence_prob = None
        if presence_logits is not None:
            pred_presence_prob = torch.sigmoid(presence_logits)
        cost = self._plane_matching_cost(plane_x0_pred, plane_target, pred_presence_prob)
        slot_idx = self._solve_assignment(cost)
        return (
            self._gather_slots(eps_plane_pred, slot_idx),
            self._gather_slots(plane_x0_pred, slot_idx),
            self._gather_slots(presence_logits, slot_idx),
        )

    def _timestep_weight(self, alpha_bar_t: torch.Tensor, mode: str) -> torch.Tensor:
        if mode in {"none", ""}:
            return torch.ones_like(alpha_bar_t)
        if mode == "snr":
            return snr_weight(alpha_bar_t)
        if mode == "min_snr":
            return min_snr_weight(alpha_bar_t, gamma=self.gamma)
        if mode == "p2":
            return p2_weight(alpha_bar_t, k=self.k, gamma=self.gamma)
        if mode == "truncated_snr":
            return truncated_snr_weight(alpha_bar_t, min_snr=self.min_snr, max_snr=self.max_snr)
        if mode == "alpha":
            return alpha_bar_t
        raise ValueError("Invalid loss.recon_timestep_weighting. Expected 'loss', 'none', 'snr', 'min_snr', 'p2', 'truncated_snr', or 'alpha'")

    def __call__(
        self,
        model_output: dict,
        eps_points: torch.Tensor,
        eps_plane: torch.Tensor,
        x_t: torch.Tensor | None = None,
        plane_t: torch.Tensor | None = None,
        x0: torch.Tensor | None = None,
        x0_input: torch.Tensor | None = None,
        plane0: torch.Tensor | None = None,
        alpha_bar_t: torch.Tensor | None = None,
        current_step: int | None = None,
        **kwargs,
    ):
        eps_points_pred = model_output["eps_points"]
        eps_plane_pred = model_output["eps_plane"]
        presence_logits = model_output.get("plane_presence_logits")

        loss_diff_raw = F.mse_loss(eps_points_pred, eps_points, reduction="none")
        loss_diff_per = loss_diff_raw.mean(dim=tuple(range(1, loss_diff_raw.dim())))
        weight = None
        if alpha_bar_t is not None:
            weight = self._timestep_weight(alpha_bar_t, self.weighting)
            loss_diff = (loss_diff_per * weight).mean()
        else:
            loss_diff = loss_diff_per.mean()

        loss_plane = torch.zeros((), device=loss_diff.device, dtype=loss_diff.dtype)
        loss_recon = torch.zeros((), device=loss_diff.device, dtype=loss_diff.dtype)
        loss_plane_consistency = torch.zeros((), device=loss_diff.device, dtype=loss_diff.dtype)
        loss_boundary = torch.zeros((), device=loss_diff.device, dtype=loss_diff.dtype)
        loss_plane_presence = torch.zeros((), device=loss_diff.device, dtype=loss_diff.dtype)

        if plane_t is None or alpha_bar_t is None:
            raise ValueError("true joint loss requires plane_t and alpha_bar_t")
        if plane0 is None:
            raise ValueError("true joint loss requires plane0")

        batch_size = plane_t.shape[0]
        abar_plane = alpha_bar_t.view(*([-1] + [1] * (plane_t.dim() - 1)))
        plane_x0_pred = (plane_t - torch.sqrt((1.0 - abar_plane).clamp(min=1e-8)) * eps_plane_pred) / torch.sqrt(abar_plane.clamp(min=1e-8))
        plane_x0_pred = normalize_active_planes(plane_x0_pred, threshold=self.inactive_plane_norm_threshold)

        eps_plane_pred, plane_x0_pred, presence_logits = self._align_predicted_planes(
            eps_plane_pred=eps_plane_pred,
            plane_x0_pred=plane_x0_pred,
            plane_target=plane0,
            presence_logits=presence_logits,
        )

        loss_plane_raw = F.mse_loss(eps_plane_pred, eps_plane, reduction="none")
        plane_reduce_dims = tuple(range(1, loss_plane_raw.dim()))
        if self.mask_plane_loss_by_presence:
            presence_target = self._presence_target(plane0)
            active = presence_target.unsqueeze(-1)
            per_sample_num = (loss_plane_raw * active).sum(dim=plane_reduce_dims)
            per_sample_den = (active.sum(dim=tuple(range(1, active.dim()))).clamp(min=1.0) * float(loss_plane_raw.shape[-1]))
            loss_plane_per = per_sample_num / per_sample_den
        else:
            loss_plane_per = loss_plane_raw.mean(dim=plane_reduce_dims)
        if weight is not None:
            loss_plane = (loss_plane_per * weight).mean()
        else:
            loss_plane = loss_plane_per.mean()

        if self.lambda_plane_presence > 0.0:
            if presence_logits is None:
                raise ValueError("loss.lambda_plane_presence > 0 requires model_output['plane_presence_logits']")
            presence_target = self._presence_target(plane0)
            pos_weight = torch.tensor(self.presence_pos_weight, dtype=presence_logits.dtype, device=presence_logits.device)
            bce_raw = F.binary_cross_entropy_with_logits(
                presence_logits,
                presence_target,
                pos_weight=pos_weight,
                reduction="none",
            )
            loss_presence_per = bce_raw.mean(dim=1)
            if weight is not None:
                loss_plane_presence = (loss_presence_per * weight).mean()
            else:
                loss_plane_presence = loss_presence_per.mean()

        need_x0_pred = self.lambda_recon > 0.0 or self.lambda_plane_consistency > 0.0 or self.lambda_boundary > 0.0
        if need_x0_pred:
            if x_t is None:
                raise ValueError("true joint recon/consistency losses require x_t")
            batch_size = x_t.shape[0]
            abar_points = alpha_bar_t.view(batch_size, 1, 1)
            x0_pred = (x_t - torch.sqrt((1.0 - abar_points).clamp(min=1e-8)) * eps_points_pred) / torch.sqrt(abar_points.clamp(min=1e-8))

            if self.lambda_plane_consistency > 0.0:
                plane_target = normalize_active_planes(plane0, threshold=self.inactive_plane_norm_threshold)
                # Compute active plane mask (batch_size, num_planes)
                active_mask = (plane_target[..., :3].norm(dim=-1) > self.inactive_plane_norm_threshold).float()
                
                # Replace inactive slots with dummy non-zero normals to avoid NaN gradients in F.cosine_similarity
                dummy_normal = torch.tensor([1.0, 0.0, 0.0], dtype=plane_target.dtype, device=plane_target.device).view(1, 1, 3)
                plane_target_safe = torch.where(active_mask.unsqueeze(-1) > 0.5, plane_target[..., :3], dummy_normal)
                plane_pred_safe = torch.where(active_mask.unsqueeze(-1) > 0.5, plane_x0_pred[..., :3], dummy_normal)
                
                normal_cos = F.cosine_similarity(plane_pred_safe, plane_target_safe, dim=-1)
                loss_plane_normal_raw = 1.0 - normal_cos
                loss_plane_normal_masked = loss_plane_normal_raw * active_mask
                loss_plane_normal = loss_plane_normal_masked.sum() / active_mask.sum().clamp(min=1.0)
                
                loss_plane_offset_raw = F.smooth_l1_loss(plane_x0_pred[..., 3], plane_target[..., 3], reduction="none")
                loss_plane_offset_masked = loss_plane_offset_raw * active_mask
                loss_plane_offset = loss_plane_offset_masked.sum() / active_mask.sum().clamp(min=1.0)
                
                loss_plane_consistency = (self.plane_normal_weight * loss_plane_normal) + (self.plane_offset_weight * loss_plane_offset)

            if self.lambda_boundary > 0.0:
                recon_plane = normalize_active_planes(plane0, threshold=self.inactive_plane_norm_threshold)
                if recon_plane.dim() == 2:
                    recon_plane_iter = recon_plane.unsqueeze(1)
                else:
                    recon_plane_iter = recon_plane
                
                loss_boundary_acc = torch.zeros((), device=loss_diff.device, dtype=loss_diff.dtype)
                active_plane_count = 0
                for p_idx in range(recon_plane_iter.shape[1]):
                    p_curr = recon_plane_iter[:, p_idx, :]
                    active_plane = (torch.norm(p_curr[:, :3], dim=-1) > self.inactive_plane_norm_threshold)
                    if not bool(active_plane.any()):
                        continue
                    normals = p_curr[:, :3].unsqueeze(1)
                    offsets = p_curr[:, 3].unsqueeze(1).unsqueeze(2)
                    dists_to_plane = torch.abs(torch.bmm(x0_pred, normals.transpose(1, 2)) + offsets).squeeze(-1)
                    min_dists, _ = torch.topk(dists_to_plane, k=max(1, int(x0_pred.shape[1] * self.boundary_frac)), dim=1, largest=False)
                    per_sample = torch.relu(min_dists - self.boundary_margin).mean(dim=1)
                    weight_active = active_plane.to(per_sample.dtype)
                    loss_boundary_acc += (per_sample * weight_active).sum() / weight_active.sum().clamp(min=1.0)
                    active_plane_count += 1
                if active_plane_count > 0:
                    loss_boundary = loss_boundary_acc / float(active_plane_count)
                else:
                    loss_boundary = torch.zeros((), device=loss_diff.device, dtype=loss_diff.dtype)
            else:
                loss_boundary = torch.zeros((), device=loss_diff.device, dtype=loss_diff.dtype)

            if self.lambda_recon > 0.0:
                if x0 is None:
                    raise ValueError("loss.lambda_recon > 0 requires x0")
                if self.geometry_mode == "full":
                    x0_reconstructed = x0_pred
                    weight_x = torch.ones(x0_reconstructed.shape[:2], device=x0_reconstructed.device)
                    weight_y = torch.ones(x0.shape[:2], device=x0.device)
                else:
                    recon_plane = normalize_active_planes(plane0, threshold=self.inactive_plane_norm_threshold)
                    if self.warmup_steps > 0 and current_step is not None:
                        warmup_progress = min(1.0, float(current_step) / float(self.warmup_steps))
                        if warmup_progress < 1.0:
                            with torch.no_grad():
                                prob = torch.rand(recon_plane.shape[0], device=recon_plane.device)
                                use_pred = prob < warmup_progress
                            
                            use_pred_expanded = use_pred
                            for _ in range(recon_plane.dim() - 1):
                                use_pred_expanded = use_pred_expanded.unsqueeze(-1)
                            use_pred_expanded = use_pred_expanded.expand_as(recon_plane)
                            recon_plane = torch.where(use_pred_expanded, plane_x0_pred, recon_plane)
                            recon_plane = normalize_active_planes(recon_plane, threshold=self.inactive_plane_norm_threshold)
                            
                    if recon_plane.dim() == 2:
                        recon_plane_iter = recon_plane.unsqueeze(1)
                    else:
                        recon_plane_iter = recon_plane
                        
                    reconstructed_parts = [x0_pred]
                    for p_idx in range(recon_plane_iter.shape[1]):
                        p_curr = recon_plane_iter[:, p_idx, :]
                        if not bool((torch.norm(p_curr[:, :3], dim=-1) > self.inactive_plane_norm_threshold).any()):
                            continue
                        reconstructed_parts.append(reflect_points(x0_pred, p_curr))
                    x0_reconstructed = torch.cat(reconstructed_parts, dim=1)
                    
                    weight_x_acc = torch.ones_like(x0_reconstructed[..., 0])
                    weight_y_acc = torch.ones_like(x0[..., 0])
                    
                    # Compute minimum distance to any plane for weighting
                    for p_idx in range(recon_plane_iter.shape[1]):
                        p_curr = recon_plane_iter[:, p_idx, :]
                        if not bool((torch.norm(p_curr[:, :3], dim=-1) > self.inactive_plane_norm_threshold).any()):
                            continue
                        normals_recon = p_curr[:, :3].unsqueeze(1)
                        offsets_recon = p_curr[:, 3].unsqueeze(1).unsqueeze(2)
                        
                        dists_pred = torch.abs(torch.bmm(x0_reconstructed, normals_recon.transpose(1, 2)) + offsets_recon).squeeze(-1)
                        w_x = torch.exp(- (dists_pred ** 2) / (2 * self.recon_cd_sigma ** 2))
                        weight_x_acc = torch.min(weight_x_acc, w_x)
                        
                        dists_gt = torch.abs(torch.bmm(x0, normals_recon.transpose(1, 2)) + offsets_recon).squeeze(-1)
                        w_y = torch.exp(- (dists_gt ** 2) / (2 * self.recon_cd_sigma ** 2))
                        weight_y_acc = torch.min(weight_y_acc, w_y)
                        
                    weight_x = weight_x_acc
                    weight_y = weight_y_acc

                if self.metric == "emd":
                    loss_recon = earth_movers_distance(x0_reconstructed, x0).mean()
                else:
                    loss_recon = weighted_chamfer_distance(x0_reconstructed, x0, weight_x, weight_y).mean()
                loss_recon = torch.nan_to_num(loss_recon, nan=0.0, posinf=0.0, neginf=0.0)
                recon_mode = self.recon_timestep_weighting
                if recon_mode == "loss":
                    recon_mode = self.weighting
                loss_recon = loss_recon * self._timestep_weight(alpha_bar_t, str(recon_mode).lower()).mean()

        recon_weight = self.lambda_recon
        if self.warmup_steps > 0 and current_step is not None:
            recon_weight = self.lambda_recon * min(1.0, float(current_step) / float(self.warmup_steps))

        total_loss = (
            (self.lambda_diff * loss_diff)
            + (self.lambda_plane * loss_plane)
            + (self.lambda_plane_presence * loss_plane_presence)
            + (recon_weight * loss_recon)
            + (self.lambda_plane_consistency * loss_plane_consistency)
            + (self.lambda_boundary * loss_boundary)
        )

        return (
            total_loss,
            loss_diff,
            loss_plane,
            loss_recon,
            loss_plane_consistency,
            loss_boundary,
        )

def build_true_joint_symmetry_plane_loss(cfg: dict) -> TrueJointSymmetryPlaneLoss:
    loss_cfg = cfg.get("loss", {})
    model_cfg = cfg.get("model", {}) or {}
    joint_cfg = cfg.get("joint_symmetry", {}) or {}
    weighting = str(loss_cfg.get("weighting", "none")).lower()
    return TrueJointSymmetryPlaneLoss(
        lambda_diff=float(loss_cfg.get("lambda_diff", 1.0)),
        lambda_plane=float(loss_cfg.get("lambda_plane", 1.0)),
        lambda_recon=float(loss_cfg.get("lambda_recon", 0.0)),
        lambda_plane_consistency=float(loss_cfg.get("lambda_plane_consistency", 0.0)),
        lambda_boundary=float(loss_cfg.get("lambda_boundary", 0.0)),
        boundary_margin=float(loss_cfg.get("boundary_margin", 0.01)),
        boundary_frac=float(loss_cfg.get("boundary_frac", 0.05)),
        recon_cd_sigma=float(loss_cfg.get("recon_cd_sigma", 0.1)),
        plane_normal_weight=float(loss_cfg.get("plane_normal_weight", 1.0)),
        plane_offset_weight=float(loss_cfg.get("plane_offset_weight", 1.0)),
        metric=str(loss_cfg.get("metric", "cd")).lower(),
        warmup_steps=int(loss_cfg.get("warmup_steps", 1000)),
        geometry_mode=str(joint_cfg.get("geometry_mode", model_cfg.get("joint_geometry_mode", "half"))).lower(),
        weighting=weighting,
        recon_timestep_weighting=str(loss_cfg.get("recon_timestep_weighting", weighting)).lower(),
        gamma=float(loss_cfg.get("gamma", 5.0)),
        k=float(loss_cfg.get("k", 1.0)),
        min_snr=float(loss_cfg.get("min_snr", 0.01)),
        max_snr=float(loss_cfg.get("max_snr", 100.0)),
        plane_matching=str(loss_cfg.get("plane_matching", "none")).lower(),
        hungarian_backend=str(loss_cfg.get("hungarian_backend", "auto")).lower(),
        matching_normal_weight=float(loss_cfg.get("matching_normal_weight", 1.0)),
        matching_offset_weight=float(loss_cfg.get("matching_offset_weight", 0.25)),
        matching_presence_weight=float(loss_cfg.get("matching_presence_weight", 0.25)),
        mask_plane_loss_by_presence=bool(loss_cfg.get("mask_plane_loss_by_presence", False)),
        lambda_plane_presence=float(loss_cfg.get("lambda_plane_presence", 0.0)),
        presence_pos_weight=float(loss_cfg.get("presence_pos_weight", 1.0)),
        inactive_plane_norm_threshold=float(joint_cfg.get("inactive_plane_norm_threshold", 1e-5)),
    )
