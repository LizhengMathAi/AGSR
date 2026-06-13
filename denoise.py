"""
Reusable denoising utilities for 3-channel cardiac segmentation sequences.

Channel order:
    0 = RV
    1 = MYO
    2 = LV

Main entry point:
    denoise_sequence_with_anchor(noisy_seq, clean_anchor_frame, anchor_idx, ...)
"""

import numpy as np
from skimage.measure import label
from skimage.morphology import dilation, closing, convex_hull_image, disk, remove_small_objects, remove_small_holes
from skimage.registration import optical_flow_tvl1
from skimage.transform import warp


# ---------------------------------------------------------------------
# Basic mask helpers
# ---------------------------------------------------------------------

def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Return only the largest connected component of a binary mask."""
    mask = mask.astype(bool)
    if not mask.any():
        return mask

    labeled = label(mask)
    best_label = 0
    best_area = 0

    for k in range(1, labeled.max() + 1):
        area = np.sum(labeled == k)
        if area > best_area:
            best_label = k
            best_area = area

    return labeled == best_label


def _remove_small_components(mask: np.ndarray, max_size: int) -> np.ndarray:
    """Remove connected components smaller than or equal to max_size pixels."""
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    return remove_small_objects(mask)


def _clean_convex_mask(mask: np.ndarray, max_size: int) -> np.ndarray:
    """Clean RV/LV masks: remove tiny pieces, keep largest component, convex-hull it."""
    mask = _remove_small_components(mask, max_size)
    if not mask.any():
        return mask
    return convex_hull_image(_largest_component(mask))


# def _clean_myo_mask(mask: np.ndarray, lv_mask: np.ndarray | None, max_size: int) -> np.ndarray:
#     """Clean MYO mask without convex-hulling, because convex hull would fill the ring."""
#     mask = _remove_small_components(mask, max_size)
#     if not mask.any():
#         return mask

#     mask = closing(_largest_component(mask), disk(2))
#     if lv_mask is not None and lv_mask.any():
#         mask &= ~lv_mask
#     return mask


# ---------------------------------------------------------------------
# Anatomy-aware relabeling
# ---------------------------------------------------------------------

def _rgb2mask(frame: np.ndarray, thres: np.ndarray) -> np.ndarray:
    """
    frame: (H, W, 3) uint8, input RGB frame.
    thres: (3,) float, threshold for each channel.
    return: (H, W) uint8, binary mask.
    """
    mask = np.stack([np.where(frame[..., label-1] > 255*thres[label-1], frame[..., label-1], 0) for label in range(1, len(thres)+1)], axis=-1)
    mask = np.concatenate([np.zeros_like(mask[..., [0]]), mask], axis=-1)
    mask = np.argmax(mask, axis=-1)
    return mask

def _get_reference_frame(frame_seq: np.ndarray, threshold: np.ndarray = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    frame_seq: (T, H, W, 3) uint8, sequence of noisy frames to be relabeled.
    return: (H, W, 3) uint8, reference frame obtained by pixel-wise majority voting across the sequence.
    """
    if threshold is None: threshold = 255 * np.array([[[[0.15, 0.3, 0.05]]]])
    frame_seq = np.where(frame_seq > threshold, frame_seq, 0)
    ref_counts = frame_seq.sum(axis=0)  # [H, W, 3], voting counts
    ref_nonzero = ref_counts.sum(axis=-1) > 0  # [H, W], whether there is any vote
    ref_labels = np.argmax(ref_counts, axis=-1) + 1  # [H, W], voting labels
    ref = np.where(ref_nonzero, ref_labels, 0)  # [H, W], voting result with 0 for no votes, otherwise the label with most votes
    return np.stack([(ref == label).astype(np.uint8) for label in range(1, 4)], axis=-1) * 255


def _relabel_by_myo(
    reference_frame: np.ndarray,
    noisy_frame: np.ndarray,
    max_size: int = 8,
    threshold: np.ndarray = None,
) -> np.ndarray:
    """
    reference_frame: (H, W, 3) uint8, clean reference frame used for anatomical guidance.
    noisy_frame: (H, W, 3) uint8, noisy frame to be relabeled according to reference_frame anatomy.
    max_size: int, remove connected components with area <= max_size.
    return: (H, W, 3) uint8, relabeled frame with the same anatomy as reference_frame and cleaned by max_size.
    """

    rv_ref, myo_ref, lv_ref = (reference_frame[..., c] > 0 for c in range(3))
    if threshold is None: threshold = 255 * np.array([[[0.15, 0.3, 0.05]]])
    noisy_frame = np.where(noisy_frame > threshold, noisy_frame, 0)
    rv_noisy, myo_noisy, lv_noisy = (
        _remove_small_components(noisy_frame[..., c] > 0, max_size)
        for c in range(3)
    )

    def holes_inside(mask: np.ndarray) -> np.ndarray:
        inv = ~mask.astype(bool)
        lab = label(inv)
        holes = inv.copy()

        border_labels = np.unique(
            np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])
        )
        for b in border_labels:
            holes[lab == b] = False

        return holes

    def is_convex(mask: np.ndarray) -> bool:
        if not mask.any():
            return False
        return np.array_equal(mask, convex_hull_image(mask))

    myo = _largest_component(myo_noisy) if myo_noisy.any() else myo_ref.copy()
    myo = closing(myo, disk(2)) if myo.any() else myo

    rv = np.zeros_like(rv_ref, dtype=bool)
    lv = np.zeros_like(lv_ref, dtype=bool)

    if myo.any():
        holes = holes_inside(myo)

        # Rule 1: MYO is a closed ring; LV must fully fill inside MYO if LV exists.
        if holes.any():
            if ((lv_noisy | lv_ref) & holes).any():
                lv = holes.copy()
            myo &= ~lv

        else:
            myo_hull = convex_hull_image(myo)

            # Rule 2: MYO is not a ring and not convex; add convex shell.
            if not is_convex(myo):
                shell_inside = myo_hull & ~myo

                if ((lv_noisy | lv_ref) & shell_inside).any():
                    lv = shell_inside.copy()

                myo = myo_hull & ~lv

            # Rule 3: MYO is not a ring but convex disk; LV must be empty.
            else:
                lv[:] = False

    # Rule 4: RV must be outside MYO.
    forbidden_inside = convex_hull_image(myo | lv) if (myo | lv).any() else myo
    rv_candidate = rv_noisy & ~forbidden_inside

    if rv_candidate.any():
        rv = _clean_convex_mask(rv_candidate, max_size)
        rv &= ~forbidden_inside

    # Rule 5: if a class does not exist in reference_frame, it cannot appear in out.
    if not rv_ref.any():
        rv[:] = False
    if not myo_ref.any():
        myo[:] = False
    if not lv_ref.any():
        lv[:] = False

    # Final non-overlap.
    lv &= ~myo
    rv &= ~myo
    rv &= ~lv

    out = np.zeros_like(reference_frame, dtype=np.uint8)
    out[..., 0] = rv.astype(np.uint8) * 255
    out[..., 1] = myo.astype(np.uint8) * 255
    out[..., 2] = lv.astype(np.uint8) * 255
    return out


# ---------------------------------------------------------------------
# Optical-flow interpolation
# ---------------------------------------------------------------------


def _warp_frame_toward_reference(cleaned_noisy, reference_frame, rho):
    src = cleaned_noisy.max(axis=-1) > 0
    ref = reference_frame.max(axis=-1) > 0

    flow_y, flow_x = optical_flow_tvl1(
        src.astype(np.float32),
        ref.astype(np.float32),
    )

    h, w = src.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = np.array([yy + rho * flow_y, xx + rho * flow_x])

    out = np.zeros_like(cleaned_noisy, dtype=np.uint8)

    for c in range(3):
        warped = warp(
            (cleaned_noisy[..., c] > 0).astype(np.float32),
            coords,
            mode="constant",
            cval=0,
            preserve_range=True,
        )
        out[..., c] = (warped > 0.5).astype(np.uint8) * 255

    return out


def denoise_one_step_by_reference(
    reference_frame: np.ndarray,
    noisy_frame: np.ndarray,
    rho: float = 1.0,
    max_size: int = 8,
) -> np.ndarray:
    """
    Denoise one frame using a clean/reference neighboring frame.

    Parameters
    ----------
    reference_frame : ndarray, shape [H, W, 3]
        Clean frame used as spatial/anatomical reference.
    noisy_frame : ndarray, shape [H, W, 3]
        Noisy frame to denoise.
    rho : float, default 1.0
        0 = return reference masks, 1 = return anatomy-cleaned noisy frame,
        values between 0 and 1 warp the cleaned noisy masks toward the reference.
    max_size : int, default 8
        Remove connected components with area <= max_size.
    """
    rho = float(np.clip(rho, 0.0, 1.0))
    reference_frame = reference_frame.astype(np.uint8)
    noisy_frame = noisy_frame.astype(np.uint8)

    if reference_frame.ndim != 3 or reference_frame.shape[-1] != 3:
        raise ValueError("reference_frame must have shape [H, W, 3]")
    if noisy_frame.shape != reference_frame.shape:
        raise ValueError("noisy_frame must have the same shape as reference_frame")

    cleaned_noisy = _relabel_by_myo(reference_frame, noisy_frame, max_size)

    if rho == 0:
        return reference_frame.copy()
    if rho == 1:
        return cleaned_noisy
    
    return _warp_frame_toward_reference(cleaned_noisy, reference_frame, rho)


# ---------------------------------------------------------------------
# Main reusable function
# ---------------------------------------------------------------------

def denoise_sequence_with_anchor(
    noisy_seq: np.ndarray,
    clean_anchor_frame: np.ndarray,
    anchor_idx: int,
    rho: float = 1.0,
    max_size: int = 8,
) -> np.ndarray:
    """
    Denoise a full sequence from one known clean anchor frame.

    Parameters
    ----------
    noisy_seq : ndarray, shape [T, H, W, 3]
        Noisy segmentation sequence.
    clean_anchor_frame : ndarray, shape [H, W, 3]
        Known clean frame at anchor_idx.
    anchor_idx : int
        Index of the known clean anchor frame.
    rho : float, default 1.0
        0 = copy previous clean reference; 1 = use cleaned noisy target;
        intermediate values use optical-flow warping.
    max_size : int, default 8
        Remove connected components with area <= max_size.

    Returns
    -------
    clean_seq : ndarray, shape [T, H, W, 3], dtype uint8
        Denoised sequence with foreground values set to 255.
    """
    noisy_seq = noisy_seq.astype(np.uint8)
    clean_anchor_frame = clean_anchor_frame.astype(np.uint8)

    if noisy_seq.ndim != 4 or noisy_seq.shape[-1] != 3:
        raise ValueError("noisy_seq must have shape [T, H, W, 3]")

    t_count, h, w, channels = noisy_seq.shape
    if clean_anchor_frame.shape != (h, w, channels):
        raise ValueError("clean_anchor_frame must have shape [H, W, 3] matching noisy_seq")
    if not 0 <= anchor_idx < t_count:
        raise ValueError("anchor_idx is out of range")

    clean_seq = np.zeros_like(noisy_seq, dtype=np.uint8)
    clean_seq[anchor_idx] = denoise_one_step_by_reference(
        clean_anchor_frame,
        noisy_seq[anchor_idx],
        rho=rho,
        max_size=max_size,
    )

    # Forward from anchor.
    for t in range(anchor_idx + 1, t_count):
        clean_seq[t] = denoise_one_step_by_reference(
            clean_seq[t - 1],
            noisy_seq[t],
            rho=rho,
            max_size=max_size,
        )

    # Backward from anchor.
    for t in range(anchor_idx - 1, -1, -1):
        clean_seq[t] = denoise_one_step_by_reference(
            clean_seq[t + 1],
            noisy_seq[t],
            rho=rho,
            max_size=max_size,
        )

    return clean_seq
