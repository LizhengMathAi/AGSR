from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from PIL import Image

import denoise


IMAGE_DIR = Path("./acdc_train/images")
MASK_DIR = Path("./acdc_train/masks")
PSEUDO_SOFT_DIR = Path("./acdc_train/pseudo_soft_masks")
PSEUDO_HARD_DIR = Path("./acdc_train/pseudo_hard_masks")
PSEUDO_OPT_DIR = Path("./acdc_train/pseudo_opt_masks")

PATTERN = "patient001_frame*_z001.png"
# PATTERN = "patient002_frame*_z009.png"

CLASS_COLORS = {
    1: [1.0, 0.0, 0.0, 0.5],
    2: [0.0, 1.0, 0.0, 0.5],
    3: [0.0, 0.0, 1.0, 0.5],
}


def load(path, mode=None, dtype=None):
    img = Image.open(path)
    if mode:
        img = img.convert(mode)
    arr = np.array(img)
    return arr.astype(dtype) if dtype else arr


def overlay(mask, alpha=0.5):
    if mask.ndim == 2:
        out = np.zeros((*mask.shape, 4), dtype=np.float32)
        for label, color in CLASS_COLORS.items():
            out[mask == label] = color
        return out

    labels = np.argmax(mask, axis=-1)
    fg = np.any(mask > 0, axis=-1)

    rgb = np.zeros((*mask.shape[:2], 3), dtype=np.float32)
    rgb[fg, labels[fg]] = 1.0

    a = (np.max(mask, axis=-1, keepdims=True) > 0).astype(np.float32) * alpha
    return np.concatenate([rgb, a], axis=-1)


def one_hot(mask):
    return np.stack([mask == c for c in range(1, 4)], axis=-1).astype(np.float32)


def one_hot_to_label(mask):
    labels = np.argmax(mask, axis=-1).astype(np.uint8) + 1
    labels[~np.any(mask > 0, axis=-1)] = 0
    return labels


class Viewer:
    def __init__(self):
        self.image_paths = sorted(IMAGE_DIR.glob(PATTERN))
        self.mask_paths = sorted(MASK_DIR.glob(PATTERN))

        if not self.image_paths:
            raise FileNotFoundError(f"No images found for {PATTERN}")
        if not self.mask_paths:
            raise FileNotFoundError(f"No masks found for {PATTERN}")

        self.i = 0
        self.bg_visible = True

        self.params = {
            "rho": 0.50,
            "max_size": 8,
        }

        self.clean_seq = self.denoise()

    def opt_cache_dir(self):
        return (
            PSEUDO_OPT_DIR
            / f"rho_{self.params['rho']:.2f}"
            / f"max_size_{int(self.params['max_size'])}"
        )

    def opt_cache_exists(self):
        cache = self.opt_cache_dir()
        return all((cache / p.name).exists() for p in self.image_paths)

    def load_opt_cache(self):
        cache = self.opt_cache_dir()
        return np.stack([
            one_hot(load(cache / p.name))
            for p in self.image_paths
        ])

    def save_opt_cache(self, clean_seq):
        cache = self.opt_cache_dir()
        cache.mkdir(parents=True, exist_ok=True)

        for frame, path in zip(clean_seq, self.image_paths):
            Image.fromarray(one_hot_to_label(frame)).save(cache / path.name)

    def denoise(self):
        if self.opt_cache_exists():
            return self.load_opt_cache()

        soft = np.stack([
            load(PSEUDO_SOFT_DIR / p.name, dtype=np.float32)
            for p in self.image_paths
        ])

        labels = np.argmax(soft, axis=-1)
        fg = np.any(soft > 0, axis=-1)

        noisy = np.zeros_like(soft)
        noisy[fg, labels[fg]] = 1.0

        anchor = one_hot(load(self.mask_paths[0]))

        clean_seq = denoise.denoise_sequence_with_anchor(
            noisy_seq=noisy,
            clean_anchor_frame=anchor,
            anchor_idx=0,
            rho=self.params["rho"],
            max_size=int(self.params["max_size"]),
        )

        self.save_opt_cache(clean_seq)
        return clean_seq

    def sample(self):
        path = self.image_paths[self.i]

        return {
            "name": path.name,
            "image": load(path, mode="RGB"),
            "soft": overlay(load(PSEUDO_SOFT_DIR / path.name)),
            "hard": overlay(load(PSEUDO_HARD_DIR / path.name)),
            "opt": overlay(self.clean_seq[self.i]),
        }

    def title(self, key, name=None):
        if key == "ed":
            return f"[ED] {self.image_paths[0].name}"

        if key == "es":
            return f"[ES] {self.image_paths[-1].name}"

        if key != "opt":
            return f"[{key.upper()}] {name}"

        return f"[OPT] {name}"

    def show(self):
        s = self.sample()

        ed_img = load(self.image_paths[0], mode="RGB")
        es_img = load(self.image_paths[-1], mode="RGB")

        ed_mask = overlay(load(self.mask_paths[0]))
        es_mask = overlay(load(self.mask_paths[-1]))
        print(np.unique(load(self.mask_paths[-1])))

        panels = {
            "ed": (ed_img, ed_mask),
            "es": (es_img, es_mask),
            "soft": (s["image"], s["soft"]),
            "hard": (s["image"], s["hard"]),
            "opt": (s["image"], s["opt"]),
        }

        self.fig, self.axes = plt.subplots(ncols=5, figsize=(24, 8))
        plt.subplots_adjust(bottom=0.30)

        self.images = {}
        self.masks = {}

        for ax, (key, (img, mask)) in zip(self.axes, panels.items()):
            self.images[key] = ax.imshow(img)
            self.masks[key] = ax.imshow(mask)
            ax.set_title(self.title(key, s["name"]))
            ax.axis("off")

        self.add_controls()
        plt.show()

    def add_controls(self):
        self.buttons = {}

        button_specs = {
            "prev": ([0.44, 0.02, 0.15, 0.05], "Previous", lambda _: self.step(-1)),
            "next": ([0.62, 0.02, 0.15, 0.05], "Next", lambda _: self.step(1)),
            "bg": ([0.80, 0.02, 0.15, 0.05], "Hide BG", self.toggle_bg),
        }

        for name, (pos, label, callback) in button_specs.items():
            self.buttons[name] = Button(plt.axes(pos), label)
            self.buttons[name].on_clicked(callback)

        self.sliders = {}

        slider_specs = {
            "rho": ([0.12, 0.18, 0.72, 0.03], 0.0, 1.0, 0.01),
            "max_size": ([0.12, 0.13, 0.72, 0.03], 1, 30, 1),
        }

        for name, (pos, vmin, vmax, step) in slider_specs.items():
            self.sliders[name] = Slider(
                plt.axes(pos),
                name,
                vmin,
                vmax,
                valinit=self.params[name],
                valstep=step,
            )
            self.sliders[name].on_changed(self.update_params)

    def update_params(self, _):
        for name, slider in self.sliders.items():
            self.params[name] = slider.val

        self.clean_seq = self.denoise()
        self.refresh()

    def step(self, delta):
        self.i = (self.i + delta) % len(self.image_paths)
        self.refresh()

    def refresh(self):
        s = self.sample()

        for key in ["soft", "hard", "opt"]:
            self.images[key].set_data(s["image"])

        self.masks["soft"].set_data(s["soft"])
        self.masks["hard"].set_data(s["hard"])
        self.masks["opt"].set_data(s["opt"])

        self.axes[2].set_title(self.title("soft", s["name"]))
        self.axes[3].set_title(self.title("hard", s["name"]))
        self.axes[4].set_title(self.title("opt", s["name"]))

        self.fig.canvas.draw_idle()

    def toggle_bg(self, _):
        self.bg_visible = not self.bg_visible

        for img in self.images.values():
            img.set_visible(self.bg_visible)

        self.buttons["bg"].label.set_text(
            "Hide BG" if self.bg_visible else "Show BG"
        )

        self.fig.canvas.draw_idle()

    def save_tape(self, name="tape.png", rows=None, frame_ids=None, bg=True, has_mask=True, switch_rows_cols=False):
        """
        Parameters
        ----------
        name : str
            Output image path.

        frame_ids : list[int] | None
            Frames to include. Examples:
                [0, 5, 10]
                [-1]              # last frame
            If None, use all frames.
        """
        if frame_ids is None:
            frame_ids = list(range(len(self.image_paths)))

        frame_ids = [i % len(self.image_paths) for i in frame_ids]

        # rows = ["soft", "hard", "opt"]
        tape_rows = []

        for row_key in rows:
            cells = []

            for idx in frame_ids:
                path = self.image_paths[idx]

                image = load(path, mode="RGB").astype(np.float32) / 255.0

                if not has_mask:
                    cells.append((image * 255).astype(np.uint8))
                    continue

                if row_key == "img":
                    mask = overlay(load(MASK_DIR / path.name))
                elif row_key == "soft":
                    mask = overlay(load(PSEUDO_SOFT_DIR / path.name))
                elif row_key == "hard":
                    mask = overlay(load(PSEUDO_HARD_DIR / path.name))
                elif row_key == "opt":
                    mask = overlay(self.clean_seq[idx])
                else:
                    raise ValueError(row_key)

                rgb = mask[..., :3]
                alpha = mask[..., 3:4]

                if bg:
                    blended = image * (1.0 - alpha) + rgb * alpha
                else:
                    blended = rgb * alpha
                blended = (blended * 255).clip(0, 255).astype(np.uint8)

                cells.append(blended)

            tape_rows.append(cells)

        if switch_rows_cols:
            tape_rows = list(zip(*tape_rows))
        tape = np.concatenate([np.concatenate(cells, axis=1) for cells in tape_rows], axis=0)

        out_path = Path(name)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        Image.fromarray(tape).save(out_path)
        print(f"Saved tape to {out_path}")


if __name__ == "__main__":
    Viewer().show()