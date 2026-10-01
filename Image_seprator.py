"""
Resident Evil (PS1) Credit FMV Slicer  -  now with native .STF support.

Opens either a normal image (PNG/BMP/...) or a Resident Evil PSX credit
".STF" file (staff.stf, STAFF2.STF, STAFFDS.STF) and exports it as PNGs.

Slicing modes
  frames   : fixed-height frames (default 320x240, image centred) - the original
             behaviour. Perfect for STAFF.STF (20 pages of 256x240).
  sections : auto-split at blank gaps between credit blocks. Meant for the
             scrolling strips (STAFF2.STF / STAFFDS.STF) where fixed-height
             slicing would cut through lines of text.
  full     : just convert the whole image to a single PNG.

Run with no arguments for the GUI, or use the command line:
  python Image_seprator.py STAFF.STF --mode frames  -o out
  python Image_seprator.py STAFF2.STF --mode sections --min-gap 24 -o out
  python Image_seprator.py STAFFDS.STF --mode full  -o out

STF format (reverse-engineered from CUE's stf.exe)
  0x000-0x1FF : 256-entry palette, 16-bit little-endian PSX colour
                (bits 0-4 R, 5-9 G, 10-14 B, bit 15 STP)
  0x200-EOF   : consecutive blocks, each one 256x16 pixel 8-bit strip, RLE coded:
                  u16 block_size  - distance to next block
                  u16 runlen_off  - offset (from block start) to u16 run lengths, ends with 0
                  u16 value_off   - offset (from block start) to one palette index per run
"""
import os
import sys
import struct
import argparse
from PIL import Image, ImageChops

# Optional OCR: point pytesseract at the Tesseract-OCR folder bundled next to the
# script / inside the PyInstaller .exe (same setup as the original tool).
def get_resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = os.path.join(
        get_resource_path("Tesseract-OCR"), "tesseract.exe")
except ImportError:
    pytesseract = None  # OCR unavailable; slicing does not need it

STF_WIDTH = 256
STF_STRIP_H = 16
SCROLL_NAMES = ("staff2", "staffds", "staff_ds")


# --------------------------------------------------------------------------
# STF decoding
# --------------------------------------------------------------------------
def decode_stf(path):
    """Decode a Resident Evil PSX .STF file into an RGBA PIL image."""
    with open(path, "rb") as fh:
        data = fh.read()

    if len(data) < 0x200 + 6:
        raise ValueError("File is too small to be a valid STF.")

    # Palette: 256 x BGR555 (PSX order: bits 0-4 = R)
    palette = []
    for i in range(256):
        c = struct.unpack_from("<H", data, i * 2)[0]
        r, g, b = c & 0x1F, (c >> 5) & 0x1F, (c >> 10) & 0x1F
        palette.extend(((r << 3) | (r >> 2), (g << 3) | (g >> 2), (b << 3) | (b >> 2)))

    pos = 0x200
    strips = []
    n = len(data)
    while pos + 6 <= n:
        size, run_off, val_off = struct.unpack_from("<HHH", data, pos)
        if size == 0:
            break
        if size < 6 or pos + size > n or pos + run_off >= n or pos + val_off >= n:
            raise ValueError(f"Corrupt STF block at offset 0x{pos:X}.")

        strip = bytearray()
        r = pos + run_off
        v = pos + val_off
        while True:
            if r + 2 > n:
                raise ValueError(f"Corrupt STF run data at offset 0x{pos:X}.")
            count = struct.unpack_from("<H", data, r)[0]
            if count == 0:
                break
            if v >= n:
                raise ValueError(f"Corrupt STF run data at offset 0x{pos:X}.")
            r += 2
            strip += bytes((data[v],)) * count
            v += 1

        # stf.exe caps a strip at 4096 pixels (256x16); pad if it comes up short.
        want = STF_WIDTH * STF_STRIP_H
        strip = strip[:want]
        if len(strip) < want:
            strip += bytes(want - len(strip))
        strips.append(bytes(strip))
        pos += size

    if not strips:
        raise ValueError("No image data found - is this really a Resident Evil STF?")

    img = Image.frombytes("P", (STF_WIDTH, len(strips) * STF_STRIP_H), b"".join(strips))
    img.putpalette(palette)
    return img.convert("RGBA")


def load_any_image(path):
    """Load an STF (by extension) or any normal image as RGBA."""
    if path.lower().endswith(".stf"):
        return decode_stf(path)
    return Image.open(path).convert("RGBA")


# --------------------------------------------------------------------------
# Image helpers
# --------------------------------------------------------------------------
def remove_black_background(img, threshold):
    """Make pixels with R,G,B all below `threshold` fully transparent."""
    out = []
    getter = getattr(img, "get_flattened_data", img.getdata)  # getdata is deprecated in new Pillow
    for px in getter():
        if px[0] < threshold and px[1] < threshold and px[2] < threshold:
            out.append((0, 0, 0, 0))
        else:
            out.append(px)
    img = img.copy()
    img.putdata(out)
    return img


def soft_black_matte(img, vmax=None, floor=0):
    """Distortion-free 'black -> transparent'.

    Brightness becomes opacity and the colour is un-multiplied, so the result
    composited over black is identical to the original image, while over video
    the soft glyph edges blend naturally (no hard cut, no dark fringes).
    `vmax` = brightness that should be fully opaque (the text's brightest shade);
    pass the same value for every slice so they all match.
    """
    import numpy as np
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    m = arr.max(axis=2)
    if vmax is None:
        vmax = max(float(m.max()), 1.0)
    alpha = np.clip(m / vmax, 0.0, 1.0)
    alpha[m <= floor] = 0.0
    safe = np.where(alpha > 0, alpha, 1.0)[..., None]
    color = np.clip(arr / safe, 0, 255)
    color[alpha == 0] = 0
    out = np.dstack([color, (alpha * 255.0)[..., None]])
    return Image.fromarray(np.rint(out).astype(np.uint8), "RGBA")


RESAMPLE = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
}


def upscale(img, scale, method):
    """Upscale on an opaque RGB image (over black) so no transparent-edge halos appear."""
    if scale <= 1:
        return img
    return img.convert("RGB").resize((img.width * scale, img.height * scale), RESAMPLE[method])


def row_has_content(img, threshold):
    """List of bools, one per image row: True if the row contains any
    pixel brighter than `threshold` (and not fully transparent)."""
    r, g, b, a = img.split()
    bright = ImageChops.lighter(ImageChops.lighter(r, g), b)
    mask = bright.point(lambda v: 255 if v >= threshold else 0)
    mask = ImageChops.multiply(mask, a.point(lambda v: 255 if v else 0))
    _, yproj = mask.getprojection()
    return [bool(v) for v in yproj]


def find_sections(content_rows, min_gap, pad):
    """Return [(top, bottom), ...] spans of content separated by >= min_gap blank rows."""
    h = len(content_rows)
    spans, y = [], 0
    while y < h:
        if not content_rows[y]:
            y += 1
            continue
        top = y
        last = y
        while y < h:
            if content_rows[y]:
                last = y
                y += 1
            elif y - last >= min_gap:
                break
            else:
                y += 1
        spans.append((max(0, top - pad), min(h, last + 1 + pad)))
    return spans


# --------------------------------------------------------------------------
# Core slicing
# --------------------------------------------------------------------------
def slice_file(file_path, output_dir=None, mode="frames", target_w=320, target_h=240,
               remove_bg=False, threshold=30, min_gap=24, pad=4, skip_blank=False,
               save_full=False, log=print, soft=True, scale=1, resample="nearest"):
    """Slice `file_path` and return the list of files written.

    remove_bg : make the black background transparent.
    soft      : use the soft (alpha-from-brightness) matte instead of the old hard cutoff.
    scale     : integer upscale factor (applied BEFORE the matte). Frame size
                (target_w x target_h) is the FINAL size; frames take target_h//scale source rows.
    """
    output_dir = output_dir or os.path.dirname(os.path.abspath(file_path))
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(file_path))[0]
    scale = max(1, int(scale))

    img = load_any_image(file_path)
    img_w, img_h = img.size
    log(f"Loaded {os.path.basename(file_path)}: {img_w}x{img_h}")

    # Emptiness is measured on the untouched source image.
    content = row_has_content(img, threshold) if (mode == "sections" or skip_blank) else None

    # One shared 'fully opaque' brightness so every slice gets identical colours.
    vmax = max(float(max(ImageChops.lighter(ImageChops.lighter(*img.split()[:2]),
                                            img.split()[2]).getextrema())), 1.0)

    def finish(chunk):
        """upscale (on black) -> background removal."""
        chunk = upscale(chunk, scale, resample)
        if not remove_bg:
            return chunk.convert("RGBA")
        return soft_black_matte(chunk, vmax, floor=threshold) if soft \
            else remove_black_background(chunk.convert("RGBA"), threshold)

    written = []

    def save(image, name):
        p = os.path.join(output_dir, name)
        image.save(p, "PNG")
        written.append(p)

    if save_full or mode == "full":
        save(finish(img), f"{base}_full.png")

    if mode == "frames":
        bg_color = (0, 0, 0, 0) if remove_bg else (0, 0, 0, 255)
        step = max(1, target_h // scale)
        count = 0
        for y in range(0, img_h, step):
            y2 = min(y + step, img_h)
            if skip_blank and not any(content[y:y2]):
                continue
            chunk = finish(img.crop((0, y, img_w, y2)))
            frame = Image.new("RGBA", (target_w, target_h), bg_color)
            # paste with no mask: the chunk already carries its own alpha
            frame.paste(chunk, ((target_w - chunk.width) // 2, (target_h - chunk.height) // 2))
            save(frame, f"{base}_slice_{count:03d}.png")
            count += 1
        log(f"Exported {count} frame(s) of {target_w}x{target_h}.")

    elif mode == "sections":
        spans = find_sections(content, min_gap, pad)
        for i, (top, bottom) in enumerate(spans):
            save(finish(img.crop((0, top, img_w, bottom))), f"{base}_section_{i:03d}.png")
        log(f"Exported {len(spans)} section(s) (min gap {min_gap} rows).")

    elif mode != "full":
        raise ValueError(f"Unknown mode: {mode}")

    return written


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------
def run_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.title("Resident Evil Credit FMV Slicer")
    root.geometry("540x450")
    root.resizable(False, False)

    mode_var = tk.StringVar(value="frames")
    var_remove_bg = tk.BooleanVar(value=False)
    var_skip_blank = tk.BooleanVar(value=False)
    var_save_full = tk.BooleanVar(value=False)
    var_soft = tk.BooleanVar(value=True)
    var_resample = tk.StringVar(value="nearest")

    def browse_file():
        p = filedialog.askopenfilename(filetypes=[
            ("STF / images", "*.stf *.STF *.png *.bmp *.jpg *.jpeg *.tga *.gif"),
            ("Resident Evil STF", "*.stf *.STF"),
            ("All files", "*.*")])
        if not p:
            return
        entry_file.delete(0, tk.END)
        entry_file.insert(0, p)
        # Sensible default per file: scrolling credits -> sections, normal -> frames
        name = os.path.basename(p).lower()
        mode_var.set("sections" if any(k in name for k in SCROLL_NAMES) else "frames")

    def browse_out():
        p = filedialog.askdirectory()
        if p:
            entry_output.delete(0, tk.END)
            entry_output.insert(0, p)

    def process():
        file_path = entry_file.get().strip()
        if not file_path or not os.path.isfile(file_path):
            messagebox.showerror("Error", "Please select a valid STF or image file.")
            return
        output_dir = entry_output.get().strip() or os.path.dirname(file_path)

        mode = mode_var.get()
        name = os.path.basename(file_path).lower()
        if mode == "frames" and any(k in name for k in SCROLL_NAMES):
            ok = messagebox.askyesno(
                "Scrolling credits notice",
                "STAFF2 / STAFFDS are continuous scrolling credit strips (PS1 Dual Shock "
                "version style).\n\nFixed-height slicing will cut through lines of text. "
                "'Auto-split sections' or 'Full image' is usually what you want.\n\n"
                "Continue with fixed-height frames anyway?")
            if not ok:
                return
        try:
            target_w = int(entry_width.get().strip())
            target_h = int(entry_height.get().strip())
            threshold = int(entry_threshold.get().strip())
            min_gap = int(entry_gap.get().strip())
            scale = int(entry_scale.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers.")
            return

        try:
            btn_process.config(state="disabled", text="Processing...")
            root.update()
            files = slice_file(file_path, output_dir, mode, target_w, target_h,
                               var_remove_bg.get(), threshold, min_gap, 4,
                               var_skip_blank.get(), var_save_full.get(), log=lambda s: None,
                               soft=var_soft.get(), scale=scale, resample=var_resample.get())
            messagebox.showinfo("Success", f"Exported {len(files)} PNG file(s) to:\n{output_dir}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{e}")
        finally:
            btn_process.config(state="normal", text="Extract & Export PNGs")

    tk.Label(root, text="Select STF / Image:").grid(row=0, column=0, padx=10, pady=10, sticky="e")
    entry_file = tk.Entry(root, width=34)
    entry_file.grid(row=0, column=1, columnspan=2, sticky="w")
    tk.Button(root, text="Browse...", command=browse_file).grid(row=0, column=3, padx=5)

    tk.Label(root, text="Save Folder:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
    entry_output = tk.Entry(root, width=34)
    entry_output.grid(row=1, column=1, columnspan=2, sticky="w")
    tk.Button(root, text="Browse...", command=browse_out).grid(row=1, column=3, padx=5)

    # Mode
    frame_mode = tk.LabelFrame(root, text="Slicing mode")
    frame_mode.grid(row=2, column=0, columnspan=4, padx=12, pady=8, sticky="we")
    tk.Radiobutton(frame_mode, text="Fixed frames (e.g. STAFF.STF)", variable=mode_var,
                   value="frames").pack(anchor="w")
    tk.Radiobutton(frame_mode, text="Auto-split credit sections (scrolling: STAFF2 / STAFFDS)",
                   variable=mode_var, value="sections").pack(anchor="w")
    tk.Radiobutton(frame_mode, text="Full image only (convert to one PNG)", variable=mode_var,
                   value="full").pack(anchor="w")

    # Dimensions
    frame_dims = tk.Frame(root)
    frame_dims.grid(row=3, column=1, columnspan=3, pady=4, sticky="w")
    tk.Label(frame_dims, text="Frame W:").pack(side="left")
    entry_width = tk.Entry(frame_dims, width=5)
    entry_width.insert(0, "320")
    entry_width.pack(side="left", padx=(2, 12))
    tk.Label(frame_dims, text="H:").pack(side="left")
    entry_height = tk.Entry(frame_dims, width=5)
    entry_height.insert(0, "240")
    entry_height.pack(side="left", padx=(2, 12))
    tk.Label(frame_dims, text="Min gap (rows):").pack(side="left")
    entry_gap = tk.Entry(frame_dims, width=4)
    entry_gap.insert(0, "24")
    entry_gap.pack(side="left", padx=(2, 0))

    # Options
    frame_opts = tk.Frame(root)
    frame_opts.grid(row=4, column=1, columnspan=3, pady=4, sticky="w")
    tk.Checkbutton(frame_opts, text="Remove Black Background", variable=var_remove_bg).pack(side="left", padx=(0, 6))
    tk.Label(frame_opts, text="Cutoff:").pack(side="left")
    entry_threshold = tk.Entry(frame_opts, width=4)
    entry_threshold.insert(0, "30")
    entry_threshold.pack(side="left", padx=(2, 0))

    frame_opts2 = tk.Frame(root)
    frame_opts2.grid(row=5, column=1, columnspan=3, pady=2, sticky="w")
    tk.Checkbutton(frame_opts2, text="Skip blank frames", variable=var_skip_blank).pack(side="left", padx=(0, 10))
    tk.Checkbutton(frame_opts2, text="Also save full PNG", variable=var_save_full).pack(side="left")

    frame_opts3 = tk.Frame(root)
    frame_opts3.grid(row=6, column=1, columnspan=3, pady=2, sticky="w")
    tk.Checkbutton(frame_opts3, text="Soft edges (no distortion)", variable=var_soft).pack(side="left", padx=(0, 10))
    tk.Label(frame_opts3, text="Upscale x").pack(side="left")
    entry_scale = tk.Entry(frame_opts3, width=3)
    entry_scale.insert(0, "1")
    entry_scale.pack(side="left", padx=(2, 6))
    tk.OptionMenu(frame_opts3, var_resample, *RESAMPLE.keys()).pack(side="left")

    btn_process = tk.Button(root, text="Extract & Export PNGs", command=process,
                            bg="#28a745", fg="white", font=("Arial", 11, "bold"), padx=20, pady=5)
    btn_process.grid(row=7, column=1, columnspan=2, pady=14)

    root.mainloop()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        run_gui()
        return 0

    ap = argparse.ArgumentParser(description="Resident Evil PSX credit STF / image slicer")
    ap.add_argument("input", help=".STF file or image")
    ap.add_argument("-o", "--out", help="output folder (default: next to input)")
    ap.add_argument("--mode", choices=["frames", "sections", "full"], default=None,
                    help="default: 'sections' for STAFF2/STAFFDS, otherwise 'frames'")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--remove-bg", action="store_true", help="make black background transparent")
    ap.add_argument("--cutoff", type=int, default=30, help="black cutoff (default 30)")
    ap.add_argument("--min-gap", type=int, default=24, help="blank rows that separate sections")
    ap.add_argument("--pad", type=int, default=4, help="blank rows kept around each section")
    ap.add_argument("--skip-blank", action="store_true", help="skip empty frames")
    ap.add_argument("--save-full", action="store_true", help="also write the whole image as PNG")
    ap.add_argument("--hard", action="store_true", help="old hard-cutoff background removal (not recommended)")
    ap.add_argument("--scale", type=int, default=1, help="integer upscale factor, applied before the matte")
    ap.add_argument("--resample", choices=list(RESAMPLE), default="nearest")
    a = ap.parse_args(argv)

    mode = a.mode
    if mode is None:
        name = os.path.basename(a.input).lower()
        mode = "sections" if any(k in name for k in SCROLL_NAMES) else "frames"

    files = slice_file(a.input, a.out, mode, a.width, a.height, a.remove_bg, a.cutoff,
                       a.min_gap, a.pad, a.skip_blank, a.save_full,
                       soft=not a.hard, scale=a.scale, resample=a.resample)
    print(f"Done - {len(files)} file(s) written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
