import os
import tkinter as tk
import os
import sys
import pytesseract
from tkinter import filedialog, messagebox
from PIL import Image

# Helper to find bundled files inside PyInstaller .exe
def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# Point pytesseract to the bundled Tesseract executable
tesseract_folder = get_resource_path("Tesseract-OCR")
pytesseract.pytesseract.tesseract_cmd = os.path.join(tesseract_folder, "tesseract.exe")

def process_and_slice():
    file_path = entry_file.get().strip()
    output_dir = entry_output.get().strip() or os.path.dirname(file_path)

    if not file_path or not os.path.isfile(file_path):
        messagebox.showerror("Error", "Please select a valid image file.")
        return

    # Check filename specifically for Dual Shock variants (case-insensitive)
    filename = os.path.basename(file_path).lower()
    keywords = ["staffds", "staff_ds"]
    
    if any(key in filename for key in keywords):
        warning_msg = (
            "This image doesn't need to be sliced because it is part of the "
            "PS1 Dual Shock version which is only made for scrolling just like general credits.\n\n"
            "Continue of slicing?"
        )
        answer = messagebox.askyesno("Warning - PS1 Dual Shock Credit FMV Notice", warning_msg)
        if not answer:
            return

    try:
        target_w = int(entry_width.get().strip())
        target_h = int(entry_height.get().strip())
        threshold = int(entry_threshold.get().strip())
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers.")
        return

    try:
        btn_process.config(state="disabled", text="Processing...")
        root.update()

        # Load original image in RGBA mode
        img = Image.open(file_path).convert("RGBA")
        img_w, img_h = img.size

        # Optional Background Removal
        if var_remove_bg.get():
            datas = img.getdata()
            new_data = []
            for item in datas:
                if item[0] < threshold and item[1] < threshold and item[2] < threshold:
                    new_data.append((0, 0, 0, 0))
                else:
                    new_data.append(item)
            img.putdata(new_data)

        # Slice top to bottom
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        count = 0

        for y in range(0, img_h, target_h):
            box = (0, y, img_w, min(y + target_h, img_h))
            chunk = img.crop(box)

            bg_color = (0, 0, 0, 0) if var_remove_bg.get() else (0, 0, 0, 255)
            frame = Image.new("RGBA", (target_w, target_h), bg_color)

            paste_x = (target_w - chunk.width) // 2
            paste_y = (target_h - chunk.height) // 2

            frame.paste(chunk, (paste_x, paste_y), chunk if var_remove_bg.get() else None)

            output_file = os.path.join(output_dir, f"{base_name}_slice_{count:03d}.png")
            frame.save(output_file, "PNG")
            count += 1

        messagebox.showinfo("Success", f"Exported {count} exact 320x240 slices successfully!")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
    finally:
        btn_process.config(state="normal", text="Slice & Export PNGs")

# GUI Interface Setup
root = tk.Tk()
root.title("Resident Evil Credit FMV Slicer")
root.geometry("480x280")
root.resizable(False, False)

# Input Image Selection
tk.Label(root, text="Select Image:").grid(row=0, column=0, padx=10, pady=10, sticky="e")
entry_file = tk.Entry(root, width=35)
entry_file.grid(row=0, column=1)
tk.Button(root, text="Browse...", command=lambda: (entry_file.delete(0, tk.END), entry_file.insert(0, filedialog.askopenfilename()))).grid(row=0, column=2, padx=5)

# Save Directory
tk.Label(root, text="Save Folder:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
entry_output = tk.Entry(root, width=35)
entry_output.grid(row=1, column=1)
tk.Button(root, text="Browse...", command=lambda: (entry_output.delete(0, tk.END), entry_output.insert(0, filedialog.askdirectory()))).grid(row=1, column=2, padx=5)

# Dimensions Frame
frame_dims = tk.Frame(root)
frame_dims.grid(row=2, column=1, pady=10, sticky="w")

tk.Label(frame_dims, text="Width:").pack(side="left")
entry_width = tk.Entry(frame_dims, width=5)
entry_width.insert(0, "320")
entry_width.pack(side="left", padx=(2, 15))

tk.Label(frame_dims, text="Height:").pack(side="left")
entry_height = tk.Entry(frame_dims, width=5)
entry_height.insert(0, "240")
entry_height.pack(side="left", padx=(2, 0))

# Options Frame
frame_opts = tk.Frame(root)
frame_opts.grid(row=3, column=1, pady=5, sticky="w")

var_remove_bg = tk.BooleanVar(value=False)
chk_bg = tk.Checkbutton(frame_opts, text="Remove Black Background", variable=var_remove_bg)
chk_bg.pack(side="left", padx=(0, 10))

tk.Label(frame_opts, text="Cutoff:").pack(side="left")
entry_threshold = tk.Entry(frame_opts, width=4)
entry_threshold.insert(0, "30")
entry_threshold.pack(side="left", padx=(2, 0))

# Process Button
btn_process = tk.Button(
    root, text="Slice & Export PNGs", command=process_and_slice,
    bg="#28a745", fg="white", font=("Arial", 11, "bold"), padx=20, pady=5
)
btn_process.grid(row=4, column=1, pady=15)

root.mainloop()