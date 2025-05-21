import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import copy

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.utils.misc import variant_to_config_mapping

DISPLAY_SIZE = 800  # or any fixed size you want

class SAM2App:
    def __init__(self, master):
        self.master = master
        self.master.title("SAM2 Point Segmentation Demo")

        # Main frame for images and mask list
        self.main_frame = tk.Frame(master)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Side-by-side images (left)
        self.figures_frame = tk.Frame(self.main_frame)
        self.figures_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.image_label = tk.Label(self.figures_frame)
        self.image_label.pack(side=tk.LEFT)
        self.mask_label = tk.Label(self.figures_frame)
        self.mask_label.pack(side=tk.LEFT)

        # Mask list/history UI (with scrollbar, at right of preview)
        self.mask_list_frame = tk.Frame(self.main_frame)
        self.mask_list_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        self.mask_listbox = tk.Listbox(self.mask_list_frame, height=15, selectmode=tk.EXTENDED, exportselection=False)
        self.mask_listbox.pack(side=tk.TOP, fill=tk.Y, expand=True)

        self.mask_scrollbar = tk.Scrollbar(self.mask_list_frame, orient=tk.VERTICAL, command=self.mask_listbox.yview)
        self.mask_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.mask_listbox.config(yscrollcommand=self.mask_scrollbar.set)

        self.mask_listbox.bind("<<ListboxSelect>>", self.on_mask_select)
        self.new_mask_button = tk.Button(self.mask_list_frame, text="New Mask", command=self.new_mask)
        self.new_mask_button.pack(side=tk.TOP, fill=tk.X, pady=(10, 0))
        self.delete_mask_button = tk.Button(self.mask_list_frame, text="Delete Mask", command=self.delete_mask)
        self.delete_mask_button.pack(side=tk.TOP, fill=tk.X)
        self.rename_mask_button = tk.Button(self.mask_list_frame, text="Rename Mask", command=self.rename_mask)
        self.rename_mask_button.pack(side=tk.TOP, fill=tk.X)
        self.save_mask_button = tk.Button(self.mask_list_frame, text="Save Mask to File", command=self.save_mask_to_file)
        self.save_mask_button.pack(side=tk.TOP, fill=tk.X)

        # Bottom frame for main action buttons
        self.bottom_frame = tk.Frame(master)
        self.bottom_frame.pack(side=tk.BOTTOM, pady=10)

        self.load_button = tk.Button(self.bottom_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=10)
        self.clear_button = tk.Button(self.bottom_frame, text="Clear Points", command=self.clear_points)
        self.clear_button.pack(side=tk.LEFT, padx=10)
        self.remove_button = tk.Button(self.bottom_frame, text="Remove Last Point", command=self.remove_last_point)
        self.remove_button.pack(side=tk.LEFT, padx=10)
        

        self.points = []  # Each entry: (x, y, label)
        self.image = None
        self.display_image = None
        self.cursor_pos = None
        self.viewport_center = None

        # Mask history: (label, mask_array, points_list)
        self.saved_masks = []
        self.selected_mask_index = None  # None means show combined
        self.virtual_union_mask = None  # Add to __init__

        # Build model
        self.model = build_sam2(
            variant_to_config_mapping["tiny"],
            "sam2_hiera_tiny.pt",
            device="cpu"
        )
        self.image_predictor = None

        self.zoom_factor = 1.0
        self.image_label.bind("<MouseWheel>", self.on_zoom)  # Windows
        self.image_label.bind("<Button-4>", self.on_zoom)    # Linux scroll up
        self.image_label.bind("<Button-5>", self.on_zoom)    # Linux scroll down

        # Keyboard bindings for panning
        self.master.bind("<Key-w>", self.pan_up)
        self.master.bind("<Key-s>", self.pan_down)
        self.master.bind("<Key-a>", self.pan_left)
        self.master.bind("<Key-d>", self.pan_right)

        self._img_disp_cache = None
        self._mask_disp_cache = None
        self._cache_box = None

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = Image.open(file_path).convert("RGB")
            self.display_image = self.image.copy()
            self.points = []
            self.cursor_pos = None
            self.viewport_center = None
            self.saved_masks = []
            self.selected_mask_index = None
            self.image_predictor = SAM2ImagePredictor(self.model)
            self.image_predictor.set_image(np.array(self.image))
            self.update_mask_listbox()
            self.show_image()
            self.new_mask()  # Create and activate the first mask
            self.image_label.bind("<Button-1>", self.on_click)
            self.image_label.bind("<Motion>", self.on_motion)

    def clear_points(self):
        self.points = []
        self.cursor_pos = None
        self.viewport_center = None
        self.display_image = self.image.copy()
        self.update_current_mask_in_list()  # <-- Add this line
        self.show_image()
        self.run_sam2()

    def remove_last_point(self):
        if self.points:
            self.points.pop()
            self.display_image = self.image.copy()
            self.selected_mask_index = None
            self.run_sam2()

    def get_viewport(self):
        w, h = self.image.size
        zoom = self.zoom_factor
        # Use viewport_center for pan/zoom, default to center of image
        if self.viewport_center is not None:
            cx, cy = self.viewport_center
        else:
            cx, cy = w // 2, h // 2
        vw, vh = int(w / zoom), int(h / zoom)
        left = max(0, min(cx - vw // 2, w - vw))
        upper = max(0, min(cy - vh // 2, h - vh))
        right = left + vw
        lower = upper + vh
        return (left, upper, right, lower)

    def show_image(self, update_cache=True):
        if self.image is None:
            return
        box = self.get_viewport()
        # Only update the cache if needed
        if update_cache or self._img_disp_cache is None or self._cache_box != box:
            img_crop = self.image.crop(box)
            mask_img_crop = (self.display_image.crop(box)
                             if self.display_image is not None
                             else Image.new("RGB", img_crop.size, (128,128,128)))
            # Draw points (only those visible in viewport)
            draw = ImageDraw.Draw(img_crop)
            for pt in self.points:
                if box[0] <= pt[0] < box[2] and box[1] <= pt[1] < box[3]:
                    r = 10
                    color = "blue" if pt[2] == 1 else "red"
                    draw.ellipse((pt[0]-box[0]-r, pt[1]-box[1]-r, pt[0]-box[0]+r, pt[1]-box[1]+r), fill=color, outline="white", width=5)
            img_disp = img_crop.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS)
            mask_disp = mask_img_crop.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS)
            self._img_disp_cache = img_disp
            self._mask_disp_cache = mask_disp
            self._cache_box = box
        else:
            img_disp = self._img_disp_cache
            mask_disp = self._mask_disp_cache

        # Draw crosshair on a copy of the cached mask image
        mask_disp_with_cross = mask_disp.copy()
        if self.cursor_pos is not None:
            x, y = self.cursor_pos
            if box[0] <= x < box[2] and box[1] <= y < box[3]:
                draw_mask = ImageDraw.Draw(mask_disp_with_cross)
                x_vp = int((x - box[0]) * DISPLAY_SIZE / (box[2] - box[0]))
                y_vp = int((y - box[1]) * DISPLAY_SIZE / (box[3] - box[1]))
                cross_len = 25
                draw_mask.line((x_vp-cross_len, y_vp, x_vp+cross_len, y_vp), fill="white", width=9)
                draw_mask.line((x_vp, y_vp-cross_len, x_vp, y_vp+cross_len), fill="white", width=9)
                draw_mask.line((x_vp-cross_len, y_vp, x_vp+cross_len, y_vp), fill="black", width=5)
                draw_mask.line((x_vp, y_vp-cross_len, x_vp, y_vp+cross_len), fill="black", width=5)

        for pil_img, label in [(img_disp, self.image_label), (mask_disp_with_cross, self.mask_label)]:
            tk_img = ImageTk.PhotoImage(pil_img)
            label.config(image=tk_img)
            label.image = tk_img

    def on_zoom(self, event):
        # Zoom in/out with mouse wheel, keep zoom centered on cursor
        if hasattr(event, 'delta'):
            if event.delta > 0:
                self.zoom_factor = min(self.zoom_factor * 1.1, 10)
            else:
                self.zoom_factor = max(self.zoom_factor / 1.1, 1.0)
        elif hasattr(event, 'num'):
            if event.num == 4:
                self.zoom_factor = min(self.zoom_factor * 1.1, 10)
            elif event.num == 5:
                self.zoom_factor = max(self.zoom_factor / 1.1, 1.0)
        self.show_image()

    def on_click(self, event):
        # Map display coordinates to image coordinates in viewport
        box = self.get_viewport()
        x_img = int(box[0] + event.x * (box[2] - box[0]) / DISPLAY_SIZE)
        y_img = int(box[1] + event.y * (box[3] - box[1]) / DISPLAY_SIZE)
        label = 0 if (event.state & 0x0004) else 1
        self.points.append((x_img, y_img, label))
        self.update_current_mask_in_list()
        self.run_sam2()
        self.show_image()
        # Do NOT change self.selected_mask_index here!

    def on_motion(self, event):
        x_disp, y_disp = event.x, event.y
        box = self.get_viewport()
        x_img = int(box[0] + x_disp * (box[2] - box[0]) / DISPLAY_SIZE)
        y_img = int(box[1] + y_disp * (box[3] - box[1]) / DISPLAY_SIZE)
        self.cursor_pos = (x_img, y_img)
        self.show_image(update_cache=False)

    def run_sam2(self):
        if not self.points or self.image is None or self.image_predictor is None:
            self.display_image = None
            self.combined_mask = None
            self.show_image()
            return

        coords = np.array([[pt[0], pt[1]] for pt in self.points])
        labels = np.array([pt[2] for pt in self.points])
        masks, scores, _ = self.image_predictor.predict(
            point_coords=coords,
            point_labels=labels,
            box=None,
            multimask_output=False,
        )
        self.combined_mask = masks[0]
        # Overlay white mask with alpha on the image
        base = self.image.convert("RGBA")
        mask_img = Image.fromarray((self.combined_mask * 255).astype(np.uint8))
        white_overlay = Image.new("RGBA", base.size, (255, 255, 255, 180))
        base.paste(white_overlay, mask=mask_img)
        self.display_image = base.convert("RGB")
        # If "Combined" is selected, update display
        if self.selected_mask_index is None or self.selected_mask_index == len(self.saved_masks):
            self.show_image()
        self.update_mask_listbox()

    def update_mask_listbox(self):
        self.mask_listbox.delete(0, tk.END)
        for label, *_ in self.saved_masks:
            self.mask_listbox.insert(tk.END, label)
        self.mask_listbox.insert(tk.END, "Combined")
        # Only force selection if nothing is selected
        if not self.mask_listbox.curselection():
            self.mask_listbox.selection_set(self.selected_mask_index or 0)

    def on_mask_select(self, event):
        selection = self.mask_listbox.curselection()
        if not selection:
            return

        # Save current mask before switching
        self.update_current_mask_in_list()

        # If multiple masks are selected (and not "Combined"), show their union but do NOT create a new mask
        if len(selection) > 1 and all(idx < len(self.saved_masks) for idx in selection):
            masks = [self.saved_masks[idx][1] for idx in selection]
            union_mask = np.logical_or.reduce(masks)
            self.virtual_union_mask = union_mask  # Store for saving
            base = self.image.convert("RGBA")
            mask_img = Image.fromarray((union_mask * 255).astype(np.uint8))
            white_overlay = Image.new("RGBA", base.size, (255, 255, 255, 180))
            base.paste(white_overlay, mask=mask_img)
            self.display_image = base.convert("RGB")
            self.show_image()
            return

        self.virtual_union_mask = None  # Reset if not multi-select

        # Otherwise, behave as before
        idx = selection[0]
        self.selected_mask_index = idx
        if idx < len(self.saved_masks):
            mask = self.saved_masks[idx][1]
            points = self.saved_masks[idx][2]
            self.points = [tuple(pt) for pt in points]  # Restore points for this mask
            self.combined_mask = mask.copy()            # Restore mask for this mask
            base = self.image.convert("RGBA")
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            white_overlay = Image.new("RGBA", base.size, (255, 255, 255, 180))
            base.paste(white_overlay, mask=mask_img)
            self.display_image = base.convert("RGB")
            self.show_image()
        else:
            # Show the union of all saved masks
            if self.saved_masks:
                combined = np.zeros_like(self.saved_masks[0][1], dtype=bool)
                for _, mask, _ in self.saved_masks:
                    combined = np.logical_or(combined, mask)
                base = self.image.convert("RGBA")
                mask_img = Image.fromarray((combined * 255).astype(np.uint8))
                white_overlay = Image.new("RGBA", base.size, (255, 255, 255, 180))
                base.paste(white_overlay, mask=mask_img)
                self.display_image = base.convert("RGB")
                self.show_image()
            else:
                self.display_image = self.image.copy()
                self.show_image()

    def delete_mask(self):
        selection = self.mask_listbox.curselection()
        if not selection or selection[0] >= len(self.saved_masks):
            return
        idx = selection[0]
        del self.saved_masks[idx]
        self.selected_mask_index = None
        self.points = []
        self.update_mask_listbox()
        self.run_sam2()

    def rename_mask(self):
        selection = self.mask_listbox.curselection()
        if not selection or selection[0] >= len(self.saved_masks):
            return
        idx = selection[0]
        # Prompt for new name
        new_name = tk.simpledialog.askstring("Rename Mask", "Enter new mask name:", initialvalue=self.saved_masks[idx][0])
        if new_name and new_name.strip():
            label, mask, points = self.saved_masks[idx]
            self.saved_masks[idx] = (new_name.strip(), mask, points)
            self.update_mask_listbox()
            self.mask_listbox.selection_clear(0, tk.END)
            self.mask_listbox.selection_set(idx)

    def pan_up(self, event=None):
        if self.viewport_center is None:
            self.viewport_center = (self.image.width // 2, self.image.height // 2)
        cx, cy = self.viewport_center
        pan_step = 40
        cy = max(cy - pan_step, 0)
        self.viewport_center = (cx, cy)
        self.show_image(update_cache=False)

    def pan_down(self, event=None):
        if self.viewport_center is None:
            self.viewport_center = (self.image.width // 2, self.image.height // 2)
        cx, cy = self.viewport_center
        pan_step = 40
        cy = min(cy + pan_step, self.image.height)
        self.viewport_center = (cx, cy)
        self.show_image(update_cache=False)

    def pan_left(self, event=None):
        if self.viewport_center is None:
            self.viewport_center = (self.image.width // 2, self.image.height // 2)
        cx, cy = self.viewport_center
        pan_step = 40
        cx = max(cx - pan_step, 0)
        self.viewport_center = (cx, cy)
        self.show_image(update_cache=False)

    def pan_right(self, event=None):
        if self.viewport_center is None:
            self.viewport_center = (self.image.width // 2, self.image.height // 2)
        cx, cy = self.viewport_center
        pan_step = 40
        cx = min(cx + pan_step, self.image.width)
        self.viewport_center = (cx, cy)
        self.show_image(update_cache=False)

    def save_mask_to_file(self):
        mask_to_save = None
        # Save the union mask if present
        if self.virtual_union_mask is not None:
            mask_to_save = self.virtual_union_mask
        # Save the selected mask if it's a real mask
        elif (
            self.selected_mask_index is not None
            and self.selected_mask_index < len(self.saved_masks)
        ):
            mask_to_save = self.saved_masks[self.selected_mask_index][1]
        # Save the combined mask if "Combined" is selected
        elif (
            self.selected_mask_index is not None
            and self.selected_mask_index == self.mask_listbox.size() - 1  # Fix: check if "Combined" is selected
            and hasattr(self, "combined_mask")
            and self.combined_mask is not None
        ):
            mask_to_save = self.combined_mask

        if mask_to_save is not None:
            mask_img = Image.fromarray((mask_to_save * 255).astype(np.uint8))
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("Bitmap files", "*.bmp"), ("All files", "*.*")]
            )
            if file_path:
                if file_path.lower().endswith(".jpg") or file_path.lower().endswith(".jpeg"):
                    mask_img = mask_img.convert("L")
                mask_img.save(file_path)

    def new_mask(self):
        # Save the current mask before switching
        self.update_current_mask_in_list()
        # Create a new empty mask and activate it
        label = f"Mask {len(self.saved_masks)+1}"
        empty_mask = np.zeros(self.image.size[::-1], dtype=bool)
        self.points = []
        self.saved_masks.append((label, empty_mask, []))
        self.selected_mask_index = len(self.saved_masks) - 1
        self.update_mask_listbox()
        self.mask_listbox.selection_clear(0, tk.END)
        self.mask_listbox.selection_set(self.selected_mask_index)
        self.display_image = self.image.copy()
        self.combined_mask = empty_mask.copy()
        self.show_image()

    def update_current_mask_in_list(self):
        # Only update if a valid mask is selected and not in virtual union mode
        if (
            self.selected_mask_index is not None
            and self.selected_mask_index < len(self.saved_masks)
            and hasattr(self, "combined_mask")
            and self.combined_mask is not None
        ):
            label, _, _ = self.saved_masks[self.selected_mask_index]
            points_copy = [tuple(pt) for pt in self.points]
            self.saved_masks[self.selected_mask_index] = (label, self.combined_mask.copy(), points_copy)

if __name__ == "__main__":
    root = tk.Tk()
    app = SAM2App(root)
    root.mainloop()