import gradio as gr
import os
from pathlib import Path
import numpy as np
import sys
import subprocess
import torch
from sam2.build_sam import build_sam2_video_predictor
import matplotlib.pyplot as plt
from functools import partial
from PIL import Image, ImageDraw
from torch.cuda.amp import autocast
from enum import Enum
import gc
from collections import Counter


CONFIG = {
    "predictor": None,
    "inference_state": None,
    "annotated_frame_dir": Path("autolabeller/Annotated Frames"),
}


class PointState(Enum):
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    OFF = "Off"


def gc_collect():
    gc.collect()
    print("Garbage collection complete.")

    # Clear PyTorch caches (if using CUDA)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Cleared PyTorch CUDA cache.")


def seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():  # Apple MPS support
        torch.mps.manual_seed(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Seed set to {seed}")


def setup_device():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
    return device


def load_sam(
    model_cfg="sam2/configs/sam2.1/sam2.1_hiera_b+.yaml",
    sam2_checkpoint="checkpoints/sam2.1_hiera_base_plus.pt",
):
    device = CONFIG["device"]
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    sam2_checkpoint = "checkpoints/sam2.1_hiera_base_plus.pt"
    try:
        predictor = build_sam2_video_predictor(
            model_cfg, sam2_checkpoint, device=device
        )
        print("SAMv2 Predictor loaded successfully.")
        return predictor
    except Exception as e:
        print(f"Failed to load SAMv2 Predictor: {e}")


def extract_frames(videos_list, video_paths_state):
    frame_dir = CONFIG.get("frame_dir")

    for video in videos_list:
        print(f"Extracting frames from {video}")

        video_path = video_paths_state[video]

        # Create a directory for each video's frames
        output_dir = (
            frame_dir / video.split(".")[0]
        )  # Use stem to get the video name without the extension
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define the output pattern for the extracted frames
        output_pattern = "%05d.jpg"

        # Build the FFmpeg command
        command = [
            "ffmpeg",
            "-i",
            str(video_path.resolve()),  # Ensure the full path is passed
            "-q:v",
            "2",  # Set quality
            "-start_number",
            "0",  # Start numbering frames from 0
            f"{output_dir}/{output_pattern}",  # Output frame file path
        ]

        # Run the FFmpeg command
        subprocess.run(command)

    return f"Extraction Complete"


def extract_all_frames(video_paths_state):
    if not video_paths_state:
        return "No videos loaded"
    return extract_frames(list(video_paths_state.keys()), video_paths_state)


def load_videos(dir_str):
    dir_path = Path(dir_str)
    video_data = {}
    if dir_path.exists() and dir_path.is_dir():
        videos_list = list(dir_path.glob("*.mp4"))
        if videos_list:
            for video_path in videos_list:
                video_data[video_path.name] = video_path
    return video_data, list(video_data.keys())


def update_dropdown(dir_path):
    video_dict, video_names = load_videos(dir_path)
    return (
        gr.Dropdown(choices=video_names, multiselect=True, interactive=True),
        video_dict,
    )


def extract_frame_tab():
    gr.Markdown("## Frame Extraction")
    video_paths_state = gr.State({})

    with gr.Row():
        video_dir_input = gr.Textbox(
            label="Video Directory Path", placeholder="Enter path to videos...", lines=1
        )
        video_dropdown = gr.Dropdown(choices=[], multiselect=True, interactive=True)

        video_dir_input.submit(
            fn=update_dropdown,
            inputs=video_dir_input,
            outputs=[video_dropdown, video_paths_state],
        )

    with gr.Row():
        extract_selected_button = gr.Button("Extract Selected Videos")
        extract_all_button = gr.Button("Extract All")
        extract_status = gr.Textbox(label="Status")

        extract_selected_button.click(
            fn=extract_frames,
            inputs=[video_dropdown, video_paths_state],
            outputs=extract_status,
        )

        extract_all_button.click(
            fn=extract_all_frames, inputs=[video_paths_state], outputs=extract_status
        )


def load_frames(dir_str):
    predictor = CONFIG["predictor"]
    frame_data = []
    dir_path = Path(dir_str)
    if dir_path.exists() and dir_path.is_dir():
        frame_data = list(dir_path.rglob("*.jpg"))
        if frame_data:
            frame_data = sorted(frame_data, key=lambda frame: frame.stem)
        CONFIG["inference_state"] = predictor.init_state(dir_str)

    return frame_data, gr.Slider(
        minimum=0,
        maximum=len(frame_data) - 1,
        step=1,
        label="Choose Frame Index",
        interactive=True,
    )


def display_image(frame_data, frame_slider, frame_mask_data):
    image = Image.open(frame_data[frame_slider]).convert("RGB")
    image_np = np.array(image)

    if frame_slider in frame_mask_data:
        masked_image = apply_mask(image_np, frame_mask_data, frame_slider)
        masked_image_pil = Image.fromarray(masked_image)
    else:
        masked_image_pil = image

    return masked_image_pil


def load_label_data():
    label_list = [
        "Barrel",
        "Vehicle",
        "Deer",
        "Pedestrian",
        "Left Lane Must Turn",
        "Right Lane Must Turn",
        "Right Turn Only",
        "Stop",
        "Yield",
        "Barricade-t3",
    ]
    return label_list


def toggle_points(points_state, point_type):
    output_pos = None
    output_neg = None
    if points_state != PointState.OFF and points_state == PointState(point_type):
        points_state = PointState.OFF
        output_pos = f"Create Positive Points"
        output_neg = f"Create Negative Points"
    else:
        if point_type == "Positive":
            points_state = PointState.POSITIVE
            output_pos = "Stop Positive Points"
            output_neg = "Create Negative Points"
        else:
            points_state = PointState.NEGATIVE
            output_pos = "Create Positive Points"
            output_neg = "Stop Negative Points"

    return output_pos, output_neg, points_state


def display_points(image, points, points_type):
    # Check if the image is a PIL Image
    if isinstance(image, Image.Image):
        img = image.convert("RGBA")
    else:
        # Assume it's a NumPy array
        img = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    for point, p_type in zip(points, points_type):
        x, y = point
        radius = 5
        if p_type == 1:  # Positive
            fill_color = (255, 0, 0, 128)  # Semi-transparent red
            outline_color = (255, 0, 0, 255)  # Opaque red
        elif p_type == 0:  # Negative
            fill_color = (0, 0, 255, 128)  # Semi-transparent blue
            outline_color = (0, 0, 255, 255)  # Opaque blue
        else:
            continue  # Skip if undefined

        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=fill_color,
            outline=outline_color,
            width=2,
        )

    combined = Image.alpha_composite(img, overlay)

    return combined


def draw_points(points_state, image, points, points_type, evt: gr.SelectData):
    if points_state == PointState.OFF:
        return image, points, points_type

    if evt is not None:
        x, y = evt.index
        print(f"Clicked at ({x}, {y})")
        points.append((x, y))
        points_type.append(1 if points_state == PointState.POSITIVE else 0)

    image_w_points = display_points(image, points, points_type)
    return image_w_points, points, points_type


def create_mask(
    frame_data,
    frame_idx,
    created_masks,
    points,
    points_type,
    label,
    frame_mask_data,
    obj_id,
    id_2_label,
):
    predictor = CONFIG["predictor"]
    inference_state = CONFIG["inference_state"]

    labels = np.array(points_type, dtype=np.int32)

    # Get new object ID

    id_2_label[obj_id] = label

    # Create mask using predictor
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        points=points,
        labels=labels,
    )

    CONFIG["predictor"] = predictor

    # Process masks
    for i, out_obj_id in enumerate(out_obj_ids):
        mask = (out_mask_logits[i] > 0.0).cpu().numpy()

        # Initialize frame's mask dictionary if needed
        if frame_idx not in frame_mask_data:
            frame_mask_data[frame_idx] = {}
        # Store mask
        frame_mask_data[frame_idx][out_obj_id] = mask

    if frame_idx not in created_masks:
        created_masks[frame_idx] = {}
    if obj_id not in created_masks[frame_idx]:
        created_masks[frame_idx][obj_id] = {}

    created_masks[frame_idx][obj_id]["points"] = points
    created_masks[frame_idx][obj_id]["points_type"] = points_type

    points = []
    points_type = []

    blank_image_pil = Image.open(frame_data[frame_idx]).convert("RGB")
    blank_image = np.array(blank_image_pil)
    image_w_mask = apply_mask(blank_image, frame_mask_data, frame_idx)

    return (
        image_w_mask,
        points,
        points_type,
        frame_mask_data,
        created_masks,
        obj_id + 1,
        id_2_label,
    )


def show_mask(mask, obj_id=None, random_color=False):
    mask = mask.astype(bool)
    color = None
    if random_color:
        color = np.random.rand(3)
    else:
        if obj_id is None:
            color = np.array([1.0, 0.0, 0.0])
        else:
            # Use a distinct colormap
            cmap = plt.get_cmap("tab20")
            cmap_idx = obj_id % cmap.N
            color = np.array(cmap(cmap_idx)[:3])

    # Create colored mask with proper broadcasting
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.float32)
    colored_mask[mask] = color

    return colored_mask


def apply_mask(image, frame_mask_data, frame_idx, alpha=0.5):
    # Convert image to float32 for blending
    image_float = image.astype(np.float32) / 255.0
    result = image_float.copy()

    # Get masks for this frame
    masks_dict = frame_mask_data.get(frame_idx, {})

    for obj_id, mask in masks_dict.items():
        # Ensure mask is 2D
        if mask.ndim > 2:
            mask = mask.squeeze()
        if mask.ndim != 2:
            print(f"Warning: Mask shape {mask.shape} is incompatible")
            continue

        # Resize mask if necessary
        if mask.shape != (image.shape[0], image.shape[1]):
            mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
            mask_pil = mask_pil.resize((image.shape[1], image.shape[0]), Image.NEAREST)
            mask = np.array(mask_pil).astype(bool)

        # Create colored mask
        colored_mask = show_mask(mask, obj_id=obj_id)

        # Blend the mask with the result
        mask_3d = np.stack([mask] * 3, axis=-1)
        result = np.where(mask_3d, result * (1 - alpha) + colored_mask * alpha, result)

    # Convert back to uint8
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return result


def undo_points(frame_data, frame_idx, points, points_type, points_state):
    if len(points) > 0:
        removed_point = points[-1]
        new_points = points[:-1]
        new_points_type = points_type[:-1]
        print(f"Removed point: {removed_point}")
    else:
        print("Points list is empty. Cannot undo.")
        new_points = points
        new_points_type = points_type

    blank_image = Image.open(frame_data[frame_idx])
    image_w_points = display_points(blank_image, new_points, new_points_type)
    return image_w_points, new_points, new_points_type


def undo_masks(
    frame_data, frame_mask_data, created_masks, frame_idx, obj_id, id_2_label
):
    if obj_id <= 0:
        print("No masks to undo.")
        return (
            Image.open(frame_data[frame_idx]).convert("RGB"),
            frame_mask_data,
            obj_id,
            id_2_label,
        )

    obj_id -= 1
    if obj_id in frame_mask_data.get(frame_idx, {}):
        del frame_mask_data[frame_idx][obj_id]
        if obj_id in id_2_label:
            del id_2_label[obj_id]
            del created_masks[frame_idx][obj_id]
        print(f"Removed mask for object ID: {obj_id}")
    else:
        print(f"No mask found for object ID: {obj_id}")

    blank_image_pil = Image.open(frame_data[frame_idx]).convert("RGB")
    blank_image = np.array(blank_image_pil)
    image_w_mask = apply_mask(blank_image, frame_mask_data, frame_idx)

    return image_w_mask, frame_mask_data, created_masks, obj_id, id_2_label


def save_annotated_image(frame_data, frame_mask_data, frame_dir_input, cur_frame_idx):
    video_name = Path(frame_dir_input).stem
    cur_image = None
    current_annotated_frame_dir = CONFIG["annotated_frame_dir"] / video_name
    if not current_annotated_frame_dir.exists():
        current_annotated_frame_dir.mkdir(exist_ok=True, parents=True)

    for out_frame_idx, frame_path in enumerate(frame_data):
        image = Image.open(frame_path).convert("RGB")
        image_np = np.array(image)
        # Overlay the segmentation mask, if it exists for this frame
        if out_frame_idx in frame_mask_data:
            image_np = apply_mask(image_np, frame_mask_data, out_frame_idx)

        masked_image_pil = Image.fromarray(image_np)
        if out_frame_idx == cur_frame_idx:
            cur_image = masked_image_pil
        # Save the image with proper filename and extension
        output_filename = f"{out_frame_idx:05d}.jpg"
        output_path = current_annotated_frame_dir / output_filename
        masked_image_pil.save(output_path)

    return cur_image


def track_masks(
    frame_data,
    frame_mask_data,
    created_masks,
    frame_dir_input,
    cur_frame_idx,
    frame_2_propagate=None,
):
    predictor = CONFIG["predictor"]
    inference_state = CONFIG["inference_state"]
    video_segments = {}

    with torch.autocast("cuda", dtype=torch.bfloat16):
        if frame_2_propagate is not None:
            print(f"Cur Frame: {cur_frame_idx}")
            frame_2_propagate = int(frame_2_propagate)
            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in predictor.propagate_in_video(
                inference_state,
                start_frame_idx=cur_frame_idx,
                reverse=False,
                max_frame_num_to_track=frame_2_propagate,
            ):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                print(f"Out frame index {out_frame_idx}")

        else:
            forward_idx = min(frame_mask_data.keys())
            backward_idx = max(frame_mask_data.keys())
            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in predictor.propagate_in_video(
                inference_state, start_frame_idx=forward_idx, reverse=False
            ):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            # Backward propagation: From annotated frame towards the beginning of the video
            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in predictor.propagate_in_video(
                inference_state,
                start_frame_idx=backward_idx,
                reverse=True,  # Backward direction
            ):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

    cur_image = save_annotated_image(
        frame_data, video_segments, frame_dir_input, cur_frame_idx
    )

    frame_mask_data.update(video_segments)

    reinitialize_predictor(predictor, inference_state, created_masks)

    return cur_image, frame_mask_data


def reinitialize_predictor(predictor, inference_state, created_masks):
    predictor.reset_state(inference_state)
    print("Reinitialzing predictor")

    for frame_idx, objects in created_masks.items():
        for obj_id, obj_data in objects.items():
            points = obj_data.get("points", [])
            points_type = obj_data.get("points_type", [])

            labels = np.array(points_type, dtype=np.int32)

            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
            )

    CONFIG["predictor"] = predictor
    CONFIG["inference_state"] = inference_state

def update_checkBox(id_2_objs):
    print("Update Check Box")
    created_objs = []
    label_counts = Counter(id_2_objs.values())
    cur_occurances = {label: 0 for label in label_counts.keys()}
    
    for id, label in id_2_objs.items():
        entry = f"{id}_{label}_{cur_occurances[label]}"
        cur_occurances[label] += 1
        created_objs.append(entry)

    return created_objs


def annotate_frame_tab():
    with gr.Blocks():
        frame_data = gr.State([])
        label_list = load_label_data()
        points_state = gr.State(PointState.OFF)
        points = gr.State([])
        points_type = gr.State([])
        frame_mask_data = gr.State({})
        created_masks = gr.State({})
        num_obj_id = gr.State(0)
        id_2_label = gr.State({})

        with gr.Row():
            frame_dir_input = gr.Textbox(
                label="Frame Directory Path",
                placeholder="Enter path to Frame...",
                lines=1,
            )

        # Slider in a new row
        with gr.Row():
            frame_slider = gr.Slider(minimum=0, maximum=1, label="Choose Frame Index")

        with gr.Row():
            created_objs = gr.CheckboxGroup(
                choices=[],
                label="Created Object Masks",
                value=[]  
            )

        with gr.Row():
            with gr.Column(min_width=802):
                image_display = gr.Image()

            with gr.Column():
                label_dropdown = gr.Dropdown(choices=label_list, interactive=True)
                with gr.Row():
                    toggle_positive_button = gr.Button(
                        "Create Positive Points", interactive=True
                    )
                    toggle_negative_button = gr.Button(
                        "Create Negative Points", interactive=True
                    )
                    undo_points_button = gr.Button("Undo Point", interactive=True)
                with gr.Row():
                    create_mask_button = gr.Button("Create Masks", interactive=True)
                    undo_masks_button = gr.Button("Undo Mask", interactive=True)
                with gr.Row():
                    propagate_frame_input = gr.Textbox(
                        label="Frames to track",
                        placeholder="Enter Frames To Track...",
                        interactive=True,
                    )
                    track_mask_button = gr.Button("Track All")

        frame_dir_input.submit(
            fn=load_frames, inputs=frame_dir_input, outputs=[frame_data, frame_slider]
        )

        frame_slider.change(
            fn=display_image,
            inputs=[frame_data, frame_slider, frame_mask_data],
            outputs=[image_display],
        )

        toggle_positive_button.click(
            fn=toggle_points,
            inputs=[points_state, gr.State("Positive")],
            outputs=[toggle_positive_button, toggle_negative_button, points_state],
        )

        toggle_negative_button.click(
            fn=toggle_points,
            inputs=[points_state, gr.State("Negative")],
            outputs=[toggle_positive_button, toggle_negative_button, points_state],
        )

        image_display.select(
            fn=draw_points,
            inputs=[points_state, image_display, points, points_type],
            outputs=[image_display, points, points_type],
        )

        create_mask_button.click(
            fn=create_mask,
            inputs=[
                frame_data,
                frame_slider,
                created_masks,
                points,
                points_type,
                label_dropdown,
                frame_mask_data,
                num_obj_id,
                id_2_label,
            ],
            outputs=[
                image_display,
                points,
                points_type,
                frame_mask_data,
                created_masks,
                num_obj_id,
                id_2_label,
            ],
        )

        undo_points_button.click(
            fn=undo_points,
            inputs=[frame_data, frame_slider, points, points_type, points_state],
            outputs=[image_display, points, points_type],
        )

        undo_masks_button.click(
            fn=undo_masks,
            inputs=[
                frame_data,
                frame_mask_data,
                created_masks,
                frame_slider,
                num_obj_id,
                id_2_label,
            ],
            outputs=[
                image_display,
                frame_mask_data,
                created_masks,
                num_obj_id,
                id_2_label,
            ],
        )

        propagate_frame_input.submit(
            fn=track_masks,
            inputs=[
                frame_data,
                frame_mask_data,
                created_masks,
                frame_dir_input,
                frame_slider,
                propagate_frame_input,
            ],
            outputs=[image_display, frame_mask_data],
        )

        track_mask_button.click(
            fn=track_masks,
            inputs=[
                frame_data,
                frame_mask_data,
                created_masks,
                frame_dir_input,
                frame_slider,
            ],
            outputs=[image_display, frame_mask_data],
        )

        id_2_label.change(
            fn=update_checkBox,
            inputs=[id_2_label],
            outputs=[created_objs]
        )


def main():
    with gr.Blocks() as demo:
        gr.Markdown("# SAMv2 Annotator")
        with gr.Tabs():
            with gr.TabItem("Extract Frames"):
                extract_frame_tab()
            with gr.TabItem("Annotate Frames"):
                annotate_frame_tab()
    return demo


if __name__ == "__main__":
    gc_collect()
    seed()
    CONFIG["device"] = torch.device("cuda")
    CONFIG["predictor"] = load_sam()

    if CONFIG["predictor"] is None:
        print("Failed to load SAMv2 Predictor. Exiting application.")
        sys.exit(1)

    CONFIG["frame_dir"] = Path("autolabeller/Frames")
    CONFIG["frame_dir"].mkdir(exist_ok=True, parents=True)

    demo = main()
    demo.launch()
