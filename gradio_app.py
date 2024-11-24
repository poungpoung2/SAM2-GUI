import gradio as gr
import os
from pathlib import Path
import numpy as np
import sys
import subprocess
import torch
from sam2.build_sam import build_sam2_video_predictor
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from enum import Enum
import gc
from collections import Counter
from matplotlib import colors as mcolors
from pycocotools import mask as mask_utils
import json


# Configuration setting
CONFIG = {
    "predictor": None,
    "inference_state": None,
    "annotated_frame_dir": Path("autolabeller/Annotated Frames"),
    "last_frame": None,
    "json_dir": Path("autolabeller/COCO_JSON"),
    "traffic_light_states": [
        "4-rleft",
        "4-yleft1",
        "4-yleft2",
        "4-gleft",
        "5dh-red",
        "5dh-yellow",
        "5dh-green",
        "5dh-green-gleft",
        "5dh-green-yleft",
        "5dh-yellow-yleft",
        "5dh-red-gleft",
        "5dh-red-yleft",
        "5dh-off",
        "5dh-other",
    ],
}


# Positive/Negative Point Toggle Custom State
class PointState(Enum):
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    OFF = "Off"


# Collect memory
def gc_collect():
    gc.collect()
    print("Garbage collection complete.")

    # Clear PyTorch caches (if using CUDA)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Cleared PyTorch CUDA cache.")


# Set up seed
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


# Setup the device
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


# Load the sam model
def load_sam(model_cfg, sam2_checkpoint):
    device = CONFIG["device"]
    try:
        predictor = build_sam2_video_predictor(
            model_cfg, sam2_checkpoint, device=device
        )
        print("SAMv2 Predictor loaded successfully.")
        return predictor
    except Exception as e:
        print(f"Failed to load SAMv2 Predictor: {e}")


# Extract frames from video
def extract_frames(videos_list, video_paths_state):
    # Get the framed directory
    frame_dir = CONFIG.get("frame_dir")

    # Loop through video list
    for video in videos_list:
        print(f"Extracting frames from {video}")

        video_path = video_paths_state[video]

        # Create a directory for each video's frames
        output_dir = frame_dir / video.split(".")[0]
        # Use stem to get the video name without the extension
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


# Extract frames for all videos in the Videos directory
def extract_all_frames(video_paths_state):
    if not video_paths_state:
        return "No videos loaded"
    return extract_frames(list(video_paths_state.keys()), video_paths_state)


# Load the videos
def load_videos(dir_str):
    # Get Videos directory path
    dir_path = Path(dir_str)
    video_data = {}
    # Save the path for all valid videos
    if dir_path.exists() and dir_path.is_dir():
        videos_list = list(dir_path.glob("*.mp4"))
        if videos_list:
            for video_path in videos_list:
                video_data[video_path.name] = video_path
    return video_data, list(video_data.keys())


# Update the video option dropdown
def update_dropdown(dir_path):
    video_dict, video_names = load_videos(dir_path)
    return (
        gr.Dropdown(choices=video_names, multiselect=True, interactive=True),
        video_dict,
    )


# Create tab for frame extraction
def extract_frame_tab():
    gr.Markdown("## Frame Extraction")
    video_paths_state = gr.State({})

    with gr.Row():
        # Create textbox to get video directory path
        video_dir_input = gr.Textbox(
            label="Video Directory Path", placeholder="Enter path to videos...", lines=1
        )

        # Dropdown to display video options
        video_dropdown = gr.Dropdown(choices=[], multiselect=True, interactive=True)

        # Grab all the video paths in the folder when user press enter
        video_dir_input.submit(
            fn=update_dropdown,
            inputs=video_dir_input,
            outputs=[video_dropdown, video_paths_state],
        )

    with gr.Row():
        # Button to handle choose video to extract
        extract_selected_button = gr.Button("Extract Selected Videos")
        # Button to handle extract all option
        extract_all_button = gr.Button("Extract All")
        # Show status
        extract_status = gr.Textbox(label="Status")

        # Extract the frames for selected videos in dropdown
        extract_selected_button.click(
            fn=extract_frames,
            inputs=[video_dropdown, video_paths_state],
            outputs=extract_status,
        )

        # Extract frames for all videos
        extract_all_button.click(
            fn=extract_all_frames, inputs=[video_paths_state], outputs=extract_status
        )


# Load the frames in directory
def load_frames(dir_str):
    # Get the SAMv2 model
    predictor = CONFIG["predictor"]
    frame_data = []
    dir_path = Path(dir_str)
    # Check if the path exists
    if dir_path.exists() and dir_path.is_dir():
        # Grab all the frame
        frame_data = list(dir_path.rglob("*.jpg"))
        # Sort the frames
        if frame_data:
            frame_data = sorted(frame_data, key=lambda frame: frame.stem)
        # Load the inference state with the frames
        CONFIG["inference_state"] = predictor.init_state(dir_str)

    # Return a Slider to navigate frames
    return frame_data, gr.Slider(
        minimum=0,
        maximum=len(frame_data) - 1,
        step=1,
        label="Choose Frame Index",
        interactive=True,
    )


# Function to display image for certain frame index
def display_image(frame_data, frame_slider, frame_mask_data):
    # Open the image in RGB format
    image = Image.open(frame_data[frame_slider]).convert("RGB")
    # Convert it to np array
    image_np = np.array(image)

    # If there exists a mask for this frame index
    if frame_slider in frame_mask_data:
        # Apply the segmentation mask on the image
        masked_image = apply_mask(image_np, frame_mask_data, frame_slider)
        masked_image_pil = Image.fromarray(masked_image)
    else:
        masked_image_pil = image

    # Return the masked image
    return masked_image_pil


# List of possible label options
def load_label_data():
    label_list = [
        "Vehicle",
        "Deer",
        "Pedestrian",
        "Barrel",
        "Barricade-t3",
        "Stop",
        "Yield",

        "Right Lane Must Turn",
        "Left Lane Must Turn",

        "Right Turn Only (words)"
        "Left Turn Only (words)",

        "Right Turn Only (arrow)",
        "Left Turn Only (arrow)",

        "3-bulb",
        "4-bulb",
        "5-bulb",

        "Railroad-crossing-sign",
        "Railroad-light-pair-on",
        "Railroad-light-pair-off",
        
        "speedLimit5",
        "speedLimit10",
        "speedLimit15",
        "speedLimit20",
        "speedLimit25",
        "speedLimit30",
        "speedLimit35",
        "speedLimit40",
        "speedLimit45",
        "speedLimit50",
        "speedLimit55",
        "speedLimit60",
        "speedLimit65",
        "speedLimit70",
        "speedLimit75",
    ]
    return label_list


# Generate color for each object
def generate_color_for_id(obj_id):
    # Use a distinct colormap to get a unique color
    cmap = plt.get_cmap("tab20")
    color = cmap(obj_id % cmap.N)[:3]  # Get RGB tuple from colormap
    hex_color = mcolors.to_hex(color)  # Convert to hex for HTML
    return hex_color


# Function to change the point stage (Postivie/Negative/Off)
def toggle_points(points_state, point_type):
    output_pos = None
    output_neg = None
    # If the point is on and the currently selected mode button is pressed again
    if points_state != PointState.OFF and points_state == PointState(point_type):
        # Turn off the point annotating mode
        points_state = PointState.OFF
        output_pos = f"Create Positive Points"
        output_neg = f"Create Negative Points"
    else:
        # Check if the positive point button is pressed
        if point_type == "Positive":
            points_state = PointState.POSITIVE
            output_pos = "Stop Positive Points"
            output_neg = "Create Negative Points"
        # Check if the negative point button is pressed
        else:
            points_state = PointState.NEGATIVE
            output_pos = "Create Positive Points"
            output_neg = "Stop Negative Points"

    # Return button inner text and point state
    return output_pos, output_neg, points_state


# Function to display the annoated points
def display_points(image, points, points_type):
    # Check if the image is a PIL Image
    if isinstance(image, Image.Image):
        img = image.convert("RGBA")
    else:
        # Assume it's a NumPy array
        img = Image.fromarray(image).convert("RGBA")

    # Default overlay and draws
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Loop through all the points
    for point, p_type in zip(points, points_type):
        x, y = point
        radius = 5
        # Check if positive point (1)
        if p_type == 1:
            fill_color = (255, 0, 0, 128)  # Semi-transparent red
            outline_color = (255, 0, 0, 255)  # Opaque red
        # Check if negative points
        elif p_type == 0:
            fill_color = (0, 0, 255, 128)  # Semi-transparent blue
            outline_color = (0, 0, 255, 255)  # Opaque blue
        # If the point is off skip
        else:
            continue

        # Draw the point on the image
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=fill_color,
            outline=outline_color,
            width=2,
        )

    # Combine the image and the overlay
    combined = Image.alpha_composite(img, overlay)

    # Return the annoated image
    return combined


# Function to manage drawing points process on the image
def draw_points(points_state, image, points, points_type, evt: gr.SelectData):
    # Check if the point annoation is off
    if points_state == PointState.OFF:
        return image, points, points_type

    # Check if the image was clicked
    if evt is not None:
        x, y = evt.index
        # Append the clicked pixel coordinate
        points.append((x, y))
        # Append the point mode for this click action
        points_type.append(1 if points_state == PointState.POSITIVE else 0)

    # Display the point on the image and return
    image_w_points = display_points(image, points, points_type)
    return image_w_points, points, points_type


# Function to manage mask creation (using the annotated points for mask generation)
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
    is_edit,
    selected_obj,
):
    # Load the preditor and inferene state
    predictor = CONFIG["predictor"]
    inference_state = CONFIG["inference_state"]
    CONFIG["last_frame"] = frame_idx

    # Get the lables (point type) into an numpy array
    labels = np.array(points_type, dtype=np.int32)
    button_msg = "Stop Editing"

    # Check if the edit mode is on
    if is_edit:
        obj = selected_obj[0]
        obj_id = int(obj.split("_")[0])
        # Remove the previsouly created mask for the selected object
        if obj_id in frame_mask_data.get(frame_idx, {}):
            del frame_mask_data[frame_idx][obj_id]

        # Reset edit mode after creating mask
        is_edit = False
        button_msg = "Start Editing"

    else:
        id_2_label[obj_id] = label

    # Create mask using predictor
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        points=points,
        labels=labels,
    )

    # Update the predictor state
    CONFIG["predictor"] = predictor

    # Process masks
    for i, out_obj_id in enumerate(out_obj_ids):
        mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()

        # Initialize frame's mask dictionary if needed
        if frame_idx not in frame_mask_data:
            frame_mask_data[frame_idx] = {}
        # Store mask
        frame_mask_data[frame_idx][out_obj_id] = mask

    # Save the created mask into a dictionary
    if frame_idx not in created_masks:
        created_masks[frame_idx] = {}
    if obj_id not in created_masks[frame_idx]:
        created_masks[frame_idx][obj_id] = {}

    # Save the created masks data for undo
    created_masks[frame_idx][obj_id]["points"] = points
    created_masks[frame_idx][obj_id]["points_type"] = points_type

    # Reset the points and points type for next mask generation
    points = []
    points_type = []

    # Apply the created mask onto the image
    blank_image_pil = Image.open(frame_data[frame_idx]).convert("RGB")
    blank_image = np.array(blank_image_pil)
    image_w_mask = apply_mask(blank_image, frame_mask_data, frame_idx)

    return (
        image_w_mask,
        points,
        points_type,
        frame_mask_data,
        created_masks,
        obj_id + 1 if not is_edit else obj_id,
        id_2_label,
        is_edit,
        button_msg,
        [],
    )


# Function to create the colored mask for a binary mask array
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


# A function to apply the mask onto the image
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


# A function to undo points
def undo_points(frame_data, frame_idx, points, points_type):
    # Check if there is any point to remove
    if len(points) > 0:
        # Remove the points
        removed_point = points[-1]
        new_points = points[:-1]
        new_points_type = points_type[:-1]
        print(f"Removed point: {removed_point}")
    else:
        print("Points list is empty. Cannot undo.")
        new_points = points
        new_points_type = points_type

    # Update the displayed image
    blank_image = Image.open(frame_data[frame_idx])
    image_w_points = display_points(blank_image, new_points, new_points_type)
    return image_w_points, new_points, new_points_type


# A function to undo previously created mask
def undo_masks(
    frame_data, frame_mask_data, created_masks, frame_idx, obj_id, id_2_label
):
    # Check if there are no masks created
    if obj_id <= 0:
        print("No masks to undo.")
        return (
            Image.open(frame_data[frame_idx]).convert("RGB"),
            frame_mask_data,
            obj_id,
            id_2_label,
        )

    # Reduce the object id
    obj_id -= 1
    # Remove the previous mask
    if obj_id in frame_mask_data.get(frame_idx, {}):
        del frame_mask_data[frame_idx][obj_id]
        if obj_id in id_2_label:
            del id_2_label[obj_id]
            del created_masks[frame_idx][obj_id]
        print(f"Removed mask for object ID: {obj_id}")
    else:
        print(f"No mask found for object ID: {obj_id}")

    # Update the displayed image
    blank_image_pil = Image.open(frame_data[frame_idx]).convert("RGB")
    blank_image = np.array(blank_image_pil)
    image_w_mask = apply_mask(blank_image, frame_mask_data, frame_idx)

    return image_w_mask, frame_mask_data, created_masks, obj_id, id_2_label


# A function to save the annotated video
def save_annotated_video(frame_data, frame_mask_data, frame_dir_input):
    print(f"Saving Video")
    video_name = Path(frame_dir_input).stem
    current_annotated_frame_dir = CONFIG["annotated_frame_dir"] / video_name

    # Create the directory for annotated frames if it doesn't exist
    if not current_annotated_frame_dir.exists():
        current_annotated_frame_dir.mkdir(exist_ok=True, parents=True)

    # Save annotated frames
    for out_frame_idx, frame_path in enumerate(frame_data):
        image = Image.open(frame_path).convert("RGB")
        image_np = np.array(image)
        # Overlay the segmentation mask, if it exists for this frame
        if out_frame_idx in frame_mask_data:
            image_np = apply_mask(image_np, frame_mask_data, out_frame_idx)

        masked_image_pil = Image.fromarray(image_np)
        # Save the image with proper filename and extension
        output_filename = f"{out_frame_idx:05d}.jpg"
        output_path = current_annotated_frame_dir / output_filename
        masked_image_pil.save(output_path)

    # After saving frames, create a video from the images using FFmpeg
    # Define the path for the output video
    output_video_path = CONFIG["annotated_frame_dir"] / f"{video_name}_annotated.mp4"

    # Build the FFmpeg command
    # Adjust the frame rate (-framerate) as needed
    command = [
        "ffmpeg",
        "-y",  # Overwrite output files without asking
        "-framerate", "30",  # Set the input frame rate
        "-i", str(current_annotated_frame_dir / "%05d.jpg"),
        "-c:v", "libx264",  # Use the H.264 codec
        "-pix_fmt", "yuv420p",  # Set the pixel format
        str(output_video_path)
    ]

    # Run the FFmpeg command
    subprocess.run(command, check=True)

    print(f"Annotated video saved at {output_video_path}")


# A function to track masks over multiple frames
def track_masks(
    frame_data,
    frame_mask_data,
    created_masks,
    id_2_objs,
    frame_dir_input,
    cur_frame_idx,
    frame_2_propagate=None,
    save_annotated_image=False,
):
    # Load predictor and inference state
    predictor = CONFIG["predictor"]
    inference_state = CONFIG["inference_state"]
    # Dictionary to save the generated masks for each frame
    video_segments = {}

    with torch.autocast("cuda", dtype=torch.bfloat16):
        # Check if the model is propagating only a certain number of frames
        if frame_2_propagate is not None:
            print(f"Cur Frame: {cur_frame_idx}")
            frame_2_propagate = int(frame_2_propagate)
            # Propage the video and save the masks
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
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

        # If we are tracking through all the videos
        else:
            # Find the idx to start forward and bacward tracking to reduce the number of overlaps
            forward_idx = min(created_masks.keys())
            backward_idx = max(created_masks.keys())
            # Forward propagation
            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in predictor.propagate_in_video(
                inference_state, start_frame_idx=forward_idx, reverse=False
            ):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            # Backward propagation
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
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

            # Check if the annotated framem images should be saved (Default False)
            if save_annotated_image:
                save_annotated_image(
                    frame_data, video_segments, frame_dir_input, cur_frame_idx
                )
            # create_coco_json(frame_data, frame_dir_input, video_segments, id_2_objs)

    # Update the mask data for each frame
    frame_mask_data.update(video_segments)

    # Call a function reinitliaze the model as the model should be resetted after each tracking session
    reinitialize_predictor(predictor, inference_state, created_masks)

    return frame_mask_data


# A function to reload the annoated frame information onto the model
def reinitialize_predictor(predictor, inference_state, created_masks):
    # Reset the state of the predictor
    predictor.reset_state(inference_state)
    print("Reinitialzing predictor")

    # Loop through the annoated mask for the users
    for frame_idx, objects in created_masks.items():
        for obj_id, obj_data in objects.items():
            points = obj_data.get("points", [])
            points_type = obj_data.get("points_type", [])

            labels = np.array(points_type, dtype=np.int32)

            # Reload the annoatation infomration to the model
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
            )

    # Save the reloaded model
    CONFIG["predictor"] = predictor
    CONFIG["inference_state"] = inference_state


# Function to update the created objects and traffic label
def update_checkBoxes(frame_idx, id_2_objs, id_2_traffic):
    # Save the objects
    created_objs = []
    created_traffics = []
    # Count the number of occurances
    label_counts = Counter(id_2_objs.values())
    cur_occurances = {label: 0 for label in label_counts.keys()}

    # Loop through all the created objects
    for id, label in id_2_objs.items():
        # Check if this object is a traffic and a state is assinged
        if id in id_2_traffic and frame_idx in id_2_traffic[id]:
            # Fillout the text to be displayed
            traffic_data = id_2_traffic[id][frame_idx]
            entry = f"{id}_{traffic_data}_{cur_occurances[label]}"
        else:
            entry = f"{id}_{label}_{cur_occurances[label]}"
            # Check if the object is a traffic
            if "bulb" in label:
                # Save it to the list that stores unassinged traffics
                created_traffics.append(entry)

        created_objs.append(entry)
        cur_occurances[label] += 1

    return gr.CheckboxGroup(
        choices=created_objs,
        value=[],
        interactive=True,
        label="Created Object Masks {obj_id}_{label}_{cur_occurances}",
    ), gr.CheckboxGroup(
        choices=created_traffics,
        value=[],
        interactive=True,
        label="Traffic States that should be annotated",
    )


# A function to display the unannotated traffic object ranges
def unannotated_traffic(id_2_objs, id_2_traffic, total_frames):
    # Set of all frames in the video
    all_frames = set(range(total_frames))
    unannotated = []
    # Loop through all the objects
    for obj_id, label in id_2_objs.items():
        # Check if the traffic object
        if "bulb" in label:
            # Get the range of annotated frames
            annotated_frames = set(id_2_traffic.get(obj_id, {}).keys())
            # Ge the frames that the state is not assigned
            missing_frames = sorted(all_frames - annotated_frames)

            if missing_frames:
                # Group missing frames into ranges
                ranges = []
                start = prev = missing_frames[0]
                # Loop through the frame to add ranges
                for frame in missing_frames[1:]:
                    if frame == prev + 1:
                        prev = frame
                    else:
                        ranges.append((start, prev))
                        start = prev = frame
                # Add the last range
                ranges.append((start, prev))

                # Create a combined range string
                range_str = " ".join(
                    f"{start}-{end}" if start != end else f"{start}"
                    for start, end in ranges
                )
                unannotated.append(f"{obj_id}_{label}_missing: {range_str}")

    return unannotated


# A function to assign state to a traffic object
def assign_label_state(frame_idx, selected_traffics, id_2_traffic, id_2_objs, traffic_state_dropdown, start_frame, end_frame, frame_data, display_unannotated_traffic):
    if not selected_traffics:
        print("No traffic selected.")
        return id_2_traffic, display_unannotated_traffic, []
    
    # Get the selected traffic
    selected_traffic = selected_traffics[0]
    obj_id = int(selected_traffic.split("_")[0])
    label = id_2_objs[obj_id]
    
    # Get the selected traffic state from the dropdown
    selected_state = traffic_state_dropdown
    
    # Check for valid ranges
    if start_frame > end_frame:
        print("Invalid Range: start_frame is greater than end_frame.")
        return id_2_traffic, display_unannotated_traffic, []
    
    # Initialize if traffic_id not present
    if obj_id not in id_2_traffic:
        id_2_traffic[obj_id] = {}
    
    # Assign the state for designated frames
    for idx in range(int(start_frame), int(end_frame) + 1):
        id_2_traffic[obj_id][idx] = selected_state  # Use the selected state directly
    
    # Update the checkboxes 
    update_checkBoxes(frame_idx, id_2_objs, id_2_traffic)
    
    total_frame = len(frame_data)
    # Recalculate the frames that should be annotated
    unannotated = unannotated_traffic(id_2_objs, id_2_traffic, total_frame) 

    return id_2_traffic, unannotated, []


# A function to convert masks into json format
def create_coco_json(
    frame_data, frame_dir_input, frame_mask_data, id_2_objs, id_2_traffic
):
    categories = []
    images = []
    annotations = []
    coco_json = {}

    frame_dir = Path(frame_dir_input)
    video_name = frame_dir.stem

    # Define super categories and traffic states
    traffic_states = CONFIG["traffic_light_states"]
    

    # Extract unique labels from id_2_objs and remove super categories
    unique_labels_set = set(id_2_objs.values())
    # Remove intermediate lables
    intermediate_labels = {"3-bulb", "4-bulb", "5-bulb"}  # Add any other unwanted labels here
    unique_labels_set -= intermediate_labels
    unique_labels_set.update(set(traffic_states))
    unique_labels = sorted(unique_labels_set)

    # Assign unique category IDs to unique_labels
    label_to_id = {label.lower(): idx + 1 for idx, label in enumerate(unique_labels)}

    for label, category_id in label_to_id.items():
        categories.append(
            {"id": category_id, "name": label.lower(), "supercategory": ""}
        )

    # Fillout the lincense information
    coco_json["licenses"] = [{"name": "", "id": 0, "url": ""}]
    coco_json["info"] = {
        "contributor": "",
        "date_created": "",
        "description": "",
        "url": "",
        "version": "",
        "year": "",
    }

    # Image entries
    output_pattern = "%05d.jpg"

    # Loop through all frames
    for frame_idx in range(len(frame_data)):
        file_name = output_pattern % frame_idx
        # Add the image information
        images.append(
            {
                "id": frame_idx + 1,
                "license": 0,
                "file_name": file_name,
                "height": 1100,
                "width": 1604,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0,
            }
        )

    # Annotation entries
    annotation_id = 1
    # Mapping from (obj_id, label) to tracking_id
    traffic_id_mapping = {}
    # Counter per obj_id to generate unique tracking_ids
    tracking_id_counters = {}  
    # Loop through all frames
    for frame_idx in range(len(frame_data)):
        # Check if there is a mask data for the frame index
        if frame_idx not in frame_mask_data:
            continue
        # Get all masks present in the frame
        masks = frame_mask_data[frame_idx]
        for obj_id, mask in masks.items():
            # Initialize tracking ID
            tracking_id = None

            # Base tracking ID derived from obj_id
            base_tracking_id = obj_id * 100

            # Initialize counter for obj_id if not present
            if obj_id not in tracking_id_counters:
                tracking_id_counters[obj_id] = 0

            # Check if the object is a traffic light and the state is assigned
            if obj_id in id_2_traffic and frame_idx in id_2_traffic[obj_id]:
                # Traffic light with state
                label = id_2_traffic[obj_id][frame_idx]
                # Use the same tracking ID if (obj_id, label) exists
                state_key = (obj_id, label)
                if state_key in traffic_id_mapping:
                    tracking_id = traffic_id_mapping[state_key]
                else:
                    # Assign a new tracking ID
                    tracking_id = base_tracking_id + tracking_id_counters[obj_id]
                    tracking_id_counters[obj_id] += 1  # Increment counter
                    traffic_id_mapping[state_key] = tracking_id
            else:
                # Assingn state for non-traffic light or traffic light without state
                label = id_2_objs.get(obj_id, "Unknown").lower()
                tracking_id = base_tracking_id

            # Get the catogory id for the label
            category_id = label_to_id.get(label, -1)
            if category_id == -1:
                print(f"Warning: Label '{label}' not found in category list.")
                continue

            # Check if the mask is 2 dimensional (H, W)
            if mask.ndim > 2:
                mask = mask.squeeze()

            # Threshold mask to binary
            binary_mask = (mask > 0).astype(np.uint8)

            # Convert binary mask to Fortran order for RLE encoding
            mask_fortran = np.asfortranarray(binary_mask)
            rle = mask_utils.encode(mask_fortran)

            # Convert 'counts' to a list of integers (uncompressed RLE)
            counts_bytes = rle["counts"]
            if isinstance(counts_bytes, bytes):
                counts_array = np.frombuffer(counts_bytes, dtype=np.uint8)
                counts_list = counts_array.tolist()
            else:
                counts_list = rle["counts"]

            segmentation = {"size": rle["size"], "counts": counts_list}

            # Calculate area
            area = float(mask_utils.area(rle))

            # Calculate bounding box
            bbox = mask_utils.toBbox(rle).tolist()

            # Validate area and bbox
            if area == 0 or all(v == 0 for v in bbox):
                print(
                    f"Warning: Zero area or bounding box for object ID {obj_id} in frame {frame_idx}"
                )
                continue

            # Append the annotation entry
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": frame_idx + 1,
                    "category_id": category_id,
                    "segmentation": [],
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                    "attributes": {
                        "occluded": False,
                        "rotation": 0.0,
                        "track_id": tracking_id,
                        "keyframe": True,
                    },
                },
            )
            annotation_id += 1

    # Update the coco_json dictionary
    coco_json.update(
        {"categories": categories, "images": images, "annotations": annotations}
    )

    # Set the ouput path
    output_path = CONFIG["json_dir"] / f"{video_name}.json"

    # Save the json file in the frames direcotry and the json directory
    with open(output_path, "w") as json_file:
        json.dump(coco_json, json_file)

    output_path = frame_dir / "annotation.json"
    with open(output_path, "w") as json_file:
        json.dump(coco_json, json_file)

    print("COCO Annotation Saved")


# A functiont to edit the created masks
def edit_mask(selected_objs, frame_mask_data, frame_data, frame_idx, is_edit):
    if not selected_objs:
        print("No mask selected for editing.")
        return (
            Image.open(frame_data[frame_idx]).convert("RGB"),
            False,  # Reset edit mode
            "Start Editing",  # Reset button text
            [],  # Clear selection
        )

    # Get the first selected object
    obj = selected_objs[0]
    obj_id = int(obj.split("_")[0])

    # Create a copy of mask data to avoid modifying the original
    local_mask_data = frame_mask_data.copy()

    # Toggle edit mode
    if is_edit:
        # Exiting edit mode - use original mask data to show all masks
        button_msg = "Start Editing"
        print(f"Stopped editing mask for object ID: {obj_id}")
        is_edit = False
        selected = []  # Clear selection when stopping edit
        local_mask_data = frame_mask_data  # Use original mask data with all masks
    else:
        # Entering edit mode
        button_msg = "Stop Editing"
        if frame_idx in frame_mask_data and obj_id in frame_mask_data[frame_idx]:
            # Temporarily remove the mask being edited from display
            if frame_idx not in local_mask_data:
                local_mask_data[frame_idx] = {}
            local_mask_data[frame_idx] = frame_mask_data[
                frame_idx
            ].copy()  # Copy the frame's masks
            del local_mask_data[frame_idx][obj_id]  # Remove only the edited mask
            print(f"Started editing mask for object ID: {obj_id}")
        is_edit = True
        selected = selected_objs  # Maintain selection while editing

    # Update the display
    blank_image_pil = Image.open(frame_data[frame_idx]).convert("RGB")
    blank_image = np.array(blank_image_pil)
    image_w_mask = apply_mask(blank_image, local_mask_data, frame_idx)

    return image_w_mask, is_edit, button_msg, selected


def annotate_frame_tab():
    with gr.Blocks():

        # Stores the list of frame file paths
        frame_data = gr.State([])
        # Retrieves the list of available labels
        label_list = load_label_data()
        # Current state for point creation (Positive/Negative/Off)
        points_state = gr.State(PointState.OFF)
        # Stores the list of points clicked by the user
        points = gr.State([])
        # Stores the type of each point (1 for Positive, 0 for Negative)
        points_type = gr.State([])
        # Stores masks for each frame {frame_idx: {obj_id: mask}}
        frame_mask_data = gr.State({})
        # Stores created masks {frame_idx: {obj_id: mask_details}}
        created_masks = gr.State({})
        # Counter for assigning unique object IDs
        num_obj_id = gr.State(0)
        # Maps object IDs to their labels {obj_id: label}
        id_2_label = gr.State({})
        # Maps object IDs to traffic states {obj_id: {frame_idx: state}}
        id_2_traffic = gr.State({})
        # Indicates whether the user is in edit mode
        is_edit = gr.State(False)
        # Traffic light states
        traffic_light_states = CONFIG["traffic_light_states"]


        # Input for Frame Directory Selection
        with gr.Row():
            frame_dir_input = gr.Textbox(
                label="Frame Directory Path",
                placeholder="Enter path to Frame...",
                lines=1,
            )

        # Slider in a new row
        with gr.Row():
            # Slider lets users select the frame index to view and annotate.
            frame_slider = gr.Slider(minimum=0, maximum=1, label="Choose Frame Index")

        # Created Object Masks Display and Edit Button
        with gr.Row():
            with gr.Column(scale=3):
                # CheckboxGroup to display all created object masks with their IDs and labels.
                created_objs = gr.CheckboxGroup(
                    choices=[],
                    label="Created Object Masks {obj_id}_{label}_{cur_occurances}",
                    value=[],
                )
            # Button to toggle editing mode for selected masks.
            edit_button = gr.Button("Start Editing")

        with gr.Row():
            # TextArea to shows ranges of frames that have not been annotated for traffic objects.
            display_unannotated_traffic = gr.TextArea(
                label="Unannotated Traffic", lines=1, interactive=False
            )

        with gr.Row():
            # CheckboxGroup to select traffic objects for state annotation.
            with gr.Column(scale=2):
                created_traffics = gr.CheckboxGroup(
                    choices=[],
                    label="Created Traffic State",
                    value=[],
                    interactive=True,
                )

            # Radio buttons to select the state of the traffic object.
            traffic_state = gr.Dropdown(
                choices = traffic_light_states,
                label="Traffic State",
                value=traffic_light_states[0],
                interactive=True,
            )
            with gr.Row():
                # Number inputs to specify the range of frames for traffic state annotation.
                start_frame = gr.Number(
                    label="Start Frame",
                    precision=0,
                )
                end_frame = gr.Number(label="End Frame", precision=0)

        with gr.Row():
            with gr.Column(scale=3):
                # Displays the current frame with applied masks.
                image_display = gr.Image()

            with gr.Column():
                # Dropdown to select the label/category for the object being annotated.
                label_dropdown = gr.Dropdown(choices=label_list, interactive=True)
                with gr.Row():
                    # Buttons to toggle between creating positive/negative points and to undo the last point.
                    toggle_positive_button = gr.Button(
                        "Create Positive Points", interactive=True
                    )
                    toggle_negative_button = gr.Button(
                        "Create Negative Points", interactive=True
                    )
                    undo_points_button = gr.Button("Undo Point", interactive=True)
                with gr.Row():
                    # Buttons to create masks based on points and to undo the last mask.
                    create_mask_button = gr.Button("Create Masks", interactive=True)
                    undo_masks_button = gr.Button("Undo Mask", interactive=True)
                with gr.Row():
                    with gr.Column():
                        # Number input to specify how many frames to propagate masks.
                        propagate_frame_input = gr.Number(
                            label="Frames to track",
                            interactive=True,
                        )
                    with gr.Column():
                        # Button to track masks across frames and to export annotations as COCO JSON.
                        track_mask_button = gr.Button("Track All")
                        video_button = gr.Button("Save it as Video")
                        json_button = gr.Button("Export Json")


        # When the user submits (enters) the frame directory path, load the frames.
        frame_dir_input.submit(
            fn=load_frames, inputs=frame_dir_input, outputs=[frame_data, frame_slider]
        )

        # When the frame slider value changes, display the corresponding image with masks.
        frame_slider.change(
            fn=display_image,
            inputs=[frame_data, frame_slider, frame_mask_data],
            outputs=[image_display],
        )

        # Toggle positive points creation mode.
        toggle_positive_button.click(
            fn=toggle_points,
            inputs=[points_state, gr.State("Positive")],
            outputs=[toggle_positive_button, toggle_negative_button, points_state],
        )

        # Toggle negative points creation mode.
        toggle_negative_button.click(
            fn=toggle_points,
            inputs=[points_state, gr.State("Negative")],
            outputs=[toggle_positive_button, toggle_negative_button, points_state],
        )

        # When the user selects a point on the image, add it to the points list.
        image_display.select(
            fn=draw_points,
            inputs=[points_state, image_display, points, points_type],
            outputs=[image_display, points, points_type],
        )

        # Create masks based on the points and selected label.
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
                is_edit,
                created_objs,
            ],
            outputs=[
                image_display,
                points,
                points_type,
                frame_mask_data,
                created_masks,
                num_obj_id,
                id_2_label,
                is_edit,
                edit_button,
                created_objs,
            ],
        )

        # Undo the last point added.
        undo_points_button.click(
            fn=undo_points,
            inputs=[frame_data, frame_slider, points, points_type],
            outputs=[image_display, points, points_type],
        )

        # Undo the last mask created.
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

        # When the user inputs the number of frames to propagate, track masks.
        propagate_frame_input.submit(
            fn=track_masks,
            inputs=[
                frame_data,
                frame_mask_data,
                created_masks,
                id_2_label,
                frame_dir_input,
                frame_slider,
                propagate_frame_input,
            ],
            outputs=[frame_mask_data],
        )

        # When the "Track All" button is clicked, track masks across all frames.
        track_mask_button.click(
            fn=track_masks,
            inputs=[
                frame_data,
                frame_mask_data,
                created_masks,
                id_2_label,
                frame_dir_input,
                frame_slider,
            ],
            outputs=[frame_mask_data],
        )

        # Toggle edit mode for selected masks.
        edit_button.click(
            fn=edit_mask,
            inputs=[created_objs, frame_mask_data, frame_data, frame_slider, is_edit],
            outputs=[image_display, is_edit, edit_button, created_objs],
        )

        # Export annotations to COCO JSON format.
        json_button.click(
            fn=create_coco_json,
            inputs=[
                frame_data,
                frame_dir_input,
                frame_mask_data,
                id_2_label,
                id_2_traffic,
            ],
        )

        # Update the checkbox groups when the label mapping changes.
        id_2_label.change(
            fn=update_checkBoxes,
            inputs=[frame_slider, id_2_label, id_2_traffic],
            outputs=[created_objs, created_traffics],
        )

        # Update the checkbox groups when the frame slider changes.
        frame_slider.change(
            fn=update_checkBoxes,
            inputs=[frame_slider, id_2_label, id_2_traffic],
            outputs=[created_objs, created_traffics],
        )

        # Assign traffic state when the user submits the start frame number.
        start_frame.submit(
            fn=assign_label_state,
            inputs=[
                frame_slider,
                created_traffics,
                id_2_traffic,
                id_2_label,
                traffic_state,
                start_frame,
                end_frame,
                frame_data,
                display_unannotated_traffic,
            ],
            outputs=[id_2_traffic, display_unannotated_traffic, created_traffics],
        )

        # Assign traffic state when the user submits the end frame number.
        end_frame.submit(
            fn=assign_label_state,
            inputs=[
                frame_slider,
                created_traffics,
                id_2_traffic,
                id_2_label,
                traffic_state,
                start_frame,
                end_frame,
                frame_data,
                display_unannotated_traffic,
            ],
            outputs=[id_2_traffic, display_unannotated_traffic, created_traffics],
        )

        video_button.click(
            fn=save_annotated_video,
            inputs=[
                frame_data,
                frame_mask_data,
                frame_dir_input,
            ],
            outputs=None, 
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
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    sam2_checkpoint = "checkpoints/sam2.1_hiera_base_plus.pt"

    gc_collect()
    seed()
    CONFIG["device"] = setup_device()
    CONFIG["predictor"] = load_sam(model_cfg, sam2_checkpoint)

    if CONFIG["predictor"] is None:
        print("Failed to load SAMv2 Predictor. Exiting application.")
        sys.exit(1)

    CONFIG["frame_dir"] = Path("autolabeller/Frames")
    CONFIG["frame_dir"].mkdir(exist_ok=True, parents=True)
    CONFIG["json_dir"].mkdir(exist_ok=True, parents=True)

    demo = main()
    demo.launch()
