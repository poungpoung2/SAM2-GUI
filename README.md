
# SAMv2 Video Annotator

## Introduction

**SAMv2 Video Annotator** integrates into the SAMv2 repository to enable efficient annotation of video frames using the Segment Anything Model (SAMv2) and Gradio for the interface.

## Setup

1. **Clone SAMv2 Repository**

    ```bash
    git clone https://github.com/facebookresearch/segment-anything.git
    cd segment-anything
    ```
    
2. **Create Annotator Folder**

    ```bash
    mkdir sam2
    cd sam2
    ```

3. **Add Annotator Code**

    Place your `gradio_app.py` and related scripts inside the `sam2` folder.

4. **Create Videos Directory**

    ```bash
    mkdir videos
    ```

    Add your `.mp4` video files to the `videos` folder:

    ```
    sam2/
    └── sam2_gui/
        ├── gradio_app.py
        └── videos/
            ├── video1.mp4
            └── video2.mp4
    ```

5. **Install Dependencies**

    Ensure Python 3.7+ is installed. Then, install required packages:

    ```bash
    pip install gradio numpy torch matplotlib pillow pycocotools
    ```

6. **Install FFmpeg**

    - **Ubuntu/Linux:**

        ```bash
        sudo apt update
        sudo apt install ffmpeg
        ```

    - **macOS (Homebrew):**

        ```bash
        brew install ffmpeg
        ```

    - **Windows:**

        Download from [FFmpeg](https://ffmpeg.org/download.html) and follow installation instructions.


## Usage

### Run the Annotator

From the `sam2` root directory, execute:

```bash
python sam2/gradio_app.py
