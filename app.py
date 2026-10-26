
import io

import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM


st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="📸",
    layout="wide"
)

MODEL_NAME = "microsoft/Florence-2-base"
DEVICE = "cpu"


@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    model.to(DEVICE)
    model.eval()

    return processor, model


def generate_analysis(image, mode, processor, model, object_prompt=""):
    if mode == "Simple Caption":
        task = "<CAPTION>"
        prompt = task

    elif mode == "Detailed Caption":
        task = "<MORE_DETAILED_CAPTION>"
        prompt = task

    elif mode == "Object Detection":
        task = "<OD>"
        prompt = task

    else:
        task = "<CAPTION_TO_PHRASE_GROUNDING>"

        phrases = [
            phrase.strip()
            for phrase in object_prompt.replace(".", ",").split(",")
            if phrase.strip()
        ]

        prompt = task + " " + ". ".join(phrases) + "."

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    )

    with torch.inference_mode():
        output = model.generate(
            input_ids=inputs["input_ids"].to(DEVICE),
            pixel_values=inputs["pixel_values"].to(DEVICE),
            max_new_tokens=128,
            num_beams=3,
            do_sample=False
        )

    generated_text = processor.batch_decode(
        output,
        skip_special_tokens=False
    )[0]

    result = processor.post_process_generation(
        generated_text,
        task=task,
        image_size=(image.width, image.height)
    )

    return result[task]


def draw_detections(image, result, box_color="#FF3333", box_width=4):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            max(14, image.width // 55)
        )
    except OSError:
        font = ImageFont.load_default()

    boxes = result.get("bboxes", [])
    labels = result.get("labels", [])

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box)

        x1 = max(0, min(x1, image.width - 1))
        x2 = max(0, min(x2, image.width - 1))
        y1 = max(0, min(y1, image.height - 1))
        y2 = max(0, min(y2, image.height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=box_color,
            width=box_width
        )

        label = str(label)
        text_bounds = draw.textbbox(
            (0, 0),
            label,
            font=font
        )

        text_width = text_bounds[2] - text_bounds[0]
        text_height = text_bounds[3] - text_bounds[1]

        label_top = max(0, y1 - text_height - 10)

        draw.rectangle(
            [
                x1,
                label_top,
                min(image.width - 1, x1 + text_width + 10),
                min(image.height - 1, label_top + text_height + 8)
            ],
            fill=box_color
        )

        draw.text(
            (x1 + 5, label_top + 3),
            label,
            fill="white",
            font=font
        )

    return annotated


def image_to_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def results_to_text(result):
    if isinstance(result, dict):
        labels = result.get("labels", [])
        boxes = result.get("bboxes", [])

        if not labels:
            return "No objects were detected."

        lines = []

        for index, (label, box) in enumerate(
            zip(labels, boxes),
            start=1
        ):
            coordinates = ", ".join(
                str(round(float(value), 2))
                for value in box
            )

            lines.append(
                f"{index}. {label}\n"
                f"   Bounding box [x1, y1, x2, y2]: "
                f"[{coordinates}]"
            )

        return "\n".join(lines)

    return str(result)


st.title("📸 AI Image Caption Generator")

st.write(
    "Generate image captions, detailed descriptions, "
    "and visual object detections using Florence-2."
)

st.caption(
    "Runs locally on your computer after the model is downloaded. "
    "Detection results may not include every object."
)

with st.sidebar:
    st.header("Analysis Settings")

    analysis_mode = st.selectbox(
        "Analysis Mode",
        [
            "Simple Caption",
            "Detailed Caption",
            "Object Detection",
            "Find Specific Objects"
        ]
    )

    object_prompt = ""

    if analysis_mode == "Find Specific Objects":
        object_prompt = st.text_input(
            "Objects to locate",
            placeholder="man, punching bag",
            help="Separate multiple objects with commas."
        )

    st.divider()

    box_color = st.color_picker(
        "Bounding Box Color",
        "#FF3333"
    )

    box_width = st.slider(
        "Bounding Box Thickness",
        min_value=1,
        max_value=8,
        value=4
    )

    st.caption("Model: Florence-2-base")
    st.caption("Device: CPU")


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)


if uploaded_file is None:
    st.info("Upload an image to begin.")
    st.stop()


try:
    image = Image.open(uploaded_file).convert("RGB")

except Exception as error:
    st.error(f"Could not open the image: {error}")
    st.stop()


left, right = st.columns([1, 1])

with left:
    st.subheader("Original Image")
    st.image(image, use_container_width=True)
    st.caption(f"Image size: {image.width} × {image.height}")

with right:
    st.subheader("AI Analysis")

    if analysis_mode == "Find Specific Objects":
        st.write(
            "Enter the objects you want to locate in the sidebar."
        )

    generate_button = st.button(
        "Generate Analysis",
        type="primary",
        use_container_width=True
    )

    if generate_button:
        if (
            analysis_mode == "Find Specific Objects"
            and not object_prompt.strip()
        ):
            st.warning(
                "Enter at least one object name before continuing."
            )

        else:
            try:
                with st.spinner("Loading AI model and analyzing image..."):
                    processor, model = load_model()

                    result = generate_analysis(
                        image=image,
                        mode=analysis_mode,
                        processor=processor,
                        model=model,
                        object_prompt=object_prompt
                    )

                st.session_state["analysis_result"] = result
                st.session_state["analysis_mode"] = analysis_mode
                st.session_state["original_image"] = image.copy()

            except Exception as error:
                st.error(f"Analysis failed: {error}")


if "analysis_result" in st.session_state:
    result = st.session_state["analysis_result"]
    result_mode = st.session_state["analysis_mode"]

    st.divider()
    st.header("Analysis Results")

    if result_mode in [
        "Object Detection",
        "Find Specific Objects"
    ]:
        if isinstance(result, dict):
            labels = result.get("labels", [])
            boxes = result.get("bboxes", [])

            if labels and boxes:
                annotated_image = draw_detections(
                    st.session_state["original_image"],
                    result,
                    box_color,
                    box_width
                )

                result_col, details_col = st.columns([3, 2])

                with result_col:
                    st.subheader("Detected Objects")
                    st.image(
                        annotated_image,
                        use_container_width=True
                    )

                    st.download_button(
                        "Download Annotated Image",
                        data=image_to_bytes(annotated_image),
                        file_name="detected_objects.png",
                        mime="image/png",
                        use_container_width=True
                    )

                with details_col:
                    st.subheader("Detection Summary")
                    st.metric("Objects Detected", len(labels))

                    for index, (label, box) in enumerate(
                        zip(labels, boxes),
                        start=1
                    ):
                        st.write(
                            f"**{index}. {label}**"
                        )

                        st.caption(
                            "Box: " + ", ".join(
                                str(round(float(value), 1))
                                for value in box
                            )
                        )

            else:
                st.warning(
                    "The model did not return any object detections. "
                    "Try another image or use Find Specific Objects."
                )

        else:
            st.write(result)

    else:
        st.subheader("Generated Caption")
        st.write(result)

    text_result = results_to_text(result)

    st.download_button(
        "Download Text Results",
        data=text_result,
        file_name="image_analysis.txt",
        mime="text/plain",
        use_container_width=True
    )
