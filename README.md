# Structured Data Extraction from Segmented Historical Records using Gemini + Vision OCR

This project extracts both textual and positional (coordinate) data from segmented historical document images, such as British military records.  
It combines Google Vision OCR for bounding box detection and Gemini API for detailed text and entity extraction, all with intelligent key rotation and layout classification.

---

## 🔍 Features

- 🖼️ **Image Segmentation**: Automatically splits large multi-column images into 3 logical segments to handle LLM token limits.
- 🧠 **Image Classifier**: Differentiates between segmented and unsegmented image layouts to guide preprocessing.
- 🧾 **Coordinate Extraction**: Uses Google Vision OCR to get bounding box coordinates for each record on the page.
- 🔁 **Gemini API Inference**: Sends segmented OCR text to Gemini for high-quality entity extraction.
- 🔑 **Multi-API Key Rotation**: Prevents timeouts by rotating through multiple Gemini API keys during batch processing.
- 🧹 **Post-Processing**: Cleans, validates, and structures the output; results are saved in pipe-delimited `.txt` files.

---

## 🛠️ Workflow Overview

1. **Preprocess Images**  
   - Classify layout (segmented vs. full-page)
   - Segment into columns if needed

2. **Extract OCR Text + Coordinates**  
   - Use **Google Vision OCR** to extract text and bounding boxes for each record.

3. **Inference with Gemini API**  
   - For each segment, use a rotated API key to call the Gemini model.
   - Extract named entities or structured fields (e.g., Name, Rank, Date).

4. **Post-Processing**  
   - Align Gemini outputs with OCR coordinates.
   - Save results in a pipe (`|`) delimited `.txt` file.

---

## 🚀 Example Code Snippet

```python
for idx, image_path in enumerate(image_list):
    key = key_pool[idx % len(key_pool)]
    segments = segment_image_if_needed(image_path)
    ocr_data = run_google_vision_ocr(image_path)

    for segment in segments:
        response = gemini_infer(segment, key)
        record_data = align_with_coordinates(response, ocr_data)
        all_results.append(record_data)

save_as_pipe_delimited(all_results, "output.txt")
