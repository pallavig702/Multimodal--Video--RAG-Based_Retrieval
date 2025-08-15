# Multimodal RAG for Video Frames

This repository implements a **Multimodal Retrieval-Augmented Generation (RAG)** pipeline for video data.

**High-level workflow:**
1. **Extract Frames** → Split videos into frames for indexing.
2. **Index Frames in ChromaDB** → Store frame embeddings + metadata.
3. **Retrieve Relevant Frames** → Given a text query, find matching frames/videos.
4. **Generate Output**:
   - **Option A:** GPT-4o via LangChain (multimodal reasoning using API).
   - **Option B:** LLaVA-Next (open-source Vision-Language Model).

## 📂 Scripts Overview

| Step | Script | Purpose | Key Steps |
|------|--------|---------|-----------|
| 1 | **`Extract_VideoFrames_ChromaDB.py`** | Prepare data + index | (Optional) Extract frames → Initialize ChromaDB collection (OpenCLIP) → Add frames with metadata linking to source videos |
| 2A | **`RAG_Based_ImageRetrieval_From_Video.py`** | Retrieval + GPT-4o | Query ChromaDB → Select representative frame(s) + related image/text → Package into LangChain multimodal prompt → Call GPT-4o → Display results |
| 2B | **`With_LLaVa_Next.py`** | Retrieval + LLaVA | Query ChromaDB → Send retrieved frame(s) to LLaVA-Next (image or video variant) → Generate scene descriptions |
| — | **`Real3.py`** | Experimental glue | Combine or test retrieval + generation flows; can be adapted for custom pipelines |

## 🛠 Requirements - requirements.txt

## 📁 Folder Paths
Default paths in scripts: <br>
**ChromaDB storage** → RagBasedVideoAnalysis/mm_vdb<br>
**Videos (input)** → /home/pgupt60/scripts/CPU_ConvertedVideos/subset/<br>
**Frames (output)** → RagBasedVideoAnalysis/StockVideos-CC0-frames/<br>


## ▶️ Step-by-Step Usage
#### Step 1 — Extract Frames + Build the Index
Run this once to create the ChromaDB index.
```bash
python Extract_VideoFrames_ChromaDB.py
```

##### If you need to extract frames from .mp4:
* Uncomment the extract_frames(...) section in the script.<br>
* Set:<br>
   * video_folder_path → folder with .mp4 files<br>
   * output_folder_path → folder to save extracted frames<br>
* Script actions:<br>
   * Creates a PersistentClient at RagBasedVideoAnalysis/mm_vdb<br>
   * Creates a video_collection with OpenCLIP embeddings<br>
* Adds each frame with metadata: {'video_uri': <path_to_video>}<br>

#### Step 2A — Retrieval + GPT-4o (API Path)
```bash
export OPENAI_API_KEY=your_key_here
python RAG_Based_ImageRetrieval_From_Video.py
```
* Retrieves the most relevant frames for your query from ChromaDB.
   * Packages:
      User query
      Retrieved text (if available)
      Base64-encoded images (frame + related image)
* Sends them to GPT-4o using langchain_openai.ChatOpenAI.
* Displays:
   * The source video for the retrieved frame
   * GPT-4o’s multimodal reasoning output
 
#### Step 2B - Retrieval + Generation via LLaVA-NeXT (open-source path)
```bash
python With_LLaVa_Next.py
```
* Loads LLaVA-Next with:
      * 4-bit quantization (bitsandbytes)
      * FP16 computation
* Sends retrieved frame(s) to LLaVA-Next for local generation.
* Always open images with PIL:<br>
```bash
from PIL import Image as PILImage
image = PILImage.open(frame).convert("RGB")
```

##### Model usage tips:
* Video variant (LlavaNextVideoProcessor + LlavaNextVideoForConditionalGeneration): pass a sequence of frames (clip).
* Image variant (LlavaNextProcessor + LlavaNextForConditionalGeneration): pass a single frame.

Always open images with PIL:
GPT-4o’s multimodal reasoning output
Update these inside each script (video_dir, frames_dir, path=) to match your system.
## 🧠 Extra Inference: Different kind of vector databases <br>
![VectorDatabases](https://github.com/pallavig702/MultimodalRAG-Based_Retrieval/blob/main/Images/Vectordb.png)
## 🧠 Extra Inference: How RAG Works Differently for Text vs. Audio/Image Files <br>
Yes right! RAG behaves differently for text compared to audio, images, or other large files. Here’s why:<br>

### 1️⃣ How RAG Works for Text<br>
✅ Stores Text Data as Embeddings<br>
✅ Retrieves Relevant Text Passages<br>
✅ Uses Retrieved Text to Augment Response<br>
✅ Returns Processed/Generated Text (not just a reference)<br>

#### 🔹 Example: Text-based RAG in Action<br>
A user asks:<br>
➝ "What is Einstein’s theory of relativity?"<br>
RAG searches a vector database of text documents.<br>
#### It retrieves a relevant passage:<br>
➝ "Einstein’s theory of relativity consists of special and general relativity, describing the relationship between space, time, and gravity."
#### The model incorporates this retrieved text into its generated response.<br>
#### 🔑 Key Difference:<br>
For text, RAG doesn’t just return a reference; it actually feeds the retrieved text into the model so it can generate a response based on the information.<br>

### 2️⃣ How RAG Works for Audio, Images, or Large Files<br>
❌ Does NOT store raw audio/images<br>
✅ Stores only Embeddings + Metadata (paths, descriptions, timestamps, etc.)<br>
✅ Retrieves File Paths or Metadata<br>
✅ Returns Links or Plays the Files (Instead of Using Content for Generation)<br>

#### 🔹 Example: Audio-based RAG in Action<br>
A user asks:<br>
➝ "Find me a speech about climate change."<br>
#### RAG searches a vector database of audio embeddings + metadata.<br>
It retrieves metadata for an audio file:<br>
➝ speech_climate.wav (Path: /audio/speech_climate.wav)<br>
#### The system returns a reference to the file or allows playback.<br>
#### 🔑 Key Difference:<br>
For audio, images, or large files, RAG does not use the actual content to generate a response—it just retrieves references to the files.

### Why the Difference?<br>
Text is small & easy to process → It can be directly used in the model’s response.<br>
Audio, images, and videos are large → They are stored externally, and RAG just fetches references.<br>
Text-based RAG augments generation, while audio/image RAG optimizes retrieval.<br>
#### Final Takeaways<br>
✅ Text RAG: Retrieves and incorporates text directly into responses.<br>
✅ Audio/Image RAG: Retrieves file paths or metadata, not the actual content.<br>
✅ RAG Optimizes Retrieval, but Its Role Changes Based on the Data Type.<br>
