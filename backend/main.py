import glob
import json
import pathlib
import pickle
import shutil
import subprocess
import time
import uuid
import boto3
import cv2
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import modal
import numpy as np
from pydantic import BaseModel
import os
from google import genai
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple

import pysubs2
from tqdm import tqdm
import whisperx


class ProcessVideoRequest(BaseModel):
    s3_key: str


image = (modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install(["ffmpeg", "libgl1-mesa-glx", "wget", "libcudnn8", "libcudnn8-dev"])
    .pip_install_from_requirements("requirements.txt")
    .run_commands(["mkdir -p /usr/share/fonts/truetype/custom",
                   "wget -O /usr/share/fonts/truetype/custom/Anton-Regular.ttf https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf",
                   "fc-cache -f -v"])
    .add_local_dir("asd", "/asd", copy=True)
    .run_commands([
        "sed -i 's/np.int/int/g' /asd/model/faceDetector/s3fd/box_utils.py",
        "sed -i 's/np.float/np.float64/g' /asd/utils/get_ava_active_speaker_performance.py"
    ]))

app = modal.App("ai-podcast-clipper", image=image)

volume = modal.Volume.from_name(
    "ai-podcast-clipper-model-cache", create_if_missing=True
)

mount_path = "/root/.cache/torch"

auth_scheme = HTTPBearer()

def optimize_transcript_format(segments: List[Dict]) -> str:
    """Optimize transcript format to reduce token usage"""
    optimized_segments = []
    for segment in segments:
        # Only include essential fields and round timestamps to 2 decimal places
        optimized_segment = {
            "s": round(segment.get("start", 0), 2),  # start -> s
            "e": round(segment.get("end", 0), 2),    # end -> e  
            "w": segment.get("word", "").strip()      # word -> w
        }
        if optimized_segment["w"]:  # Only include non-empty words
            optimized_segments.append(optimized_segment)
    return json.dumps(optimized_segments, separators=(',', ':'))

def create_transcript_chunks(segments: List[Dict], chunk_duration: float = 300, overlap_duration: float = 30) -> List[Tuple[float, float, List[Dict]]]:
    """
    Create overlapping chunks from transcript segments
    
    Args:
        segments: List of transcript segments with start, end, word
        chunk_duration: Duration of each chunk in seconds (default: 5 minutes)
        overlap_duration: Overlap between chunks in seconds (default: 30 seconds)
    
    Returns:
        List of tuples: (chunk_start, chunk_end, chunk_segments)
    """
    if not segments:
        return []
    
    # Find total duration
    total_duration = max(seg.get("end", 0) for seg in segments if seg.get("end"))
    
    chunks = []
    current_start = 0
    
    while current_start < total_duration:
        chunk_end = min(current_start + chunk_duration, total_duration)
        
        # Get segments for this chunk (including overlap)
        chunk_segments = [
            seg for seg in segments
            if seg.get("start") is not None 
            and seg.get("end") is not None
            and seg.get("end") > current_start 
            and seg.get("start") < chunk_end
        ]
        
        if chunk_segments:
            chunks.append((current_start, chunk_end, chunk_segments))
        
        # Move to next chunk (with overlap)
        current_start += chunk_duration - overlap_duration
        
        # If we're close to the end, make this the last chunk
        if current_start + chunk_duration >= total_duration:
            break
    
    return chunks

def create_vertical_video(tracks,scores,pyframes_path, pyavi_path, audio_path , output_path , framerate=25):
    import ffmpegcv
    
    target_width= 1080
    target_height=1920
    
    flist = glob.glob(os.path.join(pyframes_path,"*.jpg"))
    flist.sort()
    
    faces =[[]  for _ in range(len(flist))]
    
    for tidx, track in enumerate(tracks):
        score_array = scores[tidx]
        for fidx , frame  in enumerate(track["track"]["frame"].tolist()):
            slice_start = max(fidx - 30, 0)
            slice_end = min(fidx + 30,len(score_array))
            score_slice = score_array[slice_start:slice_end]
            avg_score= float(np.mean(score_slice)
                             if(len(score_slice) > 0) else 0)

            faces[frame].append(
                {'track':tidx, 'score':avg_score,'s':track['proc_track']["s"][fidx],'x':track['proc_track']["x"][fidx],'y':track['proc_track']["y"][fidx]})
            
    temp_video_path = os.path.join(pyavi_path,"video_only.mp4")
    
    vout = None
    for fidx , fname  in tqdm(enumerate(flist),total=len(flist),desc="Creating vertical video"):
        img = cv2.imread(fname)
        if img is None:
            continue
        
        current_faces = faces[fidx]
        
        max_score_face = max(
            current_faces , key = lambda face: face['score']) if current_faces else None
        
        if max_score_face and max_score_face['score'] < 0:
            max_score_face = None
        
        if vout is None:
            vout = ffmpegcv.VideoWriterNV(
                file = temp_video_path,
                codec = None,
                fps = framerate,
                resize= (target_width, target_height)
            )
        
        if max_score_face :
            mode="crop"
        else:
            mode= "resize"
        
        if mode == "resize":
            scale = target_width / img.shape[1]
            resized_height = int(img.shape[0]* scale)
            resized_img = cv2.resize(img, (target_width, resized_height),interpolation=cv2.INTER_AREA)
            
            scale_for_bg = max(
                target_width / img.shape[1] , target_height / img.shape[0]
            )
            
            bg_width = int(img.shape[1] * scale_for_bg)
            bg_height = int(img.shape[0] * scale_for_bg)
            
            blurred_background = cv2.resize(img , (bg_width , bg_height))
            blurred_background = cv2.GaussianBlur(blurred_background, (121,121), 0)
            
            crop_x  = (bg_width - target_width) // 2
            # use bg_height for Y calculation (bugfix) and ensure slice matches target
            crop_y = (bg_height - target_height) // 2

            blurred_background = blurred_background[crop_y:crop_y + target_height,
                                                    crop_x: crop_x + target_width]

            # If for any reason the resized image is taller than target, center-crop it to fit.
            if resized_height > target_height:
                start_y = (resized_height - target_height) // 2
                resized_img = resized_img[start_y:start_y + target_height, :]
                resized_height = resized_img.shape[0]

            center_y = max(0, (target_height - resized_height) // 2)
            blurred_background[center_y: center_y + resized_height, :] = resized_img
            
            vout.write(blurred_background)
        
        elif mode == "crop":
            scale =  target_height / img.shape[0]
            resized_image = cv2.resize(img, None, fx = scale, fy= scale , interpolation= cv2.INTER_AREA)
            frame_width = resized_image.shape[1]
            
            center_x = int(
                max_score_face["x"] * scale if max_score_face else frame_width // 2)
            
            top_x = max(min(center_x - target_width //2 ,
                            frame_width - target_width), 0)
            
            image_cropped = resized_image[0:target_height,
                                          top_x: top_x +  target_width]
            
            vout.write(image_cropped)
            
    if vout:
        vout.release()
        
        ffmpeg_command = (f"ffmpeg -y -i {temp_video_path} -i {audio_path} "
                          f"-c:v h264 -preset fast -crf 23 -c:a aac -b:a 128k "
                          f"{output_path}")
        
        subprocess.run(ffmpeg_command, shell = True , check =True , text = True)

def create_subtitles_with_ffmpeg(transcript_segments: list, clip_start: float, clip_end: float, clip_video_path: str, output_path: str, max_words: int = 5):
    temp_dir = os.path.dirname(output_path)
    subtitle_path = os.path.join(temp_dir, "temp_subtitles.ass")

    clip_segments = [segment for segment in transcript_segments
                     if segment.get("start") is not None
                     and segment.get("end") is not None
                     and segment.get("end") > clip_start
                     and segment.get("start") < clip_end
                     ]

    subtitles = []
    current_words = []
    current_start = None
    current_end = None

    for segment in clip_segments:
        word = segment.get("word", "").strip()
        seg_start = segment.get("start")
        seg_end = segment.get("end")

        if not word or seg_start is None or seg_end is None:
            continue

        start_rel = max(0.0, seg_start - clip_start)
        end_rel = max(0.0, seg_end - clip_start)

        if end_rel <= 0:
            continue

        if not current_words:
            current_start = start_rel
            current_end = end_rel
            current_words = [word]
        elif len(current_words) >= max_words:
            subtitles.append(
                (current_start, current_end, ' '.join(current_words)))
            current_words = [word]
            current_start = start_rel
            current_end = end_rel
        else:
            current_words.append(word)
            current_end = end_rel

    if current_words:
        subtitles.append(
            (current_start, current_end, ' '.join(current_words)))

    subs = pysubs2.SSAFile()

    subs.info["WrapStyle"] = 0
    subs.info["ScaledBorderAndShadow"] = "yes"
    subs.info["PlayResX"] = 1080
    subs.info["PlayResY"] = 1920
    subs.info["ScriptType"] = "v4.00+"

    style_name = "Default"
    new_style = pysubs2.SSAStyle()
    new_style.fontname = "Anton"
    new_style.fontsize = 140
    new_style.primarycolor = pysubs2.Color(255, 255, 255)
    new_style.outline = 2.0
    new_style.shadow = 2.0
    new_style.shadowcolor = pysubs2.Color(0, 0, 0, 128)
    new_style.alignment = 2
    new_style.marginl = 50
    new_style.marginr = 50
    new_style.marginv = 50
    new_style.spacing = 0.0

    subs.styles[style_name] = new_style

    for i, (start, end, text) in enumerate(subtitles):
        start_time = pysubs2.make_time(s=start)
        end_time = pysubs2.make_time(s=end)
        line = pysubs2.SSAEvent(
            start=start_time, end=end_time, text=text, style=style_name)
        subs.events.append(line)

    subs.save(subtitle_path)

    ffmpeg_cmd = (f"ffmpeg -y -i {clip_video_path} -vf \"ass={subtitle_path}\" "
                  f"-c:v h264 -preset fast -crf 23 {output_path}")

    subprocess.run(ffmpeg_cmd, shell=True, check=True)
   

def process_clip(base_dir: str, original_video_path: str, s3_key: str, start_time: float, end_time: float, clip_index: int, transcript_segments: list):
    clip_name = f"clip_{clip_index}"
    # uuid/originalVideo.mp4

    s3_key_dir = os.path.dirname(s3_key)
    output_s3_key = f"{s3_key_dir}/{clip_name}.mp4"
    print(f"Output S3 key: {output_s3_key}")
    
    clip_dir = base_dir / clip_name
    clip_dir.mkdir(parents= True, exist_ok = True)
    
    #segment path : original clip from start to end
    clip_segment_path = clip_dir / f"{clip_name}_segment.mp4"
    vertical_mp4_path = clip_dir / "pyavi" / "video_out_vertical.mp4"
    subtitle_output_path = clip_dir / "pyavi" / "video_with_subtitles.mp4"

    (clip_dir / "pywork").mkdir(exist_ok= True)
    pyframes_path = clip_dir / "pyframes"
    pyavi_path = clip_dir / "pyavi"
    audio_path = clip_dir / "pyavi" / "audio.wav"
    
    pyframes_path.mkdir(exist_ok= True)
    pyavi_path.mkdir(exist_ok= True)
    
    duration = end_time - start_time
    cut_command = (f"ffmpeg -i {original_video_path} -ss {start_time} -t {duration} "
                   f"{clip_segment_path}")
    subprocess.run(cut_command, shell= True, check=True, capture_output=True,text=True)
    
    extract_cmd = f"ffmpeg -i {clip_segment_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
    subprocess.run(extract_cmd, shell=True, check=True, capture_output=True)

    shutil.copy(clip_segment_path, base_dir / f"{clip_name}.mp4")
    
    columbia_command = (f"python Columbia_test.py --videoName {clip_name} "
                        f"--videoFolder {str(base_dir)} "
                        f"--pretrainModel weight/finetuning_TalkSet.model")
    
    columbia_start_time = time.time()
    subprocess.run(columbia_command,cwd="/asd",shell=True)
    columbia_end_time = time.time()
    print(
        f"Columbia script completed in {columbia_end_time - columbia_start_time: .2f} seconds")
    
    tracks_path = clip_dir / "pywork" / "tracks.pckl"
    scores_path = clip_dir / "pywork" / "scores.pckl"
    if not tracks_path.exists() or not scores_path.exists():
        raise FileNotFoundError("Tracks or Scores not found for clip")
    
    with open(tracks_path,"rb") as f:
        tracks = pickle.load(f)
        
    with open(scores_path,"rb") as f:
        scores = pickle.load(f)
    
    cvv_start_time = time.time()
    create_vertical_video(
        tracks , scores , pyframes_path, pyavi_path, audio_path, vertical_mp4_path
    )
    cvv_end_time = time.time()
    
    print(f"Clip {clip_index} vertical video creation time: {cvv_end_time - cvv_start_time: .2f} seconds")
    
    create_subtitles_with_ffmpeg(transcript_segments, start_time,
                                 end_time, vertical_mp4_path, subtitle_output_path, max_words=5)
    
    s3_client = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "ap-south-1"))
    s3_client.upload_file(subtitle_output_path, os.environ.get("S3_BUCKET_NAME", "pd-vd-clips"), output_s3_key)
    
    
@app.cls(gpu="L40S",timeout=1800,retries=0,scaledown_window=20 , secrets=[modal.Secret.from_name("ai-podcast-clipper-secret")] , volumes={mount_path: volume})
class AiPodcastClipper:
    @modal.enter()
    def load_model(self):
        print("Loading models")

        # import whisperx here so the Modal CLI doesn't need whisperx installed locally
        import whisperx

        self.whisperx_model = whisperx.load_model(
            "large-v2", device="cuda", compute_type="float16")

        self.alignment_model, self.metadata = whisperx.load_align_model(
            language_code="en",
            device="cuda"
        )

        print("Transcription models loaded...")
        
        print("Creating gemini client... ")
        self.gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        print("Created gemini client... ")
    
    def transcribe_video(self, base_dir: str, video_path: str) -> str:
        # import whisperx here so this file can be imported locally without whisperx
        import whisperx

        audio_path = base_dir / "audio.wav"
        extract_cmd = f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
        subprocess.run(extract_cmd, shell=True, check=True, capture_output=True)

        print("Starting transcriptions with whisperX...")
        start_time = time.time()

        audio = whisperx.load_audio(str(audio_path))
        result = self.whisperx_model.transcribe(audio, batch_size=8)

        result = whisperx.align(
            result["segments"],
            self.alignment_model,
            self.metadata,
            audio,
            device="cuda",
            return_char_alignments=False
        )

        duration = time.time() - start_time
        print("Transcription and alignment took " + str(duration) + " seconds")
        
        segments = []

        # whisperx/alignment returns a `word_segments` list when available.
        # Guard for presence and iterate the correct key name to avoid KeyError.
        if "word_segments" in result and isinstance(result["word_segments"], (list, tuple)):
            for word_segment in result["word_segments"]:
                # Each word_segment is expected to contain start/end/word fields.
                segments.append({
                    "start": word_segment.get("start"),
                    "end": word_segment.get("end"),
                    "word": word_segment.get("word"),
                })

        return json.dumps(segments)
    
    def identify_moments_chunk(self, transcript_chunk: List[Dict], chunk_start: float, chunk_end: float) -> List[Dict]:
        """Process a single chunk of transcript to identify moments"""
        try:
            optimized_transcript = optimize_transcript_format(transcript_chunk)
            
            print(f"Processing chunk {chunk_start}-{chunk_end}s")
            
            prompt = f"""
This is a podcast video transcript chunk from {chunk_start}s to {chunk_end}s. The transcript uses optimized format where each word has: s=start_time, e=end_time, w=word.

I am looking to create clips between a minimum of 45 and maximum of 60 seconds long. The clip should never exceed 60 seconds.

Your task is to find and extract stories, or question and their corresponding answers from the transcript.
Each clip should begin with the question and conclude with the answer.
It is acceptable for the clip to include a few additional sentences before a question if it aids in contextualizing the question.

Please adhere to the following rules:
- Ensure that clips do not overlap with one another.
- Start and end timestamps of the clips should align perfectly with the sentence boundaries in the transcript.
- Only use the start and end timestamps provided in the input (s/e fields). Modifying timestamps is not allowed.
- Format the output as a list of JSON objects, each representing a clip with 'start' and 'end' timestamps: [{{"start": seconds, "end": seconds}}, ...clip2, clip3]. The output should always be readable by the python json.loads function.
- Aim to generate longer clips between 40-60 seconds, and ensure to include as much content from the context as viable.

Avoid including:
- Moments of greeting, thanking, or saying goodbye.
- Non-question and answer interactions.

If there are no valid clips to extract, the output should be an empty list [], in JSON format. Also readable by json.loads() in Python.

The transcript chunk is as follows: {optimized_transcript}"""

            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash", 
                contents=prompt
            )
            
            if not response or not response.text:
                print(f"Warning: Empty response for chunk {chunk_start}-{chunk_end}")
                return []
            
            # Clean and parse response
            cleaned_json = response.text.strip()
            if cleaned_json.startswith("```json"):
                cleaned_json = cleaned_json[len("```json"):].strip()
            if cleaned_json.endswith("```"):
                cleaned_json = cleaned_json[:-len("```")].strip()
            
            try:
                clips = json.loads(cleaned_json)
                if not isinstance(clips, list):
                    print(f"Error: Response is not a list for chunk {chunk_start}-{chunk_end}")
                    return []
                
                # Basic validation for clip structure
                valid_clips = []
                for clip in clips:
                    if (isinstance(clip, dict) and 
                        "start" in clip and "end" in clip and
                        isinstance(clip["start"], (int, float)) and
                        isinstance(clip["end"], (int, float)) and
                        clip["start"] < clip["end"]):
                        valid_clips.append(clip)
                    else:
                        print(f"Invalid clip structure discarded: {clip}")
                
                print(f"Found {len(valid_clips)} valid clips in chunk {chunk_start}-{chunk_end}")
                return valid_clips
                
            except json.JSONDecodeError as e:
                print(f"JSON decode error for chunk {chunk_start}-{chunk_end}: {e}")
                return []
                
        except Exception as e:
            print(f"Error processing chunk {chunk_start}-{chunk_end}: {e}")
            return []
    
    def identify_moments(self, transcript: List[Dict]) -> List[Dict]:
        """Main method to identify moments using chunking strategy"""
        if not transcript:
            return []
        
        # Create chunks with sliding windows
        chunks = create_transcript_chunks(transcript, chunk_duration=1200, overlap_duration=30)
        print(f"Created {len(chunks)} chunks for processing")
        
        if not chunks:
            return []
        
        # Process chunks in parallel
        all_clips = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:  # Limit concurrent API calls
            future_to_chunk = {
                executor.submit(self.identify_moments_chunk, chunk_segments, chunk_start, chunk_end): (chunk_start, chunk_end)
                for chunk_start, chunk_end, chunk_segments in chunks
            }
            
            for future in as_completed(future_to_chunk):
                chunk_start, chunk_end = future_to_chunk[future]
                try:
                    chunk_clips = future.result()
                    all_clips.extend(chunk_clips)
                    print(f"Processed chunk {chunk_start}-{chunk_end}: {len(chunk_clips)} clips")
                except Exception as exc:
                    print(f"Chunk {chunk_start}-{chunk_end} generated exception: {exc}")
        
        print(f"Total clips found: {len(all_clips)}")
        
        # Sort clips by start time and limit to prevent long processing
        all_clips = sorted(all_clips, key=lambda x: x.get('start', 0))
        
        return all_clips
    
    @modal.fastapi_endpoint(method="POST")
    def process_video(self, request: ProcessVideoRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        print("Processing Video" + request.s3_key)
        s3_key = request.s3_key

        if token.credentials != os.environ["AUTH_TOKEN"]:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid bearer token", headers={"WWW-Authenticate": "Bearer"})

        run_id = str(uuid.uuid4())
        base_dir = pathlib.Path("/tmp") / run_id
        base_dir.mkdir(parents=True, exist_ok=True)

        # download video file
        import boto3

        video_path = base_dir / "input.mp4"
        s3_client = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "ap-south-1"))
        s3_client.download_file(os.environ.get("S3_BUCKET_NAME", "pd-vd-clips"), s3_key, str(video_path))
        
        #1 transcription
        transcript_segments_json = self.transcribe_video(base_dir, video_path)
        transcript_segments = json.loads(transcript_segments_json)
        
        #2. identify moments for clips
        print("Identifying clip moments using chunked processing")
        clip_moments = self.identify_moments(transcript_segments)
        
        # Handle case where no moments are identified
        if not clip_moments:
            print("No moments identified, skipping clip processing")
            return {"message": "No clips were generated - no valid moments found"}
        
        print(f"Found {len(clip_moments)} clip moments")
        print(f"Processing first 3 clips to manage execution time")
        
        #3. process clips
        for index , moment in enumerate(clip_moments[:1]):
            print("Processing Clip" + str(index) + " from " + 
                  str(moment["start"]) + " to " + str(moment["end"]))
            process_clip(base_dir, video_path, s3_key, moment["start"], moment["end"], index, transcript_segments)
        
        if base_dir.exists():
            print("Cleaning up temp dir after " + str(base_dir))
            shutil.rmtree(base_dir, ignore_errors=True)
        
        return {
            "message": "Video processing completed successfully",
            "clips_found": len(clip_moments),
            "clips_processed": min(3, len(clip_moments)),
            "transcription_time": f"{transcript_segments_json is not None}"
        }
