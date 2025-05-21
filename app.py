import sys
import argparse
import whisper_at as whisper
import time
import os
from collections import defaultdict

# Define known vocal and instrumental tags
VOCAL_TAGS = {
    "Singing", "Speech", "Choir", "Female singing", "Male singing",
    "Chant", "Yodeling", "Shout", "Bellow", "Rapping", "Narration",
    "Child singing", "Vocal music", "Opera", "A capella", "Voice",
    "Male speech, man speaking", "Female speech, woman speaking",
    "Child speech, kid speaking", "Conversation", "Narration, monologue", 
    "Babbling", "Speech synthesizer", "Whoop", "Yell", "Battle cry",
    "Children shouting", "Screaming", "Whispering", "Mantra",
    "Synthetic singing", "Humming", "Whistling", "Beatboxing",
    "Gospel music", "Lullaby", "Groan", "Grunt"
}

# Definitive speech tags that guarantee vocal classification
DEFINITIVE_SPEECH_TAGS = {
     "Male speech, man speaking", "Female speech, woman speaking",
    "Child speech, kid speaking", "Conversation", "Narration, monologue"
}

INSTRUMENTAL_TAGS = {
    "Piano", "Electric piano", "Keyboard (musical)", "Musical instrument",
    "Synthesizer", "Organ", "New-age music", "Electronic organ", 
    "Christmas music", "Rhythm and blues", "Independent music", "Soundtrack music", 
    "Pop music", "Jazz", "Soul music", "Christian music", "Instrumental music", 
    "Harpsichord", "Guitar", "Bass guitar", "Drums", "Violin", "Trumpet", 
    "Flute", "Saxophone", "Plucked string instrument", "Electric guitar", 
    "Acoustic guitar", "Steel guitar, slide guitar", "Banjo", "Sitar", "Mandolin", 
    "Ukulele", "Hammond organ", "Sampler", "Percussion", "Drum kit", 
    "Drum machine", "Drum", "Snare drum", "Bass drum", "Timpani", "Tabla", 
    "Cymbal", "Hi-hat", "Tambourine", "Marimba, xylophone", 
    "Vibraphone", "Orchestra", "Brass instrument",
    "French horn", "Trombone", "Bowed string instrument", "String section", 
    "Violin, fiddle", "Cello", "Double bass", "Wind instrument, woodwind instrument", 
    "Clarinet", "Harp", "Harmonica", "Accordion", 
    "Rock music", "Heavy metal", "Punk rock", "Grunge", "Progressive rock", 
    "Rock and roll", "Psychedelic rock", "Reggae", "Country", "Swing music", 
    "Bluegrass", "Funk", "Folk music", "Middle Eastern music", "Disco", 
    "Classical music", "Electronic music", "House music", "Techno", "Dubstep", 
    "Drum and bass", "Electronica", "Electronic dance music", "Ambient music", 
    "Trance music", "Music of Latin America", "Salsa music", "Flamenco", "Blues", 
    "Music for children", "Music of Africa", "Afrobeat", "Music of Asia", 
    "Carnatic music", "Music of Bollywood", "Ska", "Traditional music", 
    "Song", "Theme music", "Video game music", "Dance music", "Wedding music", 
    "Happy music", "Funny music", "Sad music", "Tender music", "Exciting music", 
    "Angry music", "Scary music"
}

def classify_audio(top_tags):
    # Check for definitive speech tags first - if any are present, it's definitely vocal
    has_definitive_speech = any(tag in DEFINITIVE_SPEECH_TAGS for tag in top_tags)
    
    if has_definitive_speech:
        return "Vocal Audio"
    
    # Regular classification logic as fallback
    has_vocal = any(tag in VOCAL_TAGS for tag in top_tags)
    has_instrumental = any(tag in INSTRUMENTAL_TAGS for tag in top_tags)

    if has_vocal and not has_instrumental:
        return "Vocal Audio"
    elif has_instrumental and not has_vocal:
        return "Instrumental Audio"
    elif has_vocal and has_instrumental:
        return "Music"
    else:
        return "Unknown"

def process_audio(audio_path, model_size="small"):
    """Process audio file with Whisper-AT for transcription and audio tagging."""
    print(f"\nProcessing {audio_path}...")

    audio_tagging_time_resolution = 4.8

    print(f"Loading Whisper-AT model ({model_size})...")
    start_time = time.time()
    model = whisper.load_model(model_size)
    print(f"Model loaded in {time.time() - start_time:.2f} seconds.")

    if not os.path.exists(audio_path):
        print(f"Error: File '{audio_path}' not found.")
        return

    result = model.transcribe(audio_path, at_time_res=audio_tagging_time_resolution)

    #print("\nTranscription:")
    #print(result["text"])

    #print("\nAudio Tags (with confidence scores):")
    audio_tag_result = whisper.parse_at_label(
        result,
        language='en',
        top_k=15,
        p_threshold=-5
    )

    all_tags_set = set()
    tag_freq = defaultdict(int)

    for segment in audio_tag_result:
        start = segment['time']['start']
        end = segment['time']['end']
        #print(f"Time {start}-{end}s: ", end="")
        tags = [f"{tag[0]} ({tag[1]:.2f})" for tag in segment['audio tags']]
        #print(", ".join(tags))

        # Update tag set and frequency
        for tag, score in segment['audio tags']:
            all_tags_set.add(tag)
            tag_freq[tag] += 1

    # Find top tags (those that appear more than once)
    top_tags = [tag for tag, freq in tag_freq.items() if freq > 1]

    print("\nTop Tags (appeared more than once):")
    for tag in top_tags:
        print(f"- {tag}")

    # Check for definitive speech tags (even if they only appear once)
    all_detected_tags = list(all_tags_set)
    speech_tags = [tag for tag in all_detected_tags if tag in DEFINITIVE_SPEECH_TAGS]
    
    if speech_tags:
        #print("\nDefinitive Speech Tags Detected:")
        #for tag in speech_tags:
            #print(f"- {tag}")
        # Add speech tags to top_tags to ensure they're considered in classification
        for tag in speech_tags:
            if tag not in top_tags:
                top_tags.append(tag)

    classification = classify_audio(top_tags)
    print(f"\nClassification: {classification}")

def main():
    parser = argparse.ArgumentParser(description='Process audio files with Whisper-AT')
    parser.add_argument('audio_file', type=str, help='Path to the audio file')
    parser.add_argument('--model', type=str, default='small', 
                        choices=['tiny', 'base', 'small', 'medium', 'large-v1'],
                        help='Whisper model size (default: small)')
    args = parser.parse_args()

    process_audio(args.audio_file, args.model)

if __name__ == "__main__":
    main()
