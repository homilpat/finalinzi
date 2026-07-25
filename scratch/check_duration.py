import os

def check_mp3_duration(filepath):
    # Pure python MP3 duration parser
    # MP3 frames start with syncword: 11 bits set (0xFF and 3 bits of 0xE0/0xF0/0xD0/0xC0 depending on version/layer)
    # Let's read the file and find frame headers.
    with open(filepath, 'rb') as f:
        data = f.read()
    
    # We can search for frame headers
    idx = 0
    total_frames = 0
    duration = 0.0
    
    # MP3 version, layer, bitrate index, samplerate index, etc.
    bitrates_v1_l3 = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, -1]
    bitrates_v2_l3 = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, -1]
    samplerates_v1 = [44100, 48000, 32000, -1]
    samplerates_v2 = [22050, 24000, 16000, -1]
    samplerates_v25 = [11025, 12000, 8000, -1]
    
    while idx < len(data) - 4:
        # Look for syncword (12 bits)
        if data[idx] == 0xFF and (data[idx+1] & 0xE0) == 0xE0:
            # Parse header
            header = data[idx:idx+4]
            version = (header[1] & 0x18) >> 3 # 3=v1, 2=v2, 0=v2.5
            layer = (header[1] & 0x06) >> 1 # 1=Layer3
            bitrate_idx = (header[2] & 0xF0) >> 4
            samplerate_idx = (header[2] & 0x0C) >> 2
            padding = (header[2] & 0x02) >> 1
            
            # Decides tables
            if version == 3: # V1
                samplerate = samplerates_v1[samplerate_idx]
                bitrate = bitrates_v1_l3[bitrate_idx]
            elif version == 2: # V2
                samplerate = samplerates_v2[samplerate_idx]
                bitrate = bitrates_v2_l3[bitrate_idx]
            else: # V2.5
                samplerate = samplerates_v25[samplerate_idx]
                bitrate = bitrates_v2_l3[bitrate_idx]
                
            if samplerate > 0 and bitrate > 0:
                # Frame size formula for Layer 3: 144 * bitrate * 1000 / samplerate + padding
                frame_size = int(144 * bitrate * 1000 / samplerate) + padding
                # Duration of one Layer 3 frame: 1152 samples / samplerate
                frame_duration = 1152.0 / samplerate
                duration += frame_duration
                total_frames += 1
                idx += frame_size
                continue
        idx += 1
        
    print(f"{os.path.basename(filepath)}: {duration:.3f} seconds, {total_frames} frames")

check_mp3_duration('giukhaji/assets/tts/clap_seq_moca_k.mp3')
check_mp3_duration('giukhaji/assets/tts/clap_seq_k_moca.mp3')
