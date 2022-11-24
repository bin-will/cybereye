 # CyberEye: Obtain Data from Virtual Desktop by Video

## 1.	Introduction

We present a new data transmission approach named ‘CyberEye’, that can extract data file precisely from VDI(Virtual Desktop Infrastructure) even the data has never left data center. 
The main idea is encoding data file to video, then playing it at virtual desktop while recording it at host PC, decode the recorded video at last, that can recover the original data file.

## 2.	Requirements

- Runtime Environment: Python>=3.0, opencv-python>=3.4, numpy>=1.14, we provide green executable version for Win 7 and Win 2000, packaged by Pyinstaller.
- Screen Recorder: Any screen recorder supporting drag recording region should be workable, we have test Quicktime on macOS and Apower(https://www.apowersoft.com/) on Windows.

## 3.	Usage

Only four steps to get target file out of virtual desktop, very simply and quietly. 
- (1)	Upload CyberEye to virtual desktop.
- (2)	Encode target file to video file. Open a CLI terminal and assign the target file, we can get the encoded video file `target_file_name.zip.mp4`:
```
python cybereye --input_file target_file_name.zip
```
- (3)	Play `target_file_name.zip.mp4` within virtual desktop while recording screen on host PC, remember to drag a record region to mask the video window, this step is important, please refer the demonstration video, then save the recorded video like `recorded_file.mp4`.
- (4)	Decode the recorded video to original target file, open a CLI terminal and assign the video file, we can decode the file and get the original target file `decoded_recorded_file.mp4`, it is not the original name, just rename it to `target_file_name.zip`, all done.
```
python cybereye --decode_file recorded_file.mp4
```
## 4.	Evaluation

### Environment: 
- Host PC: MacBook Pro, 2.2 GHz Intel Core 7, 16 GB DDR3, Intel Iris Pro 1536, macOS 11.6 with internet access, bandwidth 50Mb, locate at city A.
- Screen recorder in host PC: QuickTime Player 10.5(1086.4.2), preinstalled in macOS.
VDI Client installed in host PC: Citrix Workspace Version 21.12.0.32(2112), virtual desktop is  Windows Server 2008 R2 Enterprise(version 6.1.7601, sp1), with preinstalled Windows Media Player.
- VDI data center locates at city B, far away from city A. 
- CyberEye is developed with Python 3.7.3 with main dependencies on OpenCV 3.4.2 and Numpy 1.21.5, compiled executable version with Pyinstaller 5.3 supporting Win 2008 and Win 7. The original ZIP files are encoded by 5 FPS, repeat twice, and recorded video are default running on 60 FPS by QuickTime.

We evaluated different size of target files to verify the effectiveness and performance, note we only use ZIP file as target file because any files can be archived to ZIP. Note QuickTime will interrupt the last frame while writing video file, which cause fail when decoding the file, use FFMPEG application to fix it by the following command, and any other media tools would work also. About FFMPEG, refer: https://github.com/FFmpeg/FFmpeg. The easy fix command is: 

`ffmpeg -i recorded_video.mov -vcodec copy recorded_video_fix.mp4`

By the practices, we can validate the approximate linear relationship between ZIP file size and video size, also encoding and decoding time. For reliability reason, we suggest splitting large ZIP file to smaller files less than 1 MB.

### Table: Encoding and Decoding Performance

| ZIP File Size(KBytes)|	Encoding Time(Seconds) |	Encoded Video Size(KBytes) |	Encoded Video Length(Seconds) |	Recorded MOV Video Size(KBytes) |	Decode Time (Seconds) 
|--------|--------|--------|--------|-------- |--------|
| 128    | 2.231  | 37,182 |	67     |	140,752	|76.338  |
| 256    |	3.627  |	64,389	| 105    |	229,110	|159.342 |
| 513    |	6.049  |	118,789|	194    |	383,230	|293.832 |
| 1,025	 | 11.04  | 227,866| 372    |	829,678	|561.398 | 
| 2,051  |	21.149	| 445,825|	728    |	1,544,364 |	1,098.988 |
| 3,077	 | 32.757	| 663,943|	1,085	  | 2,172,970	|1,634.899 |
| 5,128	 | 51.072	| 1,099,703|	1,797	| 3,855,984	|2,699.382|

