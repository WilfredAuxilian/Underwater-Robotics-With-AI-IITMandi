1) sudo apt update && sudo apt upgrade -y

2) sudo apt install -y \
   gstreamer1.0-tools \
   gstreamer1.0-plugins-base \
   gstreamer1.0-plugins-good \
   gstreamer1.0-plugins-bad \
   gstreamer1.0-plugins-ugly \
   gstreamer1.0-libav \
   gstreamer1.0-plugins-rtp \
   gstreamer1.0-plugins-base-apps

3) To test whether GStreamer is working in Ubuntu 22
   gst-launch-1.0 videotestsrc ! autovideosink 

4) to check ip address of ubuntu 22
   ip a  

5) Your BlueOS device is usually on 192.168.2.x, But your Ubuntu VM is on 192.168.98.x, which is not the same subnet where they currently   
   cannot talk directly. So change the VMware Network Adapter to "Bridged".
   Right-click your VM → Settings → Network Adapter.
   Select: ✅ Bridged (Connected directly to the physical network).
   Check the box "Replicate physical network connection state".
   Start the VM again.

6) Windows Sender Terminal Command,
   gst-launch-1.0 -v ksvideosrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! rtph264pay ! udpsink host=  
   172.16.32.73 port=5600   

7) Ubuntu Receiver Command,
   gst-launch-1.0 -v udpsrc port=5600 caps="application/x-rtp, encoding-name=H264, payload=96" ! \rtph264depay ! avdec_h264 videoconvert !     
   autovideosink  

8) You could see the webcam feed of your WindowsOs in ubuntuOS.

