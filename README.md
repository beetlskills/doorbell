Arduino Setting
 
1. Go to Arduino IDE's File/Preferences menu. Enter the following Additional boards manager URLs: 

https://dl.espressif.com/dl/package_esp32_index.json

2. Go to Tools/Board/Boards Manager menu. Search for ESP32. Install board support package for ESP32 from Espressif Systems

3. Go to Tools/Board/esp32 and select ESP32 Wrover Module

4. Go to Tools/Port and select the COM port that ESP32 is connected to

5. Open doorbell.ino from File/Open

6. Click on Upload button to compile, upload and run.

7. Once running, select Tools/Serial Monitor. 

8. Detection is done every 5 seconds or so. If the scpore is >200, it will be deemed as face detected.

First commit

