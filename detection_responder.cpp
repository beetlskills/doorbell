/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <Arduino.h>

#include "detection_responder.h"
#include "image_provider.h"

// KBH #include "smtp_client.h"
#include "esp_timer.h"
#include "esp_camera.h"
#include "esp_log.h"
#include "img_converters.h"
/* min time (in ms) between two consecuting email notifications */
#define HOLD_TIME 5000

static int64_t elapsed_time = 0;
uint8_t *jpeg_image = NULL;
size_t jpeg_img_size = 0;
camera_fb_t* camera_fb;

static const char *TAG = "tf_responder";

void RespondToDetection(tflite::ErrorReporter* error_reporter,
                        uint8_t person_score, uint8_t no_person_score) {

  if (elapsed_time == 0) {
    elapsed_time = esp_timer_get_time();
  }
  char score_str[5];
  sprintf(score_str, "%d", person_score);
  TF_LITE_REPORT_ERROR(error_reporter, score_str);

  if (person_score > 200) {
    if (((esp_timer_get_time() - elapsed_time)/1000) >= HOLD_TIME) {

      camera_fb = (camera_fb_t*)image_provider_get_camera_fb();
      TF_LITE_REPORT_ERROR(error_reporter, "person detected");
      free(jpeg_image);
      bool ret = frame2jpg(camera_fb, 80,  &jpeg_image, &jpeg_img_size);
      if (ret != true) {
        TF_LITE_REPORT_ERROR(error_reporter,"jpeg compression failed");
      }
      elapsed_time = 0;
    }
  }
}