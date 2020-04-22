# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

include(FetchContent)

message(STATUS "*********************** In exteranl_opencv.cmake")

set(OPENCV_VERSION 4.3.0)

set(OPENCV_GIT_REPO_URL https://github.com/opencv/opencv.git)

FetchContent_Declare(
    ext_opencv
    GIT_REPOSITORY ${OPENCV_GIT_REPO_URL}
    GIT_TAG ${OPENCV_VERSION}
)

set(WITH_1394 NO)
set(WITH_JASPER NO)
set(WITH_OPENJPEG NO)
set(WITH_JPEG NO)
set(WITH_WEBP NO)
set(WITH_PNG NO)
set(WITH_TIFF NO)
set(WITH_V4L NO)
set(WITH_PROTOBUF NO)
set(WITH_OPENCL NO)
set(WITH_IMGCODEC_HDR NO)
set(WITH_IMGCODEC_PXM NO)
set(WITH_IMGCODEC_PFM NO)
set(WITH_IMGCODEC_SUNRASTER NO)
set(WITH_OPENEXR NO)
set(WITH_FFMPEG NO)
set(CV_TRACE NO)

set(BUILD_opencv_apps NO)
set(BUILD_opencv_js NO)
set(BUILD_DOCS NO)
set(BUILD_EXAMPLES NO)
set(BUILD_PACKAGE NO)
set(BUILD_PERF_TESTS NO)
set(BUILD_TESTS NO)
set(BUILD_WITH_DEBUG_INFO NO)

FetchContent_MakeAvailable(ext_opencv)
