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

function(get_linux_info)
    execute_process(COMMAND cat /etc/os-release
        OUTPUT_VARIABLE ID_LONG_STRING)
    set(UBUNTU FALSE)
    set(REDHAT FALSE)
    string(REGEX MATCH "ID=\"*([a-z]+)" _ "${ID_LONG_STRING}")
    string(STRIP ${CMAKE_MATCH_1} TYPE)
    set(LINUX_TYPE ${TYPE} PARENT_SCOPE)
    message(STATUS "******************************** LINUX_TYPE '${TYPE}'")
    if (${CMAKE_MATCH_1} STREQUAL "ubuntu")
        set(UBUNTU TRUE PARENT_SCOPE)
    elseif (${CMAKE_MATCH_1} STREQUAL "centos")
        set(REDHAT TRUE PARENT_SCOPE)
    else()
        message(FATAL_ERROR "****************** not ubuntu nor centos")
    endif()
    message(STATUS "******************************** CMAKE_MATCH_1 ${CMAKE_MATCH_1}")
    string(REGEX MATCH "VERSION_ID=\"([^\"]+)\"" _ ${ID_LONG_STRING})
    message(STATUS "******************************** CMAKE_MATCH_1 ${CMAKE_MATCH_1}")
    set(LINUX_VERSION ${CMAKE_MATCH_1})
    set(LINUX_VERSION ${CMAKE_MATCH_1} PARENT_SCOPE)
endfunction()
