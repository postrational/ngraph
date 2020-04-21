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
    message(STATUS "******************************** ID_LONG_STRING ${ID_LONG_STRING}")
    string(REGEX MATCH "ID=\"([^\"]+)\"" \\1 _ ${ID_LONG_STRING})
    message(STATUS "******************************** CMAKE_MATCH_1 ${CMAKE_MATCH_1}")

    # set(DEBIAN FALSE PARENT_SCOPE)
    # set(REDHAT FALSE PARENT_SCOPE)
    # if(EXISTS /etc/debian_version)
    #     set(DEBIAN TRUE PARENT_SCOPE)
    # elseif(EXISTS /etc/redhat-release)
    #     set(REDHAT TRUE PARENT_SCOPE)
    # endif()
endfunction()
