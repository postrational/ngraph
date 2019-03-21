# ******************************************************************************
# Copyright 2017-2019 Intel Corporation
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

include(ExternalProject)

find_package(ZLIB REQUIRED)

# Override default LLVM binaries
if(NOT DEFINED LLVM_TARBALL_URL)
    if(APPLE)
        set(NGRAPH_LLVM_VERSION 8.0.0)
        set(LLVM_TARBALL_URL http://releases.llvm.org/${NGRAPH_LLVM_VERSION}/clang+llvm-${NGRAPH_LLVM_VERSION}-x86_64-apple-darwin.tar.xz)
        set(LLVM_SHA1_HASH a5674f2ce5b9ed1b67d92689d319ed3b46d66e29)
    elseif(LINUX)
        set(NGRAPH_LLVM_VERSION 8.0.0)
        if(EXISTS /etc/lsb-release)
            execute_process(COMMAND grep DISTRIB_RELEASE /etc/lsb-release OUTPUT_VARIABLE UBUNTU_VER_LINE)
            string(REGEX MATCH "[0-9.]+" UBUNTU_VER ${UBUNTU_VER_LINE})
            message(STATUS "Ubuntu version: ${UBUNTU_VER} detected.")
            if(UBUNTU_VER MATCHES "16.04")
                set(LLVM_TARBALL_URL http://releases.llvm.org/${NGRAPH_LLVM_VERSION}/clang+llvm-${NGRAPH_LLVM_VERSION}-x86_64-linux-gnu-ubuntu-16.04.tar.xz)
                set(LLVM_SHA1_HASH 2be69be355b012ae206dbc0ea7d84b831d77dc27)
            elseif(UBUNTU_VER MATCHES "18.04")
                set(LLVM_TARBALL_URL http://releases.llvm.org/${NGRAPH_LLVM_VERSION}/clang+llvm-${NGRAPH_LLVM_VERSION}-x86_64-linux-gnu-ubuntu-18.04.tar.xz)
                set(LLVM_SHA1_HASH 6aeb8aa0998d37be67d886b878f27de5e5ccc5e4)
            else()
                message(FATAL_ERROR "No prebuilt LLVM available for Ubuntu ${UBUNTU_VER} on llvm.org, please set LLVM_TARBALL_URL manually.")
            endif()
        else()
            message(FATAL_ERROR "Prebuilt LLVM: Only Ubuntu Linux is supported.")
        endif()
    else()
        message(FATAL_ERROR "Prebuilt LLVM: unsupported OS.")
    endif()
else()
    if(NOT DEFINED LLVM_SHA1_HASH)
        message(FATAL_ERROR "Prebuilt LLVM: please provide LLVM_SHA_HASH.")
    endif()
endif()

ExternalProject_Add(
    ext_llvm
    PREFIX llvm
    URL ${LLVM_TARBALL_URL}
    URL_HASH SHA1=${LLVM_SHA1_HASH}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    DOWNLOAD_NO_PROGRESS TRUE
    EXCLUDE_FROM_ALL TRUE
    )

ExternalProject_Get_Property(ext_llvm SOURCE_DIR)
set(INSTALL_DIR ${SOURCE_DIR})

set(LLVM_LINK_LIBS
    # Do not change order of libraries !!!
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangTooling${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangCodeGen${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangFrontendTool${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangFrontend${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangDriver${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangSerialization${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangParse${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangSema${CMAKE_STATIC_LIBRARY_SUFFIX}
    # Disable static analyzer
    #${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangStaticAnalyzerFrontend${CMAKE_STATIC_LIBRARY_SUFFIX}
    #${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangStaticAnalyzerCheckers${CMAKE_STATIC_LIBRARY_SUFFIX}
    #${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangStaticAnalyzerCore${CMAKE_STATIC_LIBRARY_SUFFIX}
    # 5.0.2 does not have
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangCrossTU${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangAnalysis${CMAKE_STATIC_LIBRARY_SUFFIX}
    # Disabled arc migrate
    #${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangARCMigrate${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangRewriteFrontend${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangEdit${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangAST${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangLex${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}clangBasic${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMLTO${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMPasses${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMObjCARCOpts${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMSymbolize${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMDebugInfoPDB${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMDebugInfoDWARF${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMMIRParser${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMCoverage${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMTableGen${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMDlltoolDriver${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMOrcJIT${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMObjectYAML${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMLibDriver${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMOption${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMX86Disassembler${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMX86AsmParser${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMX86CodeGen${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMGlobalISel${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMSelectionDAG${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMAsmPrinter${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMDebugInfoCodeView${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMDebugInfoMSF${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMX86Desc${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMMCDisassembler${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMX86Info${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMX86AsmPrinter${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMX86Utils${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMMCJIT${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMLineEditor${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMInterpreter${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMExecutionEngine${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMRuntimeDyld${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMCodeGen${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMTarget${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMCoroutines${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMipo${CMAKE_STATIC_LIBRARY_SUFFIX}
    # 5.0.2 does not have
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMAggressiveInstCombine${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMInstrumentation${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMVectorize${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMScalarOpts${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMLinker${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMIRReader${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMAsmParser${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMInstCombine${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMTransformUtils${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMBitWriter${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMAnalysis${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMProfileData${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMObject${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMMCParser${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMMC${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMBitReader${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMCore${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMBinaryFormat${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMSupport${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMDemangle${CMAKE_STATIC_LIBRARY_SUFFIX}
)

if(APPLE)
    set(LLVM_LINK_LIBS ${LLVM_LINK_LIBS} curses z m)
else()
    set(LLVM_LINK_LIBS ${LLVM_LINK_LIBS} tinfo z m)
endif()

add_library(libllvm INTERFACE)
add_dependencies(libllvm ext_llvm)
target_include_directories(libllvm SYSTEM INTERFACE ${INSTALL_DIR}/include)
target_link_libraries(libllvm INTERFACE ${LLVM_LINK_LIBS})
