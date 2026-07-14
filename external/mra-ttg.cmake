if (ENABLE_MRA_TTG)
  # Enable CUDA/HIP at the MADNESS project level before MRA-TTG's FetchContent subproject
  # does so in its own scope. Without this, cmake's internal CUDA compile-rule variables
  # (_CMAKE_CUDA_WHOLE_FLAG, CMAKE_CUDA_COMPILE_OBJECT, …) are never initialised for
  # the parent project, and the generate step fails when MADchem includes .cu sources.
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    check_language(HIP)
    if(CMAKE_HIP_COMPILER)
      enable_language(HIP)
    endif()
  endif()

  if (CMAKE_CXX_STANDARD LESS 20)
    message(FATAL_ERROR "ENABLE_MRA_TTG=ON requires C++20 or later "
            "(CMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}; pass -DCMAKE_CXX_STANDARD=20)")
  endif()

  include(FindOrFetchMRA_TTG)
endif()

add_feature_info(MRA-TTG ENABLE_MRA_TTG
    "TTG-based task-graph MRA implementation (github.com/devreal/MRA-TTG)")
