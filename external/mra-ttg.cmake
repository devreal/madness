if (ENABLE_MRA_TTG)
  if (CMAKE_CXX_STANDARD LESS 20)
    message(FATAL_ERROR "ENABLE_MRA_TTG=ON requires C++20 or later "
            "(CMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}; pass -DCMAKE_CXX_STANDARD=20)")
  endif()

  include(FindOrFetchMRA_TTG)
endif()

add_feature_info(MRA-TTG ENABLE_MRA_TTG
    "TTG-based task-graph MRA implementation (github.com/devreal/MRA-TTG)")
