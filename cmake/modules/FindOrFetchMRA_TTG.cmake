# Find an installed MRA-TTG (find_package(mra)) or register it for deferred
# FetchContent.
#
# IMPORTANT: FetchContent_MakeAvailable(MRATTG) must be called AFTER the
# MADworld / MADtensor / MADmra targets are defined so that MRA-TTG's own
# FindOrFetchMADNESS guard fires and does not pull a second copy of MADNESS.
# The MakeAvailable call lives in src/madness/chem/CMakeLists.txt.
#
# On success, the following namespaced targets are available:
#   mra::base   -- INTERFACE library (headers + link deps incl. MADmra, ttg-parsec)
#   mra::host   -- STATIC library (CPU implementation)
#   mra::cuda   -- STATIC library (CUDA implementation, if built)
#   mra::hip    -- STATIC library (HIP  implementation, if built)

if (NOT TARGET mra::base)
  find_package(mra CONFIG QUIET HINTS "${MRA_TTG_ROOT_DIR}")
  if (TARGET mra::base)
    message(STATUS "Found MRA-TTG: mra_CONFIG=${mra_CONFIG}")
  endif()
endif()

if (NOT TARGET mra::base)
  include(FetchContent)
  FetchContent_Declare(
    MRATTG
    GIT_REPOSITORY https://github.com/devreal/MRA-TTG.git
    GIT_TAG        ${MADNESS_TRACKED_MRA_TTG_TAG}
    GIT_SHALLOW    TRUE
  )
  message(STATUS "MRA-TTG: declared fetch from https://github.com/devreal/MRA-TTG.git"
                 " @ ${MADNESS_TRACKED_MRA_TTG_TAG}")
  # FetchContent_MakeAvailable(MRATTG) is deferred to src/madness/chem/CMakeLists.txt
  # so that MADworld/MADtensor/MADmra are already defined when MRA-TTG configures.
endif()
