cmake_minimum_required(VERSION 3.16)

project(Convolution VERSION 1.0.0 LANGUAGES C)

include(CTest)
include(ExternalProject)
include(GNUInstallDirs)
include(FindPackageHandleStandardArgs)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(CONVOLUTION_BUILD_EXAMPLE "Build the example executable" ON)
option(CONVOLUTION_FETCH_FFTW3 "Download and build FFTW3 when it is not available on the system" ON)
option(CONVOLUTION_FORCE_BUNDLED_FFTW3 "Always use the bundled FFTW3 build instead of a system installation" OFF)

set(CONVOLUTION_FFTW3_VERSION 3.3.10 CACHE STRING "FFTW3 version to fetch when using the bundled build")

function(convolution_use_bundled_fftw3 out_target)
	set(fftw3_prefix "${CMAKE_BINARY_DIR}/_deps/fftw3")
	set(fftw3_install_dir "${fftw3_prefix}/install")
	set(fftw3_include_dir "${fftw3_install_dir}/include")
	set(fftw3_library "${fftw3_install_dir}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}fftw3${CMAKE_STATIC_LIBRARY_SUFFIX}")
	file(MAKE_DIRECTORY "${fftw3_include_dir}" "${fftw3_install_dir}/lib")

	if(WIN32)
		message(FATAL_ERROR "Bundled FFTW3 builds are not supported on Windows by this project. Install FFTW3 manually or provide a packaged target.")
	endif()

	ExternalProject_Add(fftw3_external
		URL "https://www.fftw.org/fftw-${CONVOLUTION_FFTW3_VERSION}.tar.gz"
		PREFIX "${fftw3_prefix}"
		DOWNLOAD_EXTRACT_TIMESTAMP FALSE
		CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env
			CC=${CMAKE_C_COMPILER}
			<SOURCE_DIR>/configure
			--prefix=${fftw3_install_dir}
			--includedir=${fftw3_include_dir}
			--libdir=${fftw3_install_dir}/lib
			--disable-shared
			--enable-static
			--with-pic
		BUILD_BYPRODUCTS "${fftw3_library}"
	)

	add_library(FFTW3::FFTW3 STATIC IMPORTED GLOBAL)
	set_target_properties(FFTW3::FFTW3 PROPERTIES
		IMPORTED_LOCATION "${fftw3_library}"
		INTERFACE_INCLUDE_DIRECTORIES "${fftw3_include_dir}"
		INTERFACE_LINK_LIBRARIES m
	)
	add_dependencies(FFTW3::FFTW3 fftw3_external)

	set(${out_target} FFTW3::FFTW3 PARENT_SCOPE)
endfunction()

if(CONVOLUTION_FORCE_BUNDLED_FFTW3)
	convolution_use_bundled_fftw3(convolution_fftw_target)
else()
	find_package(PkgConfig QUIET)
	if(PkgConfig_FOUND)
		pkg_check_modules(FFTW3 QUIET IMPORTED_TARGET fftw3)
	endif()

	if(TARGET PkgConfig::FFTW3)
		set(convolution_fftw_target PkgConfig::FFTW3)
	else()
		find_path(FFTW3_INCLUDE_DIR NAMES fftw3.h)
		find_library(FFTW3_LIBRARY NAMES fftw3)
		find_package_handle_standard_args(
			FFTW3
			REQUIRED_VARS FFTW3_LIBRARY FFTW3_INCLUDE_DIR
		)

		if(FFTW3_FOUND)
			add_library(FFTW3::FFTW3 UNKNOWN IMPORTED)
			set_target_properties(FFTW3::FFTW3 PROPERTIES
				IMPORTED_LOCATION "${FFTW3_LIBRARY}"
				INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIR}"
			)

			set(convolution_fftw_target FFTW3::FFTW3)
		elseif(CONVOLUTION_FETCH_FFTW3)
			message(STATUS "FFTW3 was not found. Falling back to a bundled build.")
			convolution_use_bundled_fftw3(convolution_fftw_target)
		else()
			message(FATAL_ERROR "FFTW3 was not found and CONVOLUTION_FETCH_FFTW3 is OFF.")
		endif()
	endif()
endif()

add_library(conv src/convolution.c)
add_library(Convolution::conv ALIAS conv)

target_compile_features(conv PUBLIC c_std_99)
target_include_directories(conv
	PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
		$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(conv PUBLIC ${convolution_fftw_target})

if(CONVOLUTION_BUILD_EXAMPLE)
	add_executable(${PROJECT_NAME}_example examples/conv_example.c)
	target_include_directories(${PROJECT_NAME}_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
	target_link_libraries(${PROJECT_NAME}_example PRIVATE Convolution::conv)
	
	add_executable(${PROJECT_NAME}_cnn_inference examples/conv_cnn_inference.c)
	target_include_directories(${PROJECT_NAME}_cnn_inference PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
	target_link_libraries(${PROJECT_NAME}_cnn_inference PRIVATE Convolution::conv m)
endif()

if(BUILD_TESTING)
	add_executable(${PROJECT_NAME}_test tests/test_main.c)
	target_include_directories(${PROJECT_NAME}_test PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/src
		${CMAKE_CURRENT_SOURCE_DIR}/tests
	)
	target_link_libraries(${PROJECT_NAME}_test PRIVATE Convolution::conv)

	set(convolution_test_cases
		rank3_scalar_kernel
		rank3_rectangular
		rank4_mixed_values
		rank5_sparse_pattern
		rank6_high_rank
		rank3_zero_tensors
		rank4_identity_kernel
		rank5_identity_kernel
		rank3_large_deterministic
		rank4_large_deterministic
	)

	foreach(case_name IN LISTS convolution_test_cases)
		add_test(NAME convolution_${case_name} COMMAND ${PROJECT_NAME}_test ${case_name})
	endforeach()
endif()

install(
	TARGETS conv
	EXPORT ConvolutionTargets
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(FILES src/convolution.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
	EXPORT ConvolutionTargets
	NAMESPACE Convolution::
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Convolution
)
